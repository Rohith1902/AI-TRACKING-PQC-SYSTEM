"""
spike_generator.py
====================
Spike generation and output formatting for the SNN core in the
PQC-SNN SoC (FB_SNN — Neuron Processing Element / Spike Generation).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Converts membrane-potential threshold crossings into the SoC's spike     ║
║ event format (spike_out, spike_valid, layer_done) for downstream blocks. ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Threshold-crossing → spike decision (the "Spike Generation" sub-block
    of NEURON PROCESSING ELEMENT, separated out from lif_neuron.py's
    integrated step() for independent testing / RTL mapping)
  - SoC-level spike event packing (per diagram's SNN_CORE output ports:
    spike_out[15:0], spike_valid, layer_done, event_in[15:0])
  - Address-Event Representation (AER) encoding: (neuron_id, timestamp)
    pairs instead of dense per-tick vectors — matches "EVENT ROUTER (AER)"
    block (Address-Event Map, FIFO/Ring Buffer, Timestamping)
  - Dense ↔ sparse (AER) spike format conversion
  - Spike train merging from multiple neuron banks (e.g. combining
    hidden-layer outputs before the output-layer event router)

Context (per architecture diagram, "SNN CORE" boxes):
  - SNN_CORE output ports: spike_out[15:0], spike_valid, layer_done,
    event_in[15:0]  → this module produces those signals from raw
    threshold-crossing booleans.
  - EVENT ROUTER (AER): Address-Event Map, FIFO/Ring Buffer, Timestamping,
    Back-pressure Ctrl → this module's AEREvent / encode_aer / decode_aer
    are the golden-model equivalent of that block's core encoding logic.

Spike event formats:
  DENSE : List[int] of length N, 1 = spike at index i, 0 = no spike.
          (what lif_neuron.py / neuro_array.py produce internally)
  AER   : List[AEREvent], each carrying (neuron_id, timestamp) — only
          neurons that actually fired are represented, which is the
          efficient on-chip transport format for sparse spiking activity.

Matches spike_generator.sv (hardware RTL reference, within NEURON
PROCESSING ELEMENT's "Spike Generation" sub-block and feeding the
EVENT ROUTER (AER) block).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : AEGIS-NEURO architecture diagram, §3 NEUROMORPHIC SNN SUBSYSTEM
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────
# 1.  PARAMETERS
# ──────────────────────────────────────────────

#: Width of the spike_out bus per diagram (spike_out[15:0])
SPIKE_BUS_WIDTH: int = 16

#: Maximum FIFO/ring-buffer depth for AER event queue (golden-model cap)
AER_FIFO_DEPTH: int = 256


# ──────────────────────────────────────────────
# 2.  THRESHOLD-CROSSING SPIKE DECISION
# ──────────────────────────────────────────────

def spike_decision(v_mem: float, v_threshold: float) -> int:
    """
    Pure threshold-crossing spike decision (isolated from membrane update).

    Parameters
    ----------
    v_mem : float
        Current membrane potential (post-integration, pre-reset).
    v_threshold : float
        Firing threshold.

    Returns
    -------
    int
        1 if v_mem >= v_threshold, else 0.
    """
    return 1 if v_mem >= v_threshold else 0


def spike_decision_vector(v_mem_vec: List[float], v_threshold: float) -> List[int]:
    """
    Vectorized threshold-crossing decision across a layer of neurons.

    Parameters
    ----------
    v_mem_vec : List[float]
        Membrane potentials for each neuron in the layer.
    v_threshold : float
        Firing threshold (shared across the layer).

    Returns
    -------
    List[int]
        Binary spike vector.
    """
    return [spike_decision(v, v_threshold) for v in v_mem_vec]


# ──────────────────────────────────────────────
# 3.  DENSE SPIKE BUS PACKING (spike_out[15:0])
# ──────────────────────────────────────────────

def pack_spike_bus(spikes: List[int], bus_width: int = SPIKE_BUS_WIDTH) -> List[int]:
    """
    Pack a dense spike vector into fixed-width spike_out bus words.

    Mirrors the hardware's spike_out[15:0] port: if there are more
    neurons than bus_width, output is split across multiple bus words
    (one per "layer_done" cycle), matching a time-multiplexed readout.

    Parameters
    ----------
    spikes : List[int]
        Dense binary spike vector (length = n_neurons).
    bus_width : int
        Width of the output bus in bits (default 16, per diagram).

    Returns
    -------
    List[int]
        List of bus words (integers), each encoding up to `bus_width`
        neurons' spikes as a bitfield (bit i = neuron i within that word).
    """
    words = []
    for start in range(0, len(spikes), bus_width):
        chunk = spikes[start:start + bus_width]
        word = 0
        for i, s in enumerate(chunk):
            if s:
                word |= (1 << i)
        words.append(word)
    return words


def unpack_spike_bus(
    words: List[int], n_neurons: int, bus_width: int = SPIKE_BUS_WIDTH
) -> List[int]:
    """
    Unpack spike_out bus words back into a dense spike vector.

    Parameters
    ----------
    words : List[int]
        Bus words as produced by pack_spike_bus().
    n_neurons : int
        Total number of neurons to reconstruct (handles a final
        partial word correctly).
    bus_width : int
        Width of the output bus in bits.

    Returns
    -------
    List[int]
        Dense binary spike vector of length n_neurons.
    """
    spikes = []
    for word in words:
        for i in range(bus_width):
            if len(spikes) >= n_neurons:
                break
            spikes.append((word >> i) & 1)
    while len(spikes) < n_neurons:
        spikes.append(0)
    return spikes[:n_neurons]


# ──────────────────────────────────────────────
# 4.  ADDRESS-EVENT REPRESENTATION (AER)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class AEREvent:
    """A single Address-Event Representation spike event."""
    neuron_id: int
    timestamp: int


def encode_aer(spikes: List[int], timestamp: int) -> List[AEREvent]:
    """
    Convert a dense spike vector into sparse AER events for one tick.

    Parameters
    ----------
    spikes : List[int]
        Dense binary spike vector for this tick.
    timestamp : int
        Current simulation tick / timestamp to attach to each event.

    Returns
    -------
    List[AEREvent]
        One event per neuron that spiked (empty list if none fired).
    """
    return [
        AEREvent(neuron_id=i, timestamp=timestamp)
        for i, s in enumerate(spikes) if s == 1
    ]


def decode_aer(events: List[AEREvent], n_neurons: int, timestamp: int) -> List[int]:
    """
    Convert AER events back into a dense spike vector for a given timestamp.

    Parameters
    ----------
    events : List[AEREvent]
        AER events (may span multiple timestamps; only matching ones used).
    n_neurons : int
        Total neuron count for the dense vector.
    timestamp : int
        Which timestamp's events to reconstruct.

    Returns
    -------
    List[int]
        Dense binary spike vector of length n_neurons for that timestamp.
    """
    dense = [0] * n_neurons
    for ev in events:
        if ev.timestamp == timestamp and 0 <= ev.neuron_id < n_neurons:
            dense[ev.neuron_id] = 1
    return dense


def encode_aer_sequence(spike_matrix: List[List[int]]) -> List[AEREvent]:
    """
    Encode a full multi-tick spike matrix into one flat AER event stream.

    Parameters
    ----------
    spike_matrix : List[List[int]]
        One dense spike vector per tick (outer index = timestamp).

    Returns
    -------
    List[AEREvent]
        Flattened AER event stream across all ticks.
    """
    events: List[AEREvent] = []
    for t, spikes in enumerate(spike_matrix):
        events.extend(encode_aer(spikes, timestamp=t))
    return events


# ──────────────────────────────────────────────
# 5.  AER EVENT FIFO (RING BUFFER MODEL)
# ──────────────────────────────────────────────

class AEREventFIFO:
    """
    Models the EVENT ROUTER's FIFO / Ring Buffer with back-pressure
    control, queuing AER events for downstream consumption (e.g. by
    the alert/output subsystem).
    """

    def __init__(self, depth: int = AER_FIFO_DEPTH):
        """
        Initialize the event FIFO.

        Parameters
        ----------
        depth : int
            Maximum number of events the FIFO can hold before back-pressure
            (push() returns False) is applied.
        """
        self.depth = depth
        self.queue: List[AEREvent] = []
        self.dropped_count: int = 0
        self.pushed_count: int = 0
        self.popped_count: int = 0

    def push(self, event: AEREvent) -> bool:
        """
        Push a single event into the FIFO.

        Parameters
        ----------
        event : AEREvent
            Event to enqueue.

        Returns
        -------
        bool
            True if accepted, False if the FIFO was full (back-pressure;
            event is dropped and dropped_count incremented).
        """
        if len(self.queue) >= self.depth:
            self.dropped_count += 1
            return False
        self.queue.append(event)
        self.pushed_count += 1
        return True

    def push_many(self, events: List[AEREvent]) -> int:
        """
        Push multiple events, respecting back-pressure for each.

        Parameters
        ----------
        events : List[AEREvent]
            Events to enqueue.

        Returns
        -------
        int
            Number of events successfully accepted.
        """
        return sum(1 for e in events if self.push(e))

    def pop(self) -> Optional[AEREvent]:
        """
        Pop the oldest event from the FIFO (FIFO ordering).

        Returns
        -------
        AEREvent or None
            The oldest queued event, or None if the FIFO is empty.
        """
        if not self.queue:
            return None
        self.popped_count += 1
        return self.queue.pop(0)

    def pop_all(self) -> List[AEREvent]:
        """Drain and return all currently queued events, in FIFO order."""
        drained = list(self.queue)
        self.popped_count += len(drained)
        self.queue.clear()
        return drained

    def is_full(self) -> bool:
        """True if the FIFO is at capacity (back-pressure active)."""
        return len(self.queue) >= self.depth

    def occupancy(self) -> int:
        """Current number of queued events."""
        return len(self.queue)

    def stats(self) -> dict:
        """Return FIFO statistics."""
        return {
            "depth": self.depth,
            "occupancy": self.occupancy(),
            "pushed": self.pushed_count,
            "popped": self.popped_count,
            "dropped": self.dropped_count,
        }


# ──────────────────────────────────────────────
# 6.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("spike_generator.py  —  self-test")
    print("=" * 55)

    # Test 1: Threshold-crossing decision
    print("\n[ Threshold-crossing spike decision ]")
    assert spike_decision(1.2, 1.0) == 1
    assert spike_decision(0.8, 1.0) == 0
    assert spike_decision(1.0, 1.0) == 1  # exactly at threshold fires
    print(f"  ✓ Above/below/exactly-at threshold decisions correct")

    # Test 2: Vectorized decision
    print("\n[ Vectorized threshold decision ]")
    v_vec = [1.2, 0.5, 1.0, 0.0, 2.5]
    spikes = spike_decision_vector(v_vec, v_threshold=1.0)
    assert spikes == [1, 0, 1, 0, 1]
    print(f"  ✓ Membrane vector {v_vec} → spikes {spikes}")

    # Test 3: Spike bus packing (single word, fits in 16 bits)
    print("\n[ Spike bus packing — single word ]")
    spikes_small = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    words = pack_spike_bus(spikes_small, bus_width=16)
    assert len(words) == 1
    assert words[0] == 0b1101  # bits 0,2,3 set
    print(f"  ✓ 16 spikes → 1 bus word: {bin(words[0])}")

    # Test 4: Spike bus packing — multi-word (more neurons than bus width)
    print("\n[ Spike bus packing — multi-word ]")
    spikes_big = [1 if i % 3 == 0 else 0 for i in range(40)]  # 40 neurons
    words_big = pack_spike_bus(spikes_big, bus_width=16)
    assert len(words_big) == 3  # ceil(40/16) = 3
    print(f"  ✓ 40 spikes → {len(words_big)} bus words")

    # Test 5: Pack/unpack roundtrip
    print("\n[ Pack/unpack roundtrip ]")
    unpacked = unpack_spike_bus(words_big, n_neurons=40, bus_width=16)
    assert unpacked == spikes_big
    print(f"  ✓ 40-neuron spike vector roundtrips through bus packing")

    # Test 6: AER encoding — sparse representation
    print("\n[ AER encoding ]")
    dense_spikes = [0, 1, 0, 0, 1, 0, 0, 0, 1]
    events = encode_aer(dense_spikes, timestamp=42)
    assert len(events) == 3  # only 3 neurons fired
    assert events[0] == AEREvent(neuron_id=1, timestamp=42)
    assert events[1] == AEREvent(neuron_id=4, timestamp=42)
    assert events[2] == AEREvent(neuron_id=8, timestamp=42)
    print(f"  ✓ 9-neuron dense vector (3 firing) → {len(events)} AER events")

    # Test 7: AER decoding roundtrip
    print("\n[ AER decode roundtrip ]")
    decoded = decode_aer(events, n_neurons=9, timestamp=42)
    assert decoded == dense_spikes
    print(f"  ✓ AER events decode back to identical dense vector")

    # Test 8: AER sequence encoding (multi-tick)
    print("\n[ AER sequence encoding ]")
    spike_matrix = [
        [1, 0, 0],  # t=0
        [0, 0, 0],  # t=1 — no spikes, no events
        [0, 1, 1],  # t=2
    ]
    seq_events = encode_aer_sequence(spike_matrix)
    assert len(seq_events) == 3  # 1 + 0 + 2 events total
    timestamps = sorted(set(e.timestamp for e in seq_events))
    assert timestamps == [0, 2]  # t=1 contributes nothing
    print(f"  ✓ 3-tick sequence → {len(seq_events)} sparse events "
          f"(silent tick correctly produces none)")

    # Test 9: AER FIFO basic push/pop
    print("\n[ AER FIFO push/pop ]")
    fifo = AEREventFIFO(depth=4)
    e1 = AEREvent(0, 1)
    e2 = AEREvent(1, 1)
    assert fifo.push(e1) is True
    assert fifo.push(e2) is True
    assert fifo.occupancy() == 2
    popped = fifo.pop()
    assert popped == e1  # FIFO order
    print(f"  ✓ Pushed 2, popped oldest first (FIFO order preserved)")

    # Test 10: AER FIFO back-pressure
    print("\n[ AER FIFO back-pressure ]")
    fifo2 = AEREventFIFO(depth=2)
    fifo2.push(AEREvent(0, 1))
    fifo2.push(AEREvent(1, 1))
    assert fifo2.is_full()
    rejected = fifo2.push(AEREvent(2, 1))  # should be dropped
    assert rejected is False
    stats = fifo2.stats()
    assert stats["dropped"] == 1
    print(f"  ✓ FIFO at depth=2 correctly applies back-pressure, dropped={stats['dropped']}")

    # Test 11: push_many and pop_all
    print("\n[ Batch push_many / pop_all ]")
    fifo3 = AEREventFIFO(depth=10)
    batch = [AEREvent(i, 5) for i in range(5)]
    accepted = fifo3.push_many(batch)
    assert accepted == 5
    drained = fifo3.pop_all()
    assert len(drained) == 5
    assert fifo3.occupancy() == 0
    print(f"  ✓ Batch pushed {accepted} events, drained all {len(drained)} via pop_all")

    # Test 12: Full pipeline — neuron membrane → bus → AER → FIFO
    print("\n[ Full pipeline integration ]")
    v_mem_layer = [1.5, 0.2, 0.9, 1.1, 0.0]
    spikes_layer = spike_decision_vector(v_mem_layer, v_threshold=1.0)
    bus_words = pack_spike_bus(spikes_layer, bus_width=16)
    aer_events = encode_aer(spikes_layer, timestamp=100)
    pipeline_fifo = AEREventFIFO(depth=16)
    pipeline_fifo.push_many(aer_events)

    assert spikes_layer == [1, 0, 0, 1, 0]
    assert len(bus_words) == 1
    assert pipeline_fifo.occupancy() == 2  # neurons 0 and 3 fired
    print(f"  ✓ Membrane {v_mem_layer} → spikes {spikes_layer} → "
          f"bus {bin(bus_words[0])} → {pipeline_fifo.occupancy()} AER events queued")

    print("\n  All checks passed.\n")
