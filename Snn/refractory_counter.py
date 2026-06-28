"""
refractory_counter.py
=======================
Standalone refractory-period counter module for the SNN core in the
PQC-SNN SoC (FB_SNN — Neuron Processing Element / Refractory Counter).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Implements the per-neuron refractory countdown logic as an isolated,     ║
║ RTL-matching block: load-on-spike, decrement-per-tick, gate signal out.  ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Single-neuron refractory counter (load / decrement / expire)
  - Vectorized refractory bank for an entire neuron layer
  - Absolute vs. relative refractory modes
  - Refractory-gated input masking (zero out I_syn while refractory)
  - Statistics: time-in-refractory, duty cycle, per-neuron history

Context (per architecture diagram, "NEURON PROCESSING ELEMENT (PE)" box):
  The PE lists "Refractory Counter" as a distinct sub-block alongside
  Membrane/Leak/IF and Spike Generation. In `lif_neuron.py` the refractory
  behavior is embedded inside the neuron's step() method; this module
  extracts it as a standalone, independently testable/synthesizable
  unit — matching how it would appear as its own block in RTL
  (`refractory_counter.sv`), separate from the membrane integrator.

Refractory modes:
  ABSOLUTE — neuron cannot spike at all while counter > 0 (this module's
             default; matches the gating used in lif_neuron.py).
  RELATIVE — neuron CAN spike while counter > 0, but its effective
             threshold is raised (modeled here via a threshold-scaling
             helper) — included for completeness / future RTL variants.

Counter semantics:
  load(R)      : counter := R           (called on spike emission)
  tick()       : if counter > 0: counter -= 1
  is_active()  : counter > 0            (True = neuron gated)
  gate(value)  : returns 0 if active else value (masks synaptic input)

Matches refractory_counter.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : Gerstner & Kistler, "Spiking Neuron Models" (refractoriness)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ──────────────────────────────────────────────
# 1.  PARAMETERS
# ──────────────────────────────────────────────

#: Default refractory period length (ticks)
DEFAULT_REFRACTORY_TICKS: int = 2

#: Default relative-refractory threshold scaling factor (applied while
#: counter > 0, RELATIVE mode only) — neuron needs proportionally more
#: input to fire during partial recovery.
DEFAULT_RELATIVE_SCALE: float = 1.5


class RefractoryMode(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


# ──────────────────────────────────────────────
# 2.  SINGLE-NEURON REFRACTORY COUNTER
# ──────────────────────────────────────────────

@dataclass
class RefractoryCounter:
    """
    Refractory countdown logic for a single neuron.
    """
    period_ticks: int = DEFAULT_REFRACTORY_TICKS
    mode: RefractoryMode = RefractoryMode.ABSOLUTE
    relative_scale: float = DEFAULT_RELATIVE_SCALE

    counter: int = field(init=False, default=0)
    total_ticks_active: int = field(init=False, default=0)
    total_ticks_elapsed: int = field(init=False, default=0)
    load_count: int = field(init=False, default=0)

    def load(self, period_ticks: Optional[int] = None) -> None:
        """
        Load the counter (called when the neuron emits a spike).

        Parameters
        ----------
        period_ticks : int, optional
            Override the configured refractory period for this load.
            If None, uses self.period_ticks.
        """
        self.counter = period_ticks if period_ticks is not None else self.period_ticks
        self.load_count += 1

    def tick(self) -> None:
        """Advance the counter by one simulation tick (decrement if active)."""
        self.total_ticks_elapsed += 1
        if self.counter > 0:
            self.counter -= 1
            self.total_ticks_active += 1

    def is_active(self) -> bool:
        """True if the neuron is currently within its refractory period."""
        return self.counter > 0

    def gate(self, i_syn: float) -> float:
        """
        Apply refractory gating to a synaptic input value.

        In ABSOLUTE mode, returns 0.0 while active (input fully blocked).
        In RELATIVE mode, returns the input unchanged (caller should use
        `threshold_scale()` to raise the firing threshold instead of
        blocking input outright).

        Parameters
        ----------
        i_syn : float
            Raw synaptic input current for this tick.

        Returns
        -------
        float
            Gated synaptic input.
        """
        if self.mode == RefractoryMode.ABSOLUTE and self.is_active():
            return 0.0
        return i_syn

    def threshold_scale(self) -> float:
        """
        Compute the threshold scaling factor for RELATIVE refractory mode.

        Returns
        -------
        float
            1.0 if not active or mode is ABSOLUTE; otherwise a value > 1.0
            proportional to how recently the neuron spiked (linearly
            decaying back to 1.0 as the counter approaches zero).
        """
        if self.mode != RefractoryMode.RELATIVE or not self.is_active():
            return 1.0

        if self.period_ticks <= 0:
            return 1.0

        # Linear decay from relative_scale (just after spike) to 1.0
        # (just before counter expires)
        fraction = self.counter / self.period_ticks
        return 1.0 + (self.relative_scale - 1.0) * fraction

    def duty_cycle(self) -> float:
        """
        Fraction of elapsed ticks spent in the refractory state.

        Returns
        -------
        float
            total_ticks_active / total_ticks_elapsed, or 0.0 if no ticks
            have elapsed yet.
        """
        if self.total_ticks_elapsed == 0:
            return 0.0
        return self.total_ticks_active / self.total_ticks_elapsed

    def reset(self) -> None:
        """Clear the counter immediately (does not reset statistics)."""
        self.counter = 0

    def reset_stats(self) -> None:
        """Clear accumulated statistics (does not affect current counter)."""
        self.total_ticks_active = 0
        self.total_ticks_elapsed = 0
        self.load_count = 0


# ──────────────────────────────────────────────
# 3.  VECTORIZED REFRACTORY BANK (PER LAYER)
# ──────────────────────────────────────────────

class RefractoryBank:
    """
    A bank of independent RefractoryCounters, one per neuron in a layer —
    mirrors how the refractory logic would be instantiated N-wide in RTL.
    """

    def __init__(
        self,
        n_neurons: int,
        period_ticks: int = DEFAULT_REFRACTORY_TICKS,
        mode: RefractoryMode = RefractoryMode.ABSOLUTE,
    ):
        """
        Initialize a bank of refractory counters.

        Parameters
        ----------
        n_neurons : int
            Number of neurons (counters) in the bank.
        period_ticks : int
            Default refractory period for all counters in the bank.
        mode : RefractoryMode
            ABSOLUTE or RELATIVE, applied to all counters.
        """
        self.n_neurons = n_neurons
        self.counters: List[RefractoryCounter] = [
            RefractoryCounter(period_ticks=period_ticks, mode=mode)
            for _ in range(n_neurons)
        ]

    def load(self, spike_vector: List[int]) -> None:
        """
        Load counters for neurons that just spiked.

        Parameters
        ----------
        spike_vector : List[int]
            Binary spike vector (1 = neuron spiked this tick), length n_neurons.
        """
        if len(spike_vector) != self.n_neurons:
            raise ValueError(
                f"spike_vector length {len(spike_vector)} != n_neurons {self.n_neurons}"
            )
        for i, s in enumerate(spike_vector):
            if s == 1:
                self.counters[i].load()

    def tick(self) -> None:
        """Advance every counter in the bank by one tick."""
        for c in self.counters:
            c.tick()

    def is_active_vector(self) -> List[bool]:
        """Return active/refractory status for every neuron in the bank."""
        return [c.is_active() for c in self.counters]

    def gate_vector(self, i_syn_vector: List[float]) -> List[float]:
        """
        Apply refractory gating to a synaptic input vector.

        Parameters
        ----------
        i_syn_vector : List[float]
            Raw synaptic input per neuron, length n_neurons.

        Returns
        -------
        List[float]
            Gated synaptic input per neuron.
        """
        if len(i_syn_vector) != self.n_neurons:
            raise ValueError(
                f"i_syn_vector length {len(i_syn_vector)} != n_neurons {self.n_neurons}"
            )
        return [c.gate(i) for c, i in zip(self.counters, i_syn_vector)]

    def step(self, spike_vector: List[int], i_syn_vector: List[float]) -> List[float]:
        """
        Combined single-tick operation: load on this tick's spikes,
        gate the *next* tick's input, then advance the counters.

        Typical usage in a neuron layer's step():
          1. neuron layer computes raw I_syn
          2. bank.step(prev_spikes, raw_i_syn) → gated_i_syn
          3. gated_i_syn fed into LIF membrane update
          4. resulting spikes passed to bank.load() (or via this step())

        Parameters
        ----------
        spike_vector : List[int]
            Spikes emitted on the *previous* tick (to load counters now).
        i_syn_vector : List[float]
            Raw synaptic input for the *current* tick (to be gated).

        Returns
        -------
        List[float]
            Gated synaptic input for the current tick.
        """
        self.load(spike_vector)
        gated = self.gate_vector(i_syn_vector)
        self.tick()
        return gated

    def duty_cycles(self) -> List[float]:
        """Return duty cycle (fraction of time refractory) per neuron."""
        return [c.duty_cycle() for c in self.counters]

    def reset(self) -> None:
        """Clear all counters in the bank immediately."""
        for c in self.counters:
            c.reset()

    def stats(self) -> dict:
        """Return bank-level aggregate statistics."""
        duty = self.duty_cycles()
        active = self.is_active_vector()
        return {
            "n_neurons": self.n_neurons,
            "currently_active": sum(active),
            "avg_duty_cycle": sum(duty) / len(duty) if duty else 0.0,
            "max_duty_cycle": max(duty) if duty else 0.0,
            "total_loads": sum(c.load_count for c in self.counters),
        }


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("refractory_counter.py  —  self-test")
    print("=" * 55)

    # Test 1: Basic load/tick/expire cycle
    print("\n[ Basic load/tick/expire cycle ]")
    c = RefractoryCounter(period_ticks=3)
    assert not c.is_active()
    c.load()
    assert c.is_active()
    assert c.counter == 3
    c.tick(); c.tick(); c.tick()
    assert not c.is_active()
    assert c.counter == 0
    print(f"  ✓ Loaded with 3 ticks, expired after exactly 3 ticks")

    # Test 2: Absolute mode gates input to zero
    print("\n[ Absolute mode input gating ]")
    c2 = RefractoryCounter(period_ticks=2, mode=RefractoryMode.ABSOLUTE)
    c2.load()
    gated = c2.gate(5.0)
    assert gated == 0.0
    print(f"  ✓ I_syn=5.0 gated to {gated} while refractory (ABSOLUTE)")

    c2.tick(); c2.tick()
    gated_after = c2.gate(5.0)
    assert gated_after == 5.0
    print(f"  ✓ I_syn=5.0 passes through unchanged ({gated_after}) after expiry")

    # Test 3: Relative mode does not block input but raises threshold
    print("\n[ Relative mode threshold scaling ]")
    c3 = RefractoryCounter(period_ticks=4, mode=RefractoryMode.RELATIVE, relative_scale=2.0)
    c3.load()
    gated_rel = c3.gate(5.0)
    assert gated_rel == 5.0  # input NOT blocked in relative mode
    scale_at_load = c3.threshold_scale()
    assert abs(scale_at_load - 2.0) < 1e-9  # full scale right after spike
    c3.tick(); c3.tick()
    scale_mid = c3.threshold_scale()
    assert 1.0 < scale_mid < 2.0  # decaying toward 1.0
    print(f"  ✓ Relative mode: input unblocked, threshold scale {scale_at_load:.2f} → {scale_mid:.2f}")

    # Test 4: Threshold scale returns 1.0 after full recovery
    print("\n[ Threshold scale after recovery ]")
    c3.tick(); c3.tick()
    assert not c3.is_active()
    assert c3.threshold_scale() == 1.0
    print(f"  ✓ Threshold scale returns to 1.0 once fully recovered")

    # Test 5: Re-load while active extends the period
    print("\n[ Re-load while active ]")
    c4 = RefractoryCounter(period_ticks=3)
    c4.load()
    c4.tick()  # counter now 2
    c4.load()  # re-triggered (e.g. another spike) — reset to full period
    assert c4.counter == 3
    print(f"  ✓ Re-load while active resets counter to full period (3)")

    # Test 6: Duty cycle calculation
    print("\n[ Duty cycle calculation ]")
    c5 = RefractoryCounter(period_ticks=2)
    for _ in range(10):
        c5.tick()
    c5.load()
    for _ in range(10):
        c5.tick()
    # 2 active ticks out of 20 total elapsed (load doesn't count as elapsed)
    duty = c5.duty_cycle()
    assert abs(duty - (2 / 20)) < 1e-9
    print(f"  ✓ Duty cycle = {duty:.3f} (2 active / 20 elapsed ticks)")

    # Test 7: RefractoryBank — load from spike vector
    print("\n[ RefractoryBank load from spike vector ]")
    bank = RefractoryBank(n_neurons=4, period_ticks=2)
    bank.load([1, 0, 1, 0])
    active = bank.is_active_vector()
    assert active == [True, False, True, False]
    print(f"  ✓ Bank active vector: {active}")

    # Test 8: RefractoryBank gating
    print("\n[ RefractoryBank input gating ]")
    gated_vec = bank.gate_vector([5.0, 5.0, 5.0, 5.0])
    assert gated_vec == [0.0, 5.0, 0.0, 5.0]
    print(f"  ✓ Gated vector: {gated_vec}")

    # Test 9: RefractoryBank combined step()
    print("\n[ RefractoryBank combined step() ]")
    bank2 = RefractoryBank(n_neurons=3, period_ticks=2)
    prev_spikes = [1, 0, 0]
    raw_input = [10.0, 10.0, 10.0]
    gated = bank2.step(prev_spikes, raw_input)
    assert gated == [0.0, 10.0, 10.0]
    print(f"  ✓ step() loaded neuron 0, gated its input: {gated}")

    # Test 10: Dimension validation
    print("\n[ Dimension validation ]")
    bank3 = RefractoryBank(n_neurons=3)
    try:
        bank3.load([1, 0])  # wrong length
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ Mismatched spike vector length correctly rejected")

    # Test 11: Bank statistics
    print("\n[ Bank statistics ]")
    bank4 = RefractoryBank(n_neurons=5, period_ticks=3)
    for i in range(5):
        bank4.load([1 if j == i else 0 for j in range(5)])
        for _ in range(5):
            bank4.tick()
    stats = bank4.stats()
    print(f"  Currently active: {stats['currently_active']}")
    print(f"  Avg duty cycle: {stats['avg_duty_cycle']:.3f}")
    print(f"  Total loads: {stats['total_loads']}")
    assert stats["total_loads"] == 5
    print(f"  ✓ Statistics tracked correctly")

    # Test 12: Reset clears active state but not stats
    print("\n[ Reset behavior ]")
    bank4.reset()
    assert all(not c.is_active() for c in bank4.counters)
    assert bank4.stats()["total_loads"] == 5  # stats survive reset()
    print(f"  ✓ reset() clears active counters, preserves accumulated stats")

    # Test 13: Integration-style scenario — bank matches manual LIF gating
    print("\n[ Integration-style refractory gating sequence ]")
    bank5 = RefractoryBank(n_neurons=1, period_ticks=2)
    results = []
    spikes_this_tick = [0]
    for tick in range(6):
        gated_input = bank5.gate_vector([1.0])
        results.append(gated_input[0])
        if tick == 0:
            bank5.load([1])  # force a spike at tick 0
        bank5.tick()
    # tick0: gate before load → not yet active → 1.0; load() then sets counter=2
    # tick0's tick() call decrements counter 2→1 (still active)
    # tick1: gate sees counter=1 (active) → 0.0; tick() decrements 1→0
    # tick2: gate sees counter=0 (expired) → 1.0
    # tick3,4,5: stays expired → 1.0
    assert results == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    print(f"  ✓ Gating sequence over 6 ticks: {results}")

    print("\n  All checks passed.\n")
