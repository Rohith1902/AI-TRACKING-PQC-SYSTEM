"""
neuro_array.py
===============
Multi-layer neuron array (Input → Hidden → Output) for the SNN core in the
PQC-SNN SoC (FB_SNN — SNN_CORE / Neuron Processing Element array).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Assembles LIF neuron layers and synaptic weight matrices into a full      ║
║ feed-forward spiking network (N→M→K), driving spikes layer-to-layer.     ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Layer topology management (N input → M hidden → K output, per diagram)
  - Synaptic weight matrix per layer-to-layer connection
  - Forward spike propagation: spikes(layer L) · W(L→L+1) → I_syn(layer L+1)
  - Full network step (one simulation tick through all layers)
  - Full network run (multi-tick simulation over an input spike train)
  - Weight matrix initialization (uniform, normal, or explicit)
  - Per-layer / per-neuron output spike recording for STDP and readout

Context (per architecture diagram, "NEURON PROCESSING ELEMENT (PE)" and
"SNN CORE (SPIKING NEURAL NETWORK)" boxes):
  Input Neurons (N) → Spiking Neurons (M, hidden) → Output Neurons (K)
  Default topology in AEGIS-NEURO: 256 → 128 → 64 → 32 → 8 (multi-hidden)
  This module supports an arbitrary list of layer sizes, so the same
  code models any stage of that topology, or the full chain at once.

Synaptic current computation (per tick, per layer transition L→L+1):
  I_syn[j] = sum_i( spike[i] * W[i][j] )   for all source neurons i,
                                             all destination neurons j
  This vector I_syn then drives LIFNeuronLayer.step() for layer L+1.

Weight matrix convention:
  W[i][j] = synaptic weight from source neuron i (layer L)
            to destination neuron j (layer L+1)
  Shape: len(layer_L) x len(layer_L+1)

Matches neuro_array.sv / SNN_CORE (hardware RTL reference) — this is the
golden-model equivalent of the "SYNAPSE / WEIGHT MEMORY" + "NEURON
PROCESSING ELEMENT" interaction across the full array.

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : Gerstner & Kistler, "Spiking Neuron Models"; AEGIS-NEURO architecture
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional

from lif_neuron import LIFNeuronLayer, DEFAULT_REFRACTORY_TICKS, DEFAULT_TAU_MEM_MS, DEFAULT_DT_MS

# ──────────────────────────────────────────────
# 1.  WEIGHT MATRIX UTILITIES
# ──────────────────────────────────────────────

def init_weights_uniform(
    n_src: int, n_dst: int, low: float = 0.0, high: float = 0.5, seed: Optional[int] = None
) -> List[List[float]]:
    """
    Initialize a weight matrix with uniform random values.

    Parameters
    ----------
    n_src : int
        Number of source-layer neurons (rows).
    n_dst : int
        Number of destination-layer neurons (columns).
    low : float
        Minimum weight value.
    high : float
        Maximum weight value.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    List[List[float]]
        n_src x n_dst weight matrix.
    """
    rng = random.Random(seed)
    return [[rng.uniform(low, high) for _ in range(n_dst)] for _ in range(n_src)]


def init_weights_normal(
    n_src: int, n_dst: int, mean: float = 0.2, std: float = 0.05, seed: Optional[int] = None
) -> List[List[float]]:
    """
    Initialize a weight matrix with Gaussian random values (clamped >= 0).

    Parameters
    ----------
    n_src : int
        Number of source-layer neurons.
    n_dst : int
        Number of destination-layer neurons.
    mean : float
        Mean weight value.
    std : float
        Standard deviation.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    List[List[float]]
        n_src x n_dst weight matrix, non-negative.
    """
    rng = random.Random(seed)
    return [
        [max(0.0, rng.gauss(mean, std)) for _ in range(n_dst)]
        for _ in range(n_src)
    ]


def matvec_spike_current(spikes: List[int], weights: List[List[float]]) -> List[float]:
    """
    Compute synaptic current vector from a spike vector and weight matrix.

    I_syn[j] = sum_i( spikes[i] * weights[i][j] )

    Parameters
    ----------
    spikes : List[int]
        Binary spike vector from the source layer (length n_src).
    weights : List[List[float]]
        n_src x n_dst weight matrix.

    Returns
    -------
    List[float]
        Synaptic current vector for the destination layer (length n_dst).
    """
    n_src = len(weights)
    if len(spikes) != n_src:
        raise ValueError(f"spikes length {len(spikes)} != weight rows {n_src}")

    n_dst = len(weights[0]) if n_src > 0 else 0
    i_syn = [0.0] * n_dst

    for i, s in enumerate(spikes):
        if s == 0:
            continue
        row = weights[i]
        for j in range(n_dst):
            i_syn[j] += row[j]

    return i_syn


# ──────────────────────────────────────────────
# 2.  LAYER CONNECTION
# ──────────────────────────────────────────────

@dataclass
class LayerConnection:
    """A weighted connection between two adjacent layers."""
    src_size: int
    dst_size: int
    weights: List[List[float]]

    def propagate(self, src_spikes: List[int]) -> List[float]:
        """Compute the synaptic current delivered to the destination layer."""
        return matvec_spike_current(src_spikes, self.weights)


# ──────────────────────────────────────────────
# 3.  NEURON ARRAY (MULTI-LAYER NETWORK)
# ──────────────────────────────────────────────

class NeuroArray:
    """
    Feed-forward multi-layer spiking neuron array.

    Topology is defined by `layer_sizes`, e.g. [256, 128, 64, 32, 8]
    builds 4 LIFNeuronLayers connected by 4 LayerConnections.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        weight_init: str = "uniform",
        seed: Optional[int] = None,
        neuron_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the multi-layer neuron array.

        Parameters
        ----------
        layer_sizes : List[int]
            Sizes of each layer, e.g. [N_in, M_hidden1, M_hidden2, K_out].
            Must have at least 2 entries (input layer is just a spike
            source — only layers from index 1 onward have LIF neurons).
        weight_init : str
            "uniform" or "normal" weight initialization scheme.
        seed : int, optional
            RNG seed for reproducible weight initialization.
        neuron_kwargs : dict, optional
            Extra kwargs forwarded to every LIFNeuron in every layer
            (e.g. tau_mem_ms, v_threshold, refractory_ticks).
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 entries (input + 1 compute layer)")

        self.layer_sizes = list(layer_sizes)
        neuron_kwargs = neuron_kwargs or {}

        # Compute layers: one LIFNeuronLayer per layer AFTER the input layer
        self.compute_layers: List[LIFNeuronLayer] = [
            LIFNeuronLayer(n_neurons=size, **neuron_kwargs)
            for size in layer_sizes[1:]
        ]

        # Connections: one LayerConnection between each consecutive pair
        self.connections: List[LayerConnection] = []
        init_fn = init_weights_uniform if weight_init == "uniform" else init_weights_normal

        for idx in range(len(layer_sizes) - 1):
            n_src = layer_sizes[idx]
            n_dst = layer_sizes[idx + 1]
            w = init_fn(n_src, n_dst, seed=(seed + idx if seed is not None else None))
            self.connections.append(LayerConnection(n_src, n_dst, w))

        # Per-layer spike history (most recent step), indexed 0=input, 1..=compute layers
        self.last_spikes: List[List[int]] = [[0] * layer_sizes[0]] + [
            [0] * size for size in layer_sizes[1:]
        ]

        self.tick_count = 0

    def step(self, input_spikes: List[int]) -> List[int]:
        """
        Advance the full network by one tick, given input-layer spikes.

        Parameters
        ----------
        input_spikes : List[int]
            Binary spike vector for the input layer (length layer_sizes[0]).

        Returns
        -------
        List[int]
            Output-layer spike vector (length layer_sizes[-1]).
        """
        if len(input_spikes) != self.layer_sizes[0]:
            raise ValueError(
                f"input_spikes length {len(input_spikes)} != "
                f"input layer size {self.layer_sizes[0]}"
            )

        self.tick_count += 1
        self.last_spikes[0] = list(input_spikes)

        current_spikes = input_spikes
        for layer_idx, (conn, layer) in enumerate(zip(self.connections, self.compute_layers)):
            i_syn = conn.propagate(current_spikes)
            current_spikes = layer.step(i_syn)
            self.last_spikes[layer_idx + 1] = list(current_spikes)

        return current_spikes

    def run(self, input_spike_matrix: List[List[int]]) -> List[List[int]]:
        """
        Run the network over a sequence of input spike vectors.

        Parameters
        ----------
        input_spike_matrix : List[List[int]]
            One input spike vector per tick.

        Returns
        -------
        List[List[int]]
            Output-layer spike vector per tick.
        """
        return [self.step(vec) for vec in input_spike_matrix]

    def get_layer_spikes(self, layer_idx: int) -> List[int]:
        """
        Get the most recent spike vector for a given layer.

        Parameters
        ----------
        layer_idx : int
            0 = input layer, 1..N = compute layers (hidden/output).

        Returns
        -------
        List[int]
            Spike vector for that layer at the last step() call.
        """
        return self.last_spikes[layer_idx]

    def get_weights(self, connection_idx: int) -> List[List[float]]:
        """Return the weight matrix for a given layer connection."""
        return self.connections[connection_idx].weights

    def set_weights(self, connection_idx: int, weights: List[List[float]]) -> None:
        """
        Replace the weight matrix for a given connection (e.g. after STDP
        update or boot-time pre-trained weight load).
        """
        conn = self.connections[connection_idx]
        if len(weights) != conn.src_size or len(weights[0]) != conn.dst_size:
            raise ValueError("Weight matrix shape mismatch")
        conn.weights = weights

    def total_neurons(self) -> int:
        """Total number of LIF neurons across all compute layers (excludes input)."""
        return sum(self.layer_sizes[1:])

    def membrane_snapshot(self) -> List[List[float]]:
        """Return current membrane potentials for every compute layer."""
        return [layer.membrane_potentials() for layer in self.compute_layers]

    def reset(self) -> None:
        """Reset all neuron states (membrane potentials, refractory counters)."""
        for layer in self.compute_layers:
            for neuron in layer.neurons:
                neuron.reset_state()
        self.tick_count = 0
        self.last_spikes = [[0] * s for s in self.layer_sizes]


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("neuro_array.py  —  self-test")
    print("=" * 55)

    # Test 1: Weight initialization
    print("\n[ Weight matrix initialization ]")
    w_uniform = init_weights_uniform(4, 3, low=0.0, high=0.5, seed=42)
    assert len(w_uniform) == 4 and len(w_uniform[0]) == 3
    assert all(0.0 <= v <= 0.5 for row in w_uniform for v in row)
    print(f"  ✓ Uniform 4x3 weight matrix in [0.0, 0.5]")

    w_normal = init_weights_normal(4, 3, mean=0.2, std=0.05, seed=42)
    assert all(v >= 0.0 for row in w_normal for v in row)
    print(f"  ✓ Normal 4x3 weight matrix, clamped non-negative")

    # Test 2: matvec_spike_current basic
    print("\n[ Spike-current matrix-vector product ]")
    weights = [[0.5, 0.2], [0.3, 0.7], [0.0, 0.1]]
    spikes = [1, 0, 1]
    i_syn = matvec_spike_current(spikes, weights)
    assert i_syn == [0.5, 0.30000000000000004][:1] or abs(i_syn[0] - 0.5) < 1e-9
    assert abs(i_syn[1] - 0.3) < 1e-9
    print(f"  ✓ I_syn = {i_syn} for spikes={spikes}")

    # Test 3: Dimension mismatch raises
    print("\n[ Dimension mismatch validation ]")
    try:
        matvec_spike_current([1, 0], weights)  # wrong length
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ Mismatched spike vector length correctly rejected")

    # Test 4: Small NeuroArray construction
    print("\n[ NeuroArray construction ]")
    net = NeuroArray([8, 4, 2], weight_init="uniform", seed=7,
                      neuron_kwargs={"v_threshold": 1.0, "refractory_ticks": 1})
    assert len(net.compute_layers) == 2
    assert len(net.connections) == 2
    assert net.total_neurons() == 6  # 4 + 2
    print(f"  ✓ Built 8→4→2 network: {net.total_neurons()} compute neurons")

    # Test 5: Single step propagation
    print("\n[ Single-tick forward propagation ]")
    input_spikes = [1, 0, 1, 0, 1, 0, 1, 0]
    out = net.step(input_spikes)
    assert len(out) == 2
    print(f"  ✓ Input spikes → output layer spikes: {out}")

    # Test 6: Layer spike retrieval
    print("\n[ Per-layer spike retrieval ]")
    layer0 = net.get_layer_spikes(0)  # input
    layer1 = net.get_layer_spikes(1)  # hidden
    layer2 = net.get_layer_spikes(2)  # output
    assert layer0 == input_spikes
    assert len(layer1) == 4
    assert len(layer2) == 2
    print(f"  ✓ Input: {layer0}")
    print(f"  ✓ Hidden: {layer1}")
    print(f"  ✓ Output: {layer2}")

    # Test 7: Multi-tick run with strong drive causes hidden/output activity
    print("\n[ Multi-tick run with strong drive ]")
    net2 = NeuroArray([4, 3, 2], weight_init="uniform", seed=3,
                       neuron_kwargs={"v_threshold": 0.5, "refractory_ticks": 1})
    strong_input = [[1, 1, 1, 1] for _ in range(50)]
    outputs = net2.run(strong_input)
    total_output_spikes = sum(sum(row) for row in outputs)
    assert total_output_spikes > 0
    print(f"  ✓ 50-tick run with strong drive → {total_output_spikes} total output spikes")

    # Test 8: Weight get/set roundtrip
    print("\n[ Weight get/set roundtrip ]")
    original_weights = net2.get_weights(0)
    new_weights = [[0.9 for _ in range(3)] for _ in range(4)]
    net2.set_weights(0, new_weights)
    retrieved = net2.get_weights(0)
    assert retrieved == new_weights
    assert retrieved != original_weights
    print(f"  ✓ Weight matrix replaced and verified")

    # Test 9: Weight set shape validation
    print("\n[ Weight shape validation ]")
    try:
        net2.set_weights(0, [[0.1, 0.2]])  # wrong shape
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ Mismatched weight shape correctly rejected")

    # Test 10: Membrane snapshot
    print("\n[ Membrane potential snapshot ]")
    snapshot = net2.membrane_snapshot()
    assert len(snapshot) == 2  # 2 compute layers
    assert len(snapshot[0]) == 3  # hidden layer size
    assert len(snapshot[1]) == 2  # output layer size
    print(f"  ✓ Membrane snapshot shapes: {[len(s) for s in snapshot]}")

    # Test 11: Reset clears state
    print("\n[ Network reset ]")
    net2.reset()
    assert net2.tick_count == 0
    snapshot_after_reset = net2.membrane_snapshot()
    assert all(v == 0.0 for layer in snapshot_after_reset for v in layer)
    print(f"  ✓ Reset clears tick count and membrane potentials")

    # Test 12: Full AEGIS-NEURO-scale topology smoke test
    print("\n[ Full-scale topology (256→128→64→32→8) ]")
    big_net = NeuroArray(
        [256, 128, 64, 32, 8],
        weight_init="normal",
        seed=123,
        neuron_kwargs={"v_threshold": 1.0, "refractory_ticks": 2},
    )
    assert big_net.total_neurons() == 128 + 64 + 32 + 8
    sparse_input = [1 if i % 16 == 0 else 0 for i in range(256)]
    out_big = big_net.step(sparse_input)
    assert len(out_big) == 8
    print(f"  ✓ Full 256→128→64→32→8 topology: {big_net.total_neurons()} neurons, "
          f"output={out_big}")

    print("\n  All checks passed.\n")
