"""
lif_neuron.py
==============
Leaky Integrate-and-Fire (LIF) neuron model for the SNN core in the
PQC-SNN SoC (FB_SNN — Neuron Processing Element).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Implements the LIF neuron's membrane dynamics, threshold/spike/reset      ║
║ behavior, and refractory period — the core compute primitive of the SNN. ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Membrane potential integration (leaky integrator, discrete-time)
  - Threshold crossing → spike generation
  - Hard reset (membrane → reset potential after spike)
  - Refractory period (neuron ignores input for R ticks post-spike)
  - Synaptic input accumulation (weighted spike sum per tick)
  - Per-neuron state container for vectorized/batch simulation
  - Bit-exact fixed-point variant for RTL parity checking

Algorithm (discrete-time LIF, per architecture diagram
"NEURON PROCESSING ELEMENT (PE)" box — Membrane/Leak/IF/Refractory/Spike):

  Each tick:
    if refractory_counter > 0:
        refractory_counter -= 1
        V_mem unchanged (or held at V_reset, configurable)
        spike = 0
    else:
        V_mem = V_mem * leak_factor + I_syn(t)        # leaky integration
        if V_mem >= V_threshold:
            spike = 1
            V_mem = V_reset                             # hard reset
            refractory_counter = R                      # enter refractory
        else:
            spike = 0

  Where:
    leak_factor = exp(-dt / tau_mem)  in [0, 1)  (closer to 1 = less leaky)
    I_syn(t)    = sum of weighted incoming spikes at this tick
    R           = refractory period length, in ticks

Fixed-point note:
  The RTL implementation uses fixed-point membrane/weight arithmetic.
  `LIFNeuronFixedPoint` mirrors this with integer Q-format arithmetic
  so golden-model outputs can be bit-compared against RTL simulation.

Matches lif_neuron.sv (hardware RTL reference, within NEURON PROCESSING
ELEMENT / "Membrane / Leak / Integrate-and-Fire (LIF)" sub-block).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : Gerstner & Kistler, "Spiking Neuron Models" (LIF formalism)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────
# 1.  LIF PARAMETERS (FLOATING-POINT GOLDEN MODEL)
# ──────────────────────────────────────────────

#: Default membrane time constant (ms)
DEFAULT_TAU_MEM_MS: float = 20.0

#: Default simulation time-step (ms)
DEFAULT_DT_MS: float = 1.0

#: Default firing threshold (normalized units)
DEFAULT_V_THRESHOLD: float = 1.0

#: Default reset potential after spike
DEFAULT_V_RESET: float = 0.0

#: Default resting potential (initial state)
DEFAULT_V_REST: float = 0.0

#: Default refractory period (ticks)
DEFAULT_REFRACTORY_TICKS: int = 2


# ──────────────────────────────────────────────
# 2.  LIF NEURON (FLOATING-POINT)
# ──────────────────────────────────────────────

@dataclass
class LIFNeuron:
    """
    A single Leaky Integrate-and-Fire neuron (floating-point golden model).
    """
    tau_mem_ms: float = DEFAULT_TAU_MEM_MS
    dt_ms: float = DEFAULT_DT_MS
    v_threshold: float = DEFAULT_V_THRESHOLD
    v_reset: float = DEFAULT_V_RESET
    v_rest: float = DEFAULT_V_REST
    refractory_ticks: int = DEFAULT_REFRACTORY_TICKS

    v_mem: float = field(init=False)
    refractory_counter: int = field(init=False, default=0)
    spike_count: int = field(init=False, default=0)
    tick_count: int = field(init=False, default=0)
    v_mem_history: List[float] = field(init=False, default_factory=list)
    spike_history: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.v_mem = self.v_rest
        self._leak_factor = math.exp(-self.dt_ms / self.tau_mem_ms)

    @property
    def leak_factor(self) -> float:
        """Per-tick leak multiplier, derived from tau_mem and dt."""
        return self._leak_factor

    def step(self, i_syn: float) -> int:
        """
        Advance the neuron by one simulation tick.

        Parameters
        ----------
        i_syn : float
            Synaptic input current for this tick (weighted spike sum).

        Returns
        -------
        int
            1 if the neuron spiked this tick, else 0.
        """
        self.tick_count += 1

        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            spike = 0
        else:
            self.v_mem = self.v_mem * self._leak_factor + i_syn

            if self.v_mem >= self.v_threshold:
                spike = 1
                self.v_mem = self.v_reset
                self.refractory_counter = self.refractory_ticks
                self.spike_count += 1
            else:
                spike = 0

        self.v_mem_history.append(self.v_mem)
        self.spike_history.append(spike)
        return spike

    def run(self, i_syn_train: List[float]) -> List[int]:
        """
        Run the neuron over a sequence of synaptic inputs.

        Parameters
        ----------
        i_syn_train : List[float]
            Synaptic input current for each tick.

        Returns
        -------
        List[int]
            Output spike train (1 entry per input tick).
        """
        return [self.step(i) for i in i_syn_train]

    def reset_state(self) -> None:
        """Reset membrane potential and refractory counter (not history)."""
        self.v_mem = self.v_rest
        self.refractory_counter = 0

    def is_refractory(self) -> bool:
        """True if the neuron is currently in its refractory period."""
        return self.refractory_counter > 0

    def firing_rate_hz(self) -> float:
        """Compute average firing rate over the simulated duration so far."""
        if self.tick_count == 0:
            return 0.0
        duration_s = (self.tick_count * self.dt_ms) / 1000.0
        if duration_s == 0.0:
            return 0.0
        return self.spike_count / duration_s


# ──────────────────────────────────────────────
# 3.  LIF NEURON LAYER (BATCH / VECTORIZED)
# ──────────────────────────────────────────────

class LIFNeuronLayer:
    """
    A layer of N independent LIF neurons, stepped together each tick.
    Mirrors a row of the SNN_CORE's neuron array (e.g. the M hidden layer).
    """

    def __init__(self, n_neurons: int, **neuron_kwargs):
        """
        Initialize a layer of identical LIF neurons.

        Parameters
        ----------
        n_neurons : int
            Number of neurons in the layer.
        **neuron_kwargs
            Keyword arguments forwarded to each LIFNeuron (tau_mem_ms, etc.).
        """
        self.n_neurons = n_neurons
        self.neurons: List[LIFNeuron] = [
            LIFNeuron(**neuron_kwargs) for _ in range(n_neurons)
        ]

    def step(self, i_syn_vector: List[float]) -> List[int]:
        """
        Advance all neurons in the layer by one tick.

        Parameters
        ----------
        i_syn_vector : List[float]
            Synaptic input current per neuron, length == n_neurons.

        Returns
        -------
        List[int]
            Spike output per neuron (length == n_neurons).
        """
        if len(i_syn_vector) != self.n_neurons:
            raise ValueError(
                f"i_syn_vector length {len(i_syn_vector)} != n_neurons {self.n_neurons}"
            )
        return [n.step(i) for n, i in zip(self.neurons, i_syn_vector)]

    def run(self, i_syn_matrix: List[List[float]]) -> List[List[int]]:
        """
        Run the layer over a sequence of input vectors.

        Parameters
        ----------
        i_syn_matrix : List[List[float]]
            One input vector per tick; outer length = n_ticks,
            inner length = n_neurons.

        Returns
        -------
        List[List[int]]
            Spike output matrix: n_ticks x n_neurons.
        """
        return [self.step(vec) for vec in i_syn_matrix]

    def membrane_potentials(self) -> List[float]:
        """Current membrane potential of every neuron in the layer."""
        return [n.v_mem for n in self.neurons]

    def spike_counts(self) -> List[int]:
        """Total spike count so far, per neuron."""
        return [n.spike_count for n in self.neurons]


# ──────────────────────────────────────────────
# 4.  FIXED-POINT LIF NEURON (RTL PARITY MODEL)
# ──────────────────────────────────────────────

class LIFNeuronFixedPoint:
    """
    Fixed-point LIF neuron using Q-format integer arithmetic, mirroring
    the RTL datapath for bit-exact comparison.

    Uses Q(INT_BITS).(FRAC_BITS) signed fixed-point representation.
    """

    def __init__(
        self,
        frac_bits: int = 12,
        leak_factor_q: int | None = None,
        v_threshold: float = DEFAULT_V_THRESHOLD,
        v_reset: float = DEFAULT_V_RESET,
        refractory_ticks: int = DEFAULT_REFRACTORY_TICKS,
        tau_mem_ms: float = DEFAULT_TAU_MEM_MS,
        dt_ms: float = DEFAULT_DT_MS,
    ):
        """
        Initialize a fixed-point LIF neuron.

        Parameters
        ----------
        frac_bits : int
            Number of fractional bits in the Q-format representation.
        leak_factor_q : int, optional
            Pre-quantized leak factor (Q frac_bits). If None, derived
            from tau_mem_ms/dt_ms and quantized automatically.
        v_threshold : float
            Firing threshold (floating-point, quantized internally).
        v_reset : float
            Reset potential (floating-point, quantized internally).
        refractory_ticks : int
            Refractory period length in ticks.
        tau_mem_ms : float
            Membrane time constant, used only if leak_factor_q is None.
        dt_ms : float
            Simulation time-step, used only if leak_factor_q is None.
        """
        self.frac_bits = frac_bits
        self.scale = 1 << frac_bits

        if leak_factor_q is None:
            leak_factor = math.exp(-dt_ms / tau_mem_ms)
            leak_factor_q = round(leak_factor * self.scale)
        self.leak_factor_q = leak_factor_q

        self.v_threshold_q = round(v_threshold * self.scale)
        self.v_reset_q = round(v_reset * self.scale)
        self.refractory_ticks = refractory_ticks

        self.v_mem_q: int = 0
        self.refractory_counter: int = 0
        self.spike_count: int = 0
        self.tick_count: int = 0

    def to_fixed(self, value: float) -> int:
        """Quantize a floating-point value to this neuron's Q-format."""
        return round(value * self.scale)

    def to_float(self, value_q: int) -> float:
        """Convert a Q-format integer back to floating-point."""
        return value_q / self.scale

    def step(self, i_syn_q: int) -> int:
        """
        Advance the fixed-point neuron by one tick.

        Parameters
        ----------
        i_syn_q : int
            Synaptic input current, already in Q-format.

        Returns
        -------
        int
            1 if the neuron spiked this tick, else 0.
        """
        self.tick_count += 1

        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            return 0

        # Leaky integration: v_mem = (v_mem * leak_q) >> frac_bits + i_syn
        leaked = (self.v_mem_q * self.leak_factor_q) >> self.frac_bits
        self.v_mem_q = leaked + i_syn_q

        if self.v_mem_q >= self.v_threshold_q:
            self.v_mem_q = self.v_reset_q
            self.refractory_counter = self.refractory_ticks
            self.spike_count += 1
            return 1

        return 0


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("lif_neuron.py  —  self-test")
    print("=" * 55)

    # Test 1: Subthreshold input never spikes
    print("\n[ Subthreshold input ]")
    n = LIFNeuron(v_threshold=1.0)
    spikes = n.run([0.01] * 50)
    assert sum(spikes) == 0
    print(f"  ✓ Weak input (0.01/tick × 50) → no spikes")

    # Test 2: Strong constant input causes repeated firing
    print("\n[ Strong constant input ]")
    n2 = LIFNeuron(v_threshold=1.0, refractory_ticks=2)
    spikes2 = n2.run([0.6] * 50)
    assert sum(spikes2) >= 5
    print(f"  ✓ Strong input (0.6/tick × 50) → {sum(spikes2)} spikes")

    # Test 3: Hard reset after spike
    print("\n[ Hard reset after spike ]")
    n3 = LIFNeuron(v_threshold=1.0, v_reset=0.0)
    n3.step(1.5)  # forces immediate spike
    assert n3.v_mem == 0.0
    print(f"  ✓ V_mem reset to {n3.v_mem} immediately after spiking")

    # Test 4: Refractory period blocks input
    print("\n[ Refractory period ]")
    n4 = LIFNeuron(v_threshold=1.0, v_reset=0.0, refractory_ticks=3)
    s = n4.step(1.5)  # spike at tick 1
    assert s == 1
    assert n4.is_refractory()
    blocked_spikes = [n4.step(2.0) for _ in range(3)]  # huge input, should be ignored
    assert sum(blocked_spikes) == 0
    assert not n4.is_refractory()
    print(f"  ✓ 3-tick refractory period correctly blocks strong input")

    # Test 5: Leak factor decays membrane with no input
    print("\n[ Passive leak decay ]")
    n5 = LIFNeuron(tau_mem_ms=20.0, dt_ms=1.0)
    n5.v_mem = 0.8
    n5.step(0.0)  # no input, just leak
    assert n5.v_mem < 0.8
    expected = 0.8 * math.exp(-1.0 / 20.0)
    assert abs(n5.v_mem - expected) < 1e-9
    print(f"  ✓ V_mem decayed from 0.8 → {n5.v_mem:.4f} (expected {expected:.4f})")

    # Test 6: Firing rate calculation
    print("\n[ Firing rate calculation ]")
    n6 = LIFNeuron(v_threshold=1.0, refractory_ticks=0, dt_ms=1.0)
    n6.run([1.5] * 100)  # spikes nearly every tick (no refractory)
    rate = n6.firing_rate_hz()
    assert rate > 500  # should be close to 1000 Hz at dt=1ms, every-tick firing
    print(f"  ✓ Firing rate: {rate:.1f} Hz over 100 ticks ({n6.spike_count} spikes)")

    # Test 7: Neuron layer — batch stepping
    print("\n[ LIFNeuronLayer batch stepping ]")
    layer = LIFNeuronLayer(n_neurons=4, v_threshold=1.0, refractory_ticks=2)
    inputs = [0.6, 0.1, 0.6, 0.05]
    out = layer.step(inputs)
    assert len(out) == 4
    print(f"  ✓ 4-neuron layer single step: spikes={out}")

    # Test 8: Layer — independent neuron dynamics
    print("\n[ Layer — independent dynamics ]")
    layer2 = LIFNeuronLayer(n_neurons=3, v_threshold=1.0, refractory_ticks=2)
    matrix = [[0.6, 0.0, 0.3] for _ in range(30)]
    spike_matrix = layer2.run(matrix)
    counts = layer2.spike_counts()
    assert counts[0] > counts[2] > counts[1]  # strong > medium > none
    print(f"  ✓ Spike counts per neuron: {counts} (strong > medium > silent)")

    # Test 9: Layer dimension mismatch raises
    print("\n[ Layer dimension validation ]")
    layer3 = LIFNeuronLayer(n_neurons=4)
    try:
        layer3.step([0.1, 0.2])  # wrong length
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ Mismatched input vector length correctly rejected")

    # Test 10: Fixed-point neuron basic operation
    print("\n[ Fixed-point LIF neuron ]")
    fp = LIFNeuronFixedPoint(frac_bits=12, v_threshold=1.0, v_reset=0.0,
                              refractory_ticks=2, tau_mem_ms=20.0, dt_ms=1.0)
    i_syn_q = fp.to_fixed(0.6)
    spikes_fp = [fp.step(i_syn_q) for _ in range(50)]
    assert sum(spikes_fp) > 0
    print(f"  ✓ Fixed-point neuron spiked {sum(spikes_fp)} times over 50 ticks")

    # Test 11: Fixed-point vs floating-point parity (approximate)
    print("\n[ Fixed-point vs floating-point parity ]")
    n_float = LIFNeuron(tau_mem_ms=20.0, dt_ms=1.0, v_threshold=1.0,
                         v_reset=0.0, refractory_ticks=2)
    fp2 = LIFNeuronFixedPoint(frac_bits=16, v_threshold=1.0, v_reset=0.0,
                               refractory_ticks=2, tau_mem_ms=20.0, dt_ms=1.0)
    float_spikes = n_float.run([0.6] * 50)
    fp_spikes = [fp2.step(fp2.to_fixed(0.6)) for _ in range(50)]
    # With high precision (16 frac bits), spike counts should match closely
    assert abs(sum(float_spikes) - sum(fp_spikes)) <= 1
    print(f"  ✓ Float spikes={sum(float_spikes)}, Fixed-point spikes={sum(fp_spikes)} "
          f"(within tolerance)")

    # Test 12: Reset state
    print("\n[ Reset state ]")
    n7 = LIFNeuron(v_threshold=1.0)
    n7.v_mem = 0.7
    n7.refractory_counter = 2
    n7.reset_state()
    assert n7.v_mem == n7.v_rest
    assert n7.refractory_counter == 0
    print(f"  ✓ reset_state() restores resting potential, clears refractory")

    print("\n  All checks passed.\n")
