"""
stdp_engine.py
================
Spike-Timing-Dependent Plasticity engine for the SNN core in the
PQC-SNN SoC (FB_SNN — Learning & Adaptation / STDP Engine).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Computes STDP weight deltas from pre/post spike-timing differences and    ║
║ feeds them into synapse_memory.py's online-learning update path.        ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Classical pairwise STDP exponential learning-rule (LTP/LTD windows)
  - Per-synapse eligibility traces (pre-synaptic and post-synaptic),
    the standard trace-based formulation that avoids storing full
    spike-time history while remaining mathematically equivalent to
    pairwise STDP for non-overlapping spike pairs
  - All-to-all pairwise STDP (exact, for small layers / verification)
  - Weight-dependent multiplicative bounds (soft bounds, prevents
    unbounded weight growth/decay — standard stabilization technique)
  - Batch update generation compatible with synapse_memory.py's
    apply_delta_matrix() interface

Context (per architecture diagram, "LEARNING & ADAPTATION (ADVANCED)" box):
  Lists "STDP Engine", "Homeostatic Plasticity", "Lateral Inhibition",
  and "Weight Update Engine" as the four advanced learning sub-blocks.
  This module is the STDP Engine: it observes pre/post spike trains
  across a layer connection and produces the delta-weight matrix that
  the Weight Update Engine (synapse_memory.py's write path) applies.

STDP rule (pairwise, exponential kernel):
  For a pre-spike at time t_pre and post-spike at time t_post:
    Δt = t_post - t_pre
    if Δt > 0  (post AFTER pre — causal, strengthen):
        Δw = +A_plus  * exp(-Δt / tau_plus)
    if Δt < 0  (post BEFORE pre — anti-causal, weaken):
        Δw = -A_minus * exp(+Δt / tau_minus)

Trace-based formulation (used by STDPTraceEngine, the efficient online
variant suitable for per-tick hardware operation):
  Pre-synaptic trace x_pre[i]   decays each tick: x_pre  *= exp(-dt/tau_plus)
  Post-synaptic trace x_post[j] decays each tick: x_post *= exp(-dt/tau_minus)
  On a pre-spike at i:  x_pre[i]  += 1;  w[i][j] -= A_minus * x_post[j]  ∀j
  On a post-spike at j: x_post[j] += 1;  w[i][j] += A_plus  * x_pre[i]  ∀i

Weight-dependent soft bounds (applied to keep weights in [w_min, w_max]):
  Δw_LTP_bounded  = Δw_LTP  * (w_max - w)   (growth slows as w → w_max)
  Δw_LTD_bounded  = Δw_LTD  * (w - w_min)   (decay slows as w → w_min)

Matches stdp_engine.sv (hardware RTL reference, within "Learning &
Adaptation (Advanced)" → "STDP Engine" entry).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : Gerstner & Kistler, "Spiking Neuron Models"; Song, Miller & Abbott
         (2000), "Competitive Hebbian learning through STDP"
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────
# 1.  STDP PARAMETERS
# ──────────────────────────────────────────────

#: LTP (long-term potentiation) amplitude — strengthening step size
DEFAULT_A_PLUS: float = 0.01

#: LTD (long-term depression) amplitude — weakening step size
#: (conventionally slightly larger than A_plus for stable, sparse codes)
DEFAULT_A_MINUS: float = 0.012

#: LTP time constant (ticks) — how far back a pre-spike still matters
DEFAULT_TAU_PLUS: float = 20.0

#: LTD time constant (ticks)
DEFAULT_TAU_MINUS: float = 20.0

#: Default simulation time-step (ticks per call)
DEFAULT_DT: float = 1.0

#: Default soft weight bounds
DEFAULT_W_MIN: float = 0.0
DEFAULT_W_MAX: float = 1.0


# ──────────────────────────────────────────────
# 2.  CLASSICAL PAIRWISE STDP (EXACT, FOR VERIFICATION)
# ──────────────────────────────────────────────

def pairwise_stdp_delta(
    delta_t: float,
    a_plus: float = DEFAULT_A_PLUS,
    a_minus: float = DEFAULT_A_MINUS,
    tau_plus: float = DEFAULT_TAU_PLUS,
    tau_minus: float = DEFAULT_TAU_MINUS,
) -> float:
    """
    Compute the classical pairwise STDP weight change for one spike pair.

    Parameters
    ----------
    delta_t : float
        t_post - t_pre, in ticks. Positive means post fired after pre
        (causal — strengthen); negative means post fired before pre
        (anti-causal — weaken).
    a_plus : float
        LTP amplitude.
    a_minus : float
        LTD amplitude.
    tau_plus : float
        LTP time constant.
    tau_minus : float
        LTD time constant.

    Returns
    -------
    float
        Signed weight change Δw for this spike pair.
    """
    if delta_t > 0:
        return a_plus * math.exp(-delta_t / tau_plus)
    elif delta_t < 0:
        return -a_minus * math.exp(delta_t / tau_minus)
    return 0.0  # simultaneous spikes — no defined causal direction


def all_to_all_stdp(
    pre_spike_times: List[int],
    post_spike_times: List[int],
    a_plus: float = DEFAULT_A_PLUS,
    a_minus: float = DEFAULT_A_MINUS,
    tau_plus: float = DEFAULT_TAU_PLUS,
    tau_minus: float = DEFAULT_TAU_MINUS,
    cutoff_ticks: Optional[float] = None,
) -> float:
    """
    Compute total STDP weight change for one synapse, summing over EVERY
    pre/post spike-time pair (exact "all-to-all" STDP — O(n*m) but exact;
    used as a golden reference to verify the efficient trace-based engine
    against, for small spike trains).

    Parameters
    ----------
    pre_spike_times : List[int]
        Tick indices at which the pre-synaptic neuron spiked.
    post_spike_times : List[int]
        Tick indices at which the post-synaptic neuron spiked.
    a_plus, a_minus, tau_plus, tau_minus : float
        STDP rule parameters (see pairwise_stdp_delta).
    cutoff_ticks : float, optional
        If set, spike pairs with |Δt| beyond this are ignored (matches
        a finite hardware trace-window approximation).

    Returns
    -------
    float
        Total accumulated Δw for this synapse.
    """
    total = 0.0
    for t_pre in pre_spike_times:
        for t_post in post_spike_times:
            dt = t_post - t_pre
            if cutoff_ticks is not None and abs(dt) > cutoff_ticks:
                continue
            total += pairwise_stdp_delta(dt, a_plus, a_minus, tau_plus, tau_minus)
    return total


# ──────────────────────────────────────────────
# 3.  WEIGHT-DEPENDENT SOFT BOUNDS
# ──────────────────────────────────────────────

def apply_soft_bounds(
    delta_w: float, current_w: float, w_min: float = DEFAULT_W_MIN, w_max: float = DEFAULT_W_MAX
) -> float:
    """
    Scale a raw STDP delta by weight-dependent soft bounds, so growth
    slows near w_max and decay slows near w_min (prevents unbounded
    weight drift, a standard STDP stabilization technique).

    Parameters
    ----------
    delta_w : float
        Raw (unbounded) weight change.
    current_w : float
        Current weight value.
    w_min : float
        Lower soft bound.
    w_max : float
        Upper soft bound.

    Returns
    -------
    float
        Bounded weight change.
    """
    if delta_w > 0:
        return delta_w * max(0.0, w_max - current_w)
    elif delta_w < 0:
        return delta_w * max(0.0, current_w - w_min)
    return 0.0


# ──────────────────────────────────────────────
# 4.  TRACE-BASED ONLINE STDP ENGINE
# ──────────────────────────────────────────────

class STDPTraceEngine:
    """
    Efficient, per-tick trace-based STDP engine for one layer connection
    (n_pre pre-synaptic neurons → n_post post-synaptic neurons).

    Maintains exponentially-decaying eligibility traces per neuron
    instead of full spike-time history, matching how this would be
    implemented in hardware (a small accumulator per neuron, decayed
    every tick) and producing results equivalent to pairwise STDP for
    well-separated spikes.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        a_plus: float = DEFAULT_A_PLUS,
        a_minus: float = DEFAULT_A_MINUS,
        tau_plus: float = DEFAULT_TAU_PLUS,
        tau_minus: float = DEFAULT_TAU_MINUS,
        dt: float = DEFAULT_DT,
        w_min: float = DEFAULT_W_MIN,
        w_max: float = DEFAULT_W_MAX,
        use_soft_bounds: bool = True,
    ):
        """
        Initialize the trace-based STDP engine.

        Parameters
        ----------
        n_pre : int
            Number of pre-synaptic (source-layer) neurons.
        n_post : int
            Number of post-synaptic (destination-layer) neurons.
        a_plus, a_minus : float
            LTP/LTD amplitudes.
        tau_plus, tau_minus : float
            LTP/LTD trace decay time constants (ticks).
        dt : float
            Simulation time-step (ticks per call to step()).
        w_min, w_max : float
            Soft weight bounds (only used if use_soft_bounds=True).
        use_soft_bounds : bool
            If True, deltas are scaled by apply_soft_bounds() using the
            CURRENT weight matrix passed into step(); if False, raw
            (unbounded) deltas are returned.
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.dt = dt
        self.w_min = w_min
        self.w_max = w_max
        self.use_soft_bounds = use_soft_bounds

        self.x_pre: List[float] = [0.0] * n_pre
        self.x_post: List[float] = [0.0] * n_post

        self._decay_pre = math.exp(-dt / tau_plus)
        self._decay_post = math.exp(-dt / tau_minus)

        self.tick_count: int = 0
        self.total_ltp_events: int = 0
        self.total_ltd_events: int = 0

    def step(
        self,
        pre_spikes: List[int],
        post_spikes: List[int],
        current_weights: Optional[List[List[float]]] = None,
    ) -> List[List[float]]:
        """
        Advance the STDP engine by one tick and compute this tick's
        weight-delta matrix.

        Parameters
        ----------
        pre_spikes : List[int]
            Binary spike vector for the pre-synaptic layer this tick.
        post_spikes : List[int]
            Binary spike vector for the post-synaptic layer this tick.
        current_weights : List[List[float]], optional
            Current n_pre x n_post weight matrix. Required if
            use_soft_bounds=True (to compute weight-dependent scaling);
            ignored otherwise.

        Returns
        -------
        List[List[float]]
            n_pre x n_post matrix of weight deltas for this tick,
            directly compatible with synapse_memory.SynapseMemory
            .apply_delta_matrix().
        """
        if len(pre_spikes) != self.n_pre:
            raise ValueError(f"pre_spikes length {len(pre_spikes)} != n_pre {self.n_pre}")
        if len(post_spikes) != self.n_post:
            raise ValueError(f"post_spikes length {len(post_spikes)} != n_post {self.n_post}")
        if self.use_soft_bounds and current_weights is None:
            raise ValueError("current_weights required when use_soft_bounds=True")

        self.tick_count += 1

        # Decay traces first (exponential decay over this tick's dt)
        self.x_pre = [x * self._decay_pre for x in self.x_pre]
        self.x_post = [x * self._decay_post for x in self.x_post]

        delta_w = [[0.0] * self.n_post for _ in range(self.n_pre)]

        # LTD: for each pre-spike now, weaken synapses to neurons with
        # an existing (decayed) post-trace — i.e. post fired RECENTLY
        # BEFORE this pre-spike (anti-causal).
        for i, s_pre in enumerate(pre_spikes):
            if s_pre == 1:
                for j in range(self.n_post):
                    if self.x_post[j] > 0.0:
                        raw = -self.a_minus * self.x_post[j]
                        delta_w[i][j] += raw
                        self.total_ltd_events += 1

        # LTP: for each post-spike now, strengthen synapses from neurons
        # with an existing (decayed) pre-trace — i.e. pre fired RECENTLY
        # BEFORE this post-spike (causal).
        for j, s_post in enumerate(post_spikes):
            if s_post == 1:
                for i in range(self.n_pre):
                    if self.x_pre[i] > 0.0:
                        raw = self.a_plus * self.x_pre[i]
                        delta_w[i][j] += raw
                        self.total_ltp_events += 1

        # Apply soft bounds using the weight values BEFORE this tick's
        # update (matches a single-tick hardware update ordering).
        if self.use_soft_bounds:
            for i in range(self.n_pre):
                for j in range(self.n_post):
                    if delta_w[i][j] != 0.0:
                        delta_w[i][j] = apply_soft_bounds(
                            delta_w[i][j], current_weights[i][j], self.w_min, self.w_max
                        )

        # Update traces AFTER computing this tick's deltas (so a
        # simultaneous pre+post spike pair does not see its own
        # just-incremented trace within the same tick).
        for i, s in enumerate(pre_spikes):
            if s == 1:
                self.x_pre[i] += 1.0
        for j, s in enumerate(post_spikes):
            if s == 1:
                self.x_post[j] += 1.0

        return delta_w

    def run(
        self,
        pre_spike_matrix: List[List[int]],
        post_spike_matrix: List[List[int]],
        weight_matrix: Optional[List[List[float]]] = None,
        synapse_memory=None,
    ) -> List[List[List[float]]]:
        """
        Run the STDP engine over a multi-tick sequence, optionally
        applying each tick's deltas directly into a SynapseMemory bank.

        Parameters
        ----------
        pre_spike_matrix : List[List[int]]
            One pre-synaptic spike vector per tick.
        post_spike_matrix : List[List[int]]
            One post-synaptic spike vector per tick (same length).
        weight_matrix : List[List[float]], optional
            Starting weight matrix (required if use_soft_bounds=True
            and synapse_memory is None — otherwise the caller manages
            weight state via synapse_memory).
        synapse_memory : SynapseMemory, optional
            If provided, each tick's delta matrix is applied directly
            via synapse_memory.apply_delta_matrix(), and current weights
            are read back from it for soft-bound scaling each tick
            (keeps weight state in one place rather than duplicated).

        Returns
        -------
        List[List[List[float]]]
            Per-tick delta-weight matrices (n_ticks x n_pre x n_post).
        """
        if len(pre_spike_matrix) != len(post_spike_matrix):
            raise ValueError("pre/post spike matrices must have equal length")

        all_deltas = []
        local_weights = (
            [row[:] for row in weight_matrix] if weight_matrix is not None else None
        )

        for pre_t, post_t in zip(pre_spike_matrix, post_spike_matrix):
            if synapse_memory is not None:
                current = synapse_memory.to_dense()
            else:
                current = local_weights

            delta = self.step(pre_t, post_t, current_weights=current)
            all_deltas.append(delta)

            if synapse_memory is not None:
                synapse_memory.apply_delta_matrix(delta)
            elif local_weights is not None:
                for i in range(self.n_pre):
                    for j in range(self.n_post):
                        local_weights[i][j] += delta[i][j]

        return all_deltas

    def stats(self) -> dict:
        """Return STDP engine statistics."""
        return {
            "shape": (self.n_pre, self.n_post),
            "tick_count": self.tick_count,
            "ltp_events": self.total_ltp_events,
            "ltd_events": self.total_ltd_events,
            "max_pre_trace": max(self.x_pre) if self.x_pre else 0.0,
            "max_post_trace": max(self.x_post) if self.x_post else 0.0,
        }

    def reset(self) -> None:
        """Clear all traces (does not reset accumulated statistics)."""
        self.x_pre = [0.0] * self.n_pre
        self.x_post = [0.0] * self.n_post


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("stdp_engine.py  —  self-test")
    print("=" * 55)

    # Test 1: Pairwise STDP — causal (post after pre) strengthens
    print("\n[ Pairwise STDP — causal strengthening ]")
    dw_causal = pairwise_stdp_delta(delta_t=5.0)
    assert dw_causal > 0
    print(f"  ✓ post 5 ticks after pre → Δw={dw_causal:.5f} (positive, LTP)")

    # Test 2: Pairwise STDP — anti-causal (post before pre) weakens
    print("\n[ Pairwise STDP — anti-causal weakening ]")
    dw_anticausal = pairwise_stdp_delta(delta_t=-5.0)
    assert dw_anticausal < 0
    print(f"  ✓ post 5 ticks before pre → Δw={dw_anticausal:.5f} (negative, LTD)")

    # Test 3: Pairwise STDP — magnitude decays with |Δt|
    print("\n[ Pairwise STDP — temporal decay ]")
    dw_close = pairwise_stdp_delta(delta_t=1.0)
    dw_far = pairwise_stdp_delta(delta_t=30.0)
    assert dw_close > dw_far > 0
    print(f"  ✓ Δt=1 → {dw_close:.5f}, Δt=30 → {dw_far:.5f} (decays with distance)")

    # Test 4: Simultaneous spikes — no defined direction
    print("\n[ Simultaneous spikes ]")
    dw_simul = pairwise_stdp_delta(delta_t=0.0)
    assert dw_simul == 0.0
    print(f"  ✓ Δt=0 → Δw=0 (no causal direction defined)")

    # Test 5: All-to-all STDP — single clean causal pair
    print("\n[ All-to-all STDP — single pair ]")
    total = all_to_all_stdp(pre_spike_times=[10], post_spike_times=[15])
    expected = pairwise_stdp_delta(5.0)
    assert abs(total - expected) < 1e-9
    print(f"  ✓ Single pre@10, post@15 → total Δw={total:.5f} (matches single pair formula)")

    # Test 6: Soft bounds — LTP slows near w_max
    print("\n[ Soft bounds — LTP saturation ]")
    raw_ltp = 0.05
    bounded_low_w = apply_soft_bounds(raw_ltp, current_w=0.1, w_max=1.0)
    bounded_high_w = apply_soft_bounds(raw_ltp, current_w=0.95, w_max=1.0)
    assert bounded_high_w < bounded_low_w
    print(f"  ✓ Same raw LTP at w=0.1→{bounded_low_w:.4f}, at w=0.95→{bounded_high_w:.4f} "
          f"(growth slows near w_max)")

    # Test 7: Soft bounds — LTD slows near w_min
    print("\n[ Soft bounds — LTD saturation ]")
    raw_ltd = -0.05
    bounded_high_w2 = apply_soft_bounds(raw_ltd, current_w=0.9, w_min=0.0)
    bounded_low_w2 = apply_soft_bounds(raw_ltd, current_w=0.05, w_min=0.0)
    assert abs(bounded_low_w2) < abs(bounded_high_w2)
    print(f"  ✓ Same raw LTD at w=0.9→{bounded_high_w2:.4f}, at w=0.05→{bounded_low_w2:.4f} "
          f"(decay slows near w_min)")

    # Test 8: STDPTraceEngine — single causal pair produces positive delta
    print("\n[ Trace engine — single causal pair ]")
    engine = STDPTraceEngine(n_pre=2, n_post=2, use_soft_bounds=False)
    w = [[0.5, 0.5], [0.5, 0.5]]
    d1 = engine.step(pre_spikes=[1, 0], post_spikes=[0, 0], current_weights=w)
    assert all(v == 0.0 for row in d1 for v in row)  # pre alone: no post-trace yet
    d2 = engine.step(pre_spikes=[0, 0], post_spikes=[1, 0], current_weights=w)
    assert d2[0][0] > 0.0  # neuron 0 pre (1 tick ago) → neuron 0 post (now): LTP
    assert d2[1][0] == 0.0  # neuron 1 never fired
    print(f"  ✓ pre[0]@t0, post[0]@t1 → Δw[0][0]={d2[0][0]:.5f} (LTP, causal)")

    # Test 9: Trace engine — LTD direction (post before pre)
    print("\n[ Trace engine — LTD direction ]")
    engine2 = STDPTraceEngine(n_pre=2, n_post=2, use_soft_bounds=False)
    engine2.step(pre_spikes=[0, 0], post_spikes=[1, 0], current_weights=w)  # post fires first
    d_ltd = engine2.step(pre_spikes=[1, 0], post_spikes=[0, 0], current_weights=w)  # then pre
    assert d_ltd[0][0] < 0.0  # pre AFTER post → LTD
    print(f"  ✓ post[0]@t0, pre[0]@t1 → Δw[0][0]={d_ltd[0][0]:.5f} (LTD, anti-causal)")

    # Test 10: Trace engine with soft bounds requires current_weights
    print("\n[ Trace engine — soft bounds requirement ]")
    engine3 = STDPTraceEngine(n_pre=2, n_post=2, use_soft_bounds=True)
    try:
        engine3.step(pre_spikes=[1, 0], post_spikes=[0, 0], current_weights=None)
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ Soft-bounds mode correctly requires current_weights")

    # Test 11: Trace engine dimension validation
    print("\n[ Trace engine — dimension validation ]")
    try:
        engine.step(pre_spikes=[1, 0, 0], post_spikes=[0, 0], current_weights=w)
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ Mismatched pre_spikes length correctly rejected")

    # Test 12: Trace engine run() with local weight tracking
    print("\n[ Trace engine — multi-tick run() ]")
    engine4 = STDPTraceEngine(n_pre=2, n_post=2, use_soft_bounds=True,
                               a_plus=0.05, a_minus=0.05)
    pre_seq = [[1, 0], [0, 0], [0, 1], [0, 0]]
    post_seq = [[0, 0], [1, 0], [0, 0], [0, 1]]
    w_start = [[0.5, 0.5], [0.5, 0.5]]
    deltas = engine4.run(pre_seq, post_seq, weight_matrix=w_start)
    assert len(deltas) == 4
    print(f"  ✓ Ran 4-tick sequence, produced {len(deltas)} delta matrices")

    # Test 13: Integration with SynapseMemory.apply_delta_matrix
    print("\n[ Integration with synapse_memory.SynapseMemory ]")
    from synapse_memory import SynapseMemory, MemoryMode

    mem = SynapseMemory(n_rows=2, n_cols=2, precision_bits=16, mode=MemoryMode.READ_WRITE)
    mem.boot_load([[0.5, 0.5], [0.5, 0.5]])

    engine5 = STDPTraceEngine(n_pre=2, n_post=2, use_soft_bounds=True,
                               a_plus=0.05, a_minus=0.05)
    before_weights = mem.to_dense()
    engine5.run(pre_seq, post_seq, synapse_memory=mem)
    after_weights = mem.to_dense()
    assert before_weights != after_weights
    print(f"  ✓ STDP engine drove synapse_memory weight updates directly: "
          f"w[0][0] {before_weights[0][0]:.4f} → {after_weights[0][0]:.4f}")

    # Test 14: Statistics and reset
    print("\n[ Statistics and reset ]")
    stats = engine4.stats()
    print(f"  Shape: {stats['shape']}, LTP events: {stats['ltp_events']}, "
          f"LTD events: {stats['ltd_events']}")
    engine4.reset()
    assert all(x == 0.0 for x in engine4.x_pre)
    assert engine4.stats()["ltp_events"] == stats["ltp_events"]  # stats survive reset
    print(f"  ✓ reset() clears traces, preserves accumulated event counts")

    print("\n  All checks passed.\n")
