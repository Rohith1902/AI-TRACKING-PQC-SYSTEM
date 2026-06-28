"""
lateral_inhibitor.py
======================
Lateral inhibition for the SNN core in the PQC-SNN SoC
(FB_SNN — Learning & Adaptation / Lateral Inhibition).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Implements winner-take-all and soft lateral inhibition within a neuron   ║
║ layer, suppressing competing neurons once one fires to sharpen coding.   ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Hard winner-take-all (WTA): the first/strongest spiking neuron in a
    layer suppresses all other neurons for an inhibition window
  - Soft lateral inhibition: neurons near a winner receive a graded
    membrane-potential penalty based on distance (k-WTA / Mexican-hat
    style competition), rather than full suppression
  - Inhibition radius / neighborhood model (1D ring topology, configurable)
  - Self-exclusion (a neuron never inhibits itself)
  - Suppression duration tracking (inhibition decays after N ticks)
  - Vectorized application to LIFNeuronLayer-style membrane vectors

Context (per architecture diagram, "LEARNING & ADAPTATION (ADVANCED)" box):
  Lists "STDP Engine", "Homeostatic Plasticity", "Lateral Inhibition",
  and "Weight Update Engine" as the four advanced learning sub-blocks.
  This module implements the Lateral Inhibition piece: once a neuron in
  a competitive layer (typically the hidden/output decision layers) fires,
  its neighbors are suppressed so the network produces a sparse, decisive
  code rather than many neurons firing simultaneously for the same input
  — critical for clean threat-pattern discrimination in SNN_CORE.

Two inhibition modes:
  HARD (winner-take-all):
    On any spike at index w, all OTHER neurons' next-tick membrane
    potential is forced toward (or below) v_reset for `suppress_ticks`
    ticks — i.e. they cannot fire again until the winner's suppression
    window expires.

  SOFT (graded / k-WTA, Mexican-hat):
    On a spike at index w, neighbor j's membrane potential is reduced by:
      penalty(j) = inhibition_strength * exp(-(dist(j,w)^2) / (2*sigma^2))
    where dist() respects an optional inhibition_radius cutoff (no effect
    beyond that radius) and the layer topology (ring or linear).

Matches lateral_inhibitor.sv (hardware RTL reference, within the
"Learning & Adaptation (Advanced)" sub-block, "Lateral Inhibition" entry).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : Gerstner & Kistler, "Spiking Neuron Models" (WTA / lateral inhibition)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ──────────────────────────────────────────────
# 1.  PARAMETERS
# ──────────────────────────────────────────────

class InhibitionMode(Enum):
    HARD = "hard"   # winner-take-all
    SOFT = "soft"   # graded / k-WTA


#: Default suppression window length (ticks) for HARD mode
DEFAULT_SUPPRESS_TICKS: int = 3

#: Default inhibition strength (subtracted from membrane potential, SOFT mode)
DEFAULT_INHIBITION_STRENGTH: float = 0.5

#: Default Gaussian width for distance-based soft inhibition
DEFAULT_SIGMA: float = 1.5

#: Default inhibition radius (neurons beyond this distance are unaffected).
#: None means "no cutoff" — every neuron is reachable (radius = layer size).
DEFAULT_INHIBITION_RADIUS: Optional[int] = None


class Topology(Enum):
    LINEAR = "linear"  # distance = |i - j|, no wraparound
    RING = "ring"       # distance wraps around (circular layer)


# ──────────────────────────────────────────────
# 2.  DISTANCE FUNCTIONS
# ──────────────────────────────────────────────

def neuron_distance(i: int, j: int, n_neurons: int, topology: Topology) -> int:
    """
    Compute the topological distance between two neuron indices.

    Parameters
    ----------
    i : int
        First neuron index.
    j : int
        Second neuron index.
    n_neurons : int
        Total neurons in the layer (needed for RING wraparound).
    topology : Topology
        LINEAR (no wraparound) or RING (circular).

    Returns
    -------
    int
        Non-negative distance between i and j under the given topology.
    """
    raw = abs(i - j)
    if topology == Topology.LINEAR:
        return raw
    # RING: shortest path around the circle
    return min(raw, n_neurons - raw)


# ──────────────────────────────────────────────
# 3.  HARD WINNER-TAKE-ALL INHIBITOR
# ──────────────────────────────────────────────

@dataclass
class HardLateralInhibitor:
    """
    Hard winner-take-all lateral inhibition: the first neuron to spike
    in a tick suppresses every other neuron in the layer for a fixed
    number of subsequent ticks.
    """
    n_neurons: int
    suppress_ticks: int = DEFAULT_SUPPRESS_TICKS

    suppression_counters: List[int] = field(init=False)
    total_suppressions: int = field(init=False, default=0)
    winner_history: List[Optional[int]] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.suppression_counters = [0] * self.n_neurons

    def apply(self, spikes: List[int]) -> List[int]:
        """
        Apply WTA inhibition to a tick's spike vector.

        If multiple neurons spike in the same tick (a tie), the
        lowest-index neuron is declared the winner (matches a simple
        fixed-priority arbiter, as would be implemented in RTL).

        Parameters
        ----------
        spikes : List[int]
            Raw spike vector for this tick, BEFORE inhibition is applied
            (i.e. straight out of the membrane threshold check).

        Returns
        -------
        List[int]
            Inhibited spike vector: at most one neuron may spike (the
            winner), and only if it was not itself already suppressed
            by an earlier winner's window.
        """
        if len(spikes) != self.n_neurons:
            raise ValueError(
                f"spikes length {len(spikes)} != n_neurons {self.n_neurons}"
            )

        # Decay existing suppression counters first
        for i in range(self.n_neurons):
            if self.suppression_counters[i] > 0:
                self.suppression_counters[i] -= 1

        # Mask out spikes from currently-suppressed neurons
        eligible = [
            s if self.suppression_counters[i] == 0 else 0
            for i, s in enumerate(spikes)
        ]

        winner = None
        for i, s in enumerate(eligible):
            if s == 1:
                winner = i
                break  # fixed-priority: lowest index wins ties

        output = [0] * self.n_neurons
        if winner is not None:
            output[winner] = 1
            for i in range(self.n_neurons):
                if i != winner:
                    self.suppression_counters[i] = self.suppress_ticks
            self.total_suppressions += (self.n_neurons - 1)

        self.winner_history.append(winner)
        return output

    def is_suppressed(self, neuron_id: int) -> bool:
        """True if the given neuron is currently within its suppression window."""
        return self.suppression_counters[neuron_id] > 0

    def reset(self) -> None:
        """Clear all suppression counters (does not clear history/stats)."""
        self.suppression_counters = [0] * self.n_neurons

    def stats(self) -> dict:
        """Return WTA inhibitor statistics."""
        n_ticks_with_winner = sum(1 for w in self.winner_history if w is not None)
        return {
            "n_neurons": self.n_neurons,
            "total_ticks": len(self.winner_history),
            "ticks_with_winner": n_ticks_with_winner,
            "total_suppressions": self.total_suppressions,
            "currently_suppressed": sum(1 for c in self.suppression_counters if c > 0),
        }


# ──────────────────────────────────────────────
# 4.  SOFT (GRADED / k-WTA) LATERAL INHIBITOR
# ──────────────────────────────────────────────

@dataclass
class SoftLateralInhibitor:
    """
    Soft (graded) lateral inhibition: a spiking neuron applies a
    distance-weighted membrane-potential penalty to its neighbors,
    rather than fully silencing them.
    """
    n_neurons: int
    inhibition_strength: float = DEFAULT_INHIBITION_STRENGTH
    sigma: float = DEFAULT_SIGMA
    inhibition_radius: Optional[int] = DEFAULT_INHIBITION_RADIUS
    topology: Topology = Topology.LINEAR

    total_penalty_applied: float = field(init=False, default=0.0)
    apply_count: int = field(init=False, default=0)

    def compute_penalty_vector(self, spikes: List[int]) -> List[float]:
        """
        Compute the membrane-potential penalty each neuron should
        receive, given this tick's spike vector.

        Parameters
        ----------
        spikes : List[int]
            Binary spike vector for this tick (which neurons just fired).

        Returns
        -------
        List[float]
            Non-negative penalty per neuron (to be SUBTRACTED from each
            neuron's membrane potential by the caller). A spiking
            neuron never penalizes itself.
        """
        if len(spikes) != self.n_neurons:
            raise ValueError(
                f"spikes length {len(spikes)} != n_neurons {self.n_neurons}"
            )

        penalties = [0.0] * self.n_neurons
        winners = [i for i, s in enumerate(spikes) if s == 1]

        for w in winners:
            for j in range(self.n_neurons):
                if j == w:
                    continue  # self-exclusion

                dist = neuron_distance(j, w, self.n_neurons, self.topology)

                if self.inhibition_radius is not None and dist > self.inhibition_radius:
                    continue  # outside inhibition neighborhood

                penalty = self.inhibition_strength * math.exp(
                    -(dist * dist) / (2.0 * self.sigma * self.sigma)
                )
                penalties[j] += penalty

        return penalties

    def apply(self, v_mem: List[float], spikes: List[int]) -> List[float]:
        """
        Apply soft lateral inhibition to a membrane-potential vector.

        Parameters
        ----------
        v_mem : List[float]
            Current membrane potentials, length n_neurons.
        spikes : List[int]
            This tick's spike vector (which neurons fired, BEFORE
            inhibition — used to determine who inhibits whom).

        Returns
        -------
        List[float]
            Membrane potentials after subtracting lateral-inhibition
            penalties (never driven below the original v_mem - total
            penalty; caller may further clamp to v_reset if desired).
        """
        if len(v_mem) != self.n_neurons:
            raise ValueError(
                f"v_mem length {len(v_mem)} != n_neurons {self.n_neurons}"
            )

        penalties = self.compute_penalty_vector(spikes)
        self.total_penalty_applied += sum(penalties)
        self.apply_count += 1

        return [v - p for v, p in zip(v_mem, penalties)]

    def stats(self) -> dict:
        """Return soft inhibitor statistics."""
        return {
            "n_neurons": self.n_neurons,
            "apply_count": self.apply_count,
            "total_penalty_applied": self.total_penalty_applied,
            "avg_penalty_per_apply": (
                self.total_penalty_applied / self.apply_count
                if self.apply_count > 0 else 0.0
            ),
        }


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("lateral_inhibitor.py  —  self-test")
    print("=" * 55)

    # Test 1: Distance functions (linear vs ring)
    print("\n[ Distance functions ]")
    assert neuron_distance(0, 7, 8, Topology.LINEAR) == 7
    assert neuron_distance(0, 7, 8, Topology.RING) == 1  # wraps around
    assert neuron_distance(2, 5, 8, Topology.LINEAR) == 3
    print(f"  ✓ LINEAR(0,7)=7, RING(0,7)=1 (wraparound), LINEAR(2,5)=3")

    # Test 2: Hard WTA — single spike passes through unchanged
    print("\n[ Hard WTA — single spike ]")
    wta = HardLateralInhibitor(n_neurons=5, suppress_ticks=2)
    spikes_in = [0, 0, 1, 0, 0]
    out = wta.apply(spikes_in)
    assert out == spikes_in
    print(f"  ✓ Single spike at idx 2 passes through: {out}")

    # Test 3: Hard WTA — tie resolved by lowest index
    print("\n[ Hard WTA — tie resolution ]")
    wta2 = HardLateralInhibitor(n_neurons=5, suppress_ticks=2)
    tied = [0, 1, 1, 0, 1]
    out2 = wta2.apply(tied)
    assert out2 == [0, 1, 0, 0, 0]  # lowest-index winner only
    print(f"  ✓ Three-way tie {tied} → winner idx 1: {out2}")

    # Test 4: Hard WTA — winner suppresses others on subsequent ticks
    print("\n[ Hard WTA — suppression window ]")
    wta3 = HardLateralInhibitor(n_neurons=4, suppress_ticks=2)
    wta3.apply([1, 0, 0, 0])  # neuron 0 wins, suppresses 1,2,3 for 2 ticks
    assert wta3.is_suppressed(1) and wta3.is_suppressed(2) and wta3.is_suppressed(3)
    assert not wta3.is_suppressed(0)
    out_next = wta3.apply([0, 1, 0, 0])  # neuron 1 tries to fire while suppressed
    assert out_next == [0, 0, 0, 0]  # blocked
    print(f"  ✓ Suppressed neuron 1 correctly blocked from firing: {out_next}")

    # Test 5: Hard WTA — suppression expires after window
    print("\n[ Hard WTA — suppression expiry ]")
    wta4 = HardLateralInhibitor(n_neurons=3, suppress_ticks=1)
    wta4.apply([1, 0, 0])  # neuron 0 wins, suppresses 1,2 for 1 tick
    wta4.apply([0, 0, 0])  # tick passes, suppression decays
    out_after = wta4.apply([0, 1, 0])  # neuron 1 should now be free to fire
    assert out_after == [0, 1, 0]
    print(f"  ✓ Suppression expired after window, neuron 1 fires: {out_after}")

    # Test 6: Hard WTA — no spikes, no winner
    print("\n[ Hard WTA — silent tick ]")
    wta5 = HardLateralInhibitor(n_neurons=4)
    out_silent = wta5.apply([0, 0, 0, 0])
    assert out_silent == [0, 0, 0, 0]
    assert wta5.winner_history[-1] is None
    print(f"  ✓ Silent tick produces no winner")

    # Test 7: Soft inhibition — penalty peaks at winner's neighbors, zero at self
    print("\n[ Soft inhibition — penalty distribution ]")
    soft = SoftLateralInhibitor(
        n_neurons=7, inhibition_strength=1.0, sigma=1.0, topology=Topology.LINEAR
    )
    spikes_soft = [0, 0, 0, 1, 0, 0, 0]  # neuron 3 fires
    penalties = soft.compute_penalty_vector(spikes_soft)
    assert penalties[3] == 0.0  # self-exclusion
    assert penalties[2] == penalties[4]  # symmetric around winner
    assert penalties[2] > penalties[1]  # closer neighbor penalized more
    print(f"  ✓ Penalties: {[f'{p:.3f}' for p in penalties]} "
          f"(symmetric, zero at winner, decaying with distance)")

    # Test 8: Soft inhibition — apply to membrane potentials
    print("\n[ Soft inhibition — membrane potential adjustment ]")
    v_mem = [0.5, 0.6, 0.7, 1.2, 0.7, 0.6, 0.5]
    v_mem_after = soft.apply(v_mem, spikes_soft)
    assert v_mem_after[3] == v_mem[3]  # winner unaffected
    assert v_mem_after[2] < v_mem[2]    # neighbor reduced
    assert v_mem_after[0] < v_mem[0]    # even distant neuron slightly reduced
    print(f"  ✓ V_mem before: {v_mem}")
    print(f"  ✓ V_mem after:  {[round(v, 3) for v in v_mem_after]}")

    # Test 9: Soft inhibition — radius cutoff
    print("\n[ Soft inhibition — radius cutoff ]")
    soft_radius = SoftLateralInhibitor(
        n_neurons=10, inhibition_strength=1.0, sigma=1.0,
        inhibition_radius=2, topology=Topology.LINEAR
    )
    spikes_r = [0] * 10
    spikes_r[5] = 1
    penalties_r = soft_radius.compute_penalty_vector(spikes_r)
    assert penalties_r[7] > 0.0   # distance 2 → exactly at radius boundary, still included
    assert penalties_r[9] == 0.0  # distance 4 → outside radius=2, must be exactly zero
    assert penalties_r[6] > 0.0   # distance 1 → within radius, nonzero
    print(f"  ✓ Radius=2 cutoff: neuron at distance 4 has penalty={penalties_r[9]}, "
          f"neuron at distance 2 (boundary) has penalty={penalties_r[7]:.3f}")

    # Test 10: Soft inhibition — ring topology wraps around
    print("\n[ Soft inhibition — ring topology ]")
    soft_ring = SoftLateralInhibitor(
        n_neurons=8, inhibition_strength=1.0, sigma=1.0, topology=Topology.RING
    )
    spikes_ring = [1, 0, 0, 0, 0, 0, 0, 0]  # neuron 0 fires
    penalties_ring = soft_ring.compute_penalty_vector(spikes_ring)
    # neuron 7 is distance 1 from neuron 0 on a ring (wraps around)
    assert abs(penalties_ring[7] - penalties_ring[1]) < 1e-9
    print(f"  ✓ Ring topology: neuron 7 and neuron 1 (both distance 1 from "
          f"winner 0) get equal penalty {penalties_ring[1]:.3f}")

    # Test 11: Multiple simultaneous winners (soft mode, k-WTA style)
    print("\n[ Soft inhibition — multiple winners ]")
    soft_multi = SoftLateralInhibitor(n_neurons=6, inhibition_strength=0.5, sigma=1.0)
    multi_spikes = [1, 0, 0, 0, 1, 0]  # neurons 0 and 4 both fire
    penalties_multi = soft_multi.compute_penalty_vector(multi_spikes)
    # Self-exclusion means winner 0 contributes nothing to itself, and winner 4
    # contributes nothing to itself — but they DO still inhibit each other
    # (distance 4 apart), which is correct competitive behavior, not a bug.
    assert penalties_multi[0] > 0.0   # inhibited by the OTHER winner (idx 4)
    assert penalties_multi[4] > 0.0   # inhibited by the OTHER winner (idx 0)
    assert penalties_multi[2] > 0.0   # influenced by both winners
    assert penalties_multi[2] > penalties_multi[0]  # neuron 2 is closer to both winners
    print(f"  ✓ Two simultaneous winners (0,4) cross-inhibit each other "
          f"({penalties_multi[0]:.4f}); neuron 2 gets larger combined "
          f"penalty {penalties_multi[2]:.3f} from proximity to both")

    # Test 12: Dimension validation (both inhibitor types)
    print("\n[ Dimension validation ]")
    try:
        wta.apply([1, 0])  # wrong length for n_neurons=5
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ HardLateralInhibitor rejects mismatched spike vector")

    try:
        soft.compute_penalty_vector([1, 0])  # wrong length for n_neurons=7
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ SoftLateralInhibitor rejects mismatched spike vector")

    # Test 13: Statistics
    print("\n[ Statistics ]")
    wta_stats = wta3.stats()
    soft_stats = soft.stats()
    print(f"  WTA: {wta_stats}")
    print(f"  Soft: apply_count={soft_stats['apply_count']}, "
          f"avg_penalty={soft_stats['avg_penalty_per_apply']:.3f}")
    print(f"  ✓ Statistics tracked correctly for both inhibitor types")

    print("\n  All checks passed.\n")
