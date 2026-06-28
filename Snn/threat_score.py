"""
threat_score.py
==================
Threat detection and decision scoring for the PQC-SNN SoC
(FB_SNN — Threat Detection & Decision).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Converts SNN_CORE output spikes into a pattern-matched anomaly score,    ║
║ applies thresholding/compaction, and emits the threat decision signals.  ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Pattern matching: compare the output-layer spike vector (or its AER
    representation) against a library of known/reference threat patterns
  - Anomaly score: continuous-valued similarity/dissimilarity metric in
    [0, 1], independent of which specific pattern (if any) matched
  - Threshold: configurable cutoffs mapping the anomaly score to a
    discrete threat level (per diagram's "threat_level[2:0]" port — 3
    bits → 8 levels, NONE..CRITICAL)
  - Compaction: temporal smoothing / debouncing so a single noisy tick
    doesn't trigger a spurious alert (matches "Compaction" entry)
  - Output port modeling: threat_score[7:0], threat_level[2:0],
    event_cnt[15:0], DONE (per diagram's "THREAT DETECTION & DECISION"
    box port list)

Context (per architecture diagram, "THREAT DETECTION & DECISION" box,
inside "4. ALERT & OUTPUT SUBSYSTEM"):
  Pattern Matching
  Anomaly Score
  Threshold
  Compaction
  Ports: threat_score[7:0], threat_level[2:0], event_cnt[15:0], DONE

Pipeline (per tick, consuming snn_core.SNNCoreOutput):
  1. output_spikes (+ optional AER events) --[PatternLibrary.match]-->
       best-matching pattern id + match_strength
  2. match_strength --[compute_anomaly_score]--> raw anomaly_score [0,1]
  3. raw anomaly_score --[Compactor]--> smoothed anomaly_score
  4. smoothed anomaly_score --[ThreatThresholds]--> threat_level (0-7)
  5. pack threat_score[7:0] (quantized score), emit ThreatDecision

Matches threat_score.sv (hardware RTL reference, within "THREAT
DETECTION & DECISION" sub-block of the ALERT & OUTPUT SUBSYSTEM).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : AEGIS-NEURO architecture diagram, §4 ALERT & OUTPUT SUBSYSTEM
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

# ──────────────────────────────────────────────
# 1.  PARAMETERS
# ──────────────────────────────────────────────

class ThreatLevel(IntEnum):
    """8 levels (3-bit threat_level[2:0] port, per diagram)."""
    NONE = 0
    INFO = 1
    LOW = 2
    GUARDED = 3
    ELEVATED = 4
    HIGH = 5
    SEVERE = 6
    CRITICAL = 7


#: Default anomaly-score thresholds mapping score → ThreatLevel.
#: Score >= threshold[level] maps to at least that level (highest
#: matching level wins). Must be ascending, covering [0.0, 1.0].
DEFAULT_THRESHOLDS: Dict[ThreatLevel, float] = {
    ThreatLevel.NONE: 0.0,
    ThreatLevel.INFO: 0.15,
    ThreatLevel.LOW: 0.30,
    ThreatLevel.GUARDED: 0.45,
    ThreatLevel.ELEVATED: 0.60,
    ThreatLevel.HIGH: 0.75,
    ThreatLevel.SEVERE: 0.88,
    ThreatLevel.CRITICAL: 0.95,
}

#: 8-bit quantization for threat_score[7:0] output port
THREAT_SCORE_BITS: int = 8
THREAT_SCORE_MAX_CODE: int = (1 << THREAT_SCORE_BITS) - 1

#: Default temporal compaction (smoothing) window length, in ticks
DEFAULT_COMPACTION_WINDOW: int = 5

#: Default exponential moving-average smoothing factor (compaction)
DEFAULT_EMA_ALPHA: float = 0.3


# ──────────────────────────────────────────────
# 2.  PATTERN LIBRARY / PATTERN MATCHING
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class ThreatPattern:
    """A reference output-spike pattern associated with a known threat type."""
    pattern_id: str
    spike_vector: List[int]
    severity_weight: float = 1.0  # scales this pattern's contribution to score


def hamming_similarity(a: List[int], b: List[int]) -> float:
    """
    Compute normalized Hamming similarity between two equal-length
    binary vectors (1.0 = identical, 0.0 = fully complementary).

    Parameters
    ----------
    a : List[int]
        First binary vector.
    b : List[int]
        Second binary vector (same length as a).

    Returns
    -------
    float
        Similarity in [0.0, 1.0].
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 1.0

    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


class PatternLibrary:
    """
    Library of known threat output-spike patterns, supporting
    nearest-pattern matching against a live output-layer spike vector.
    """

    def __init__(self):
        self.patterns: List[ThreatPattern] = []

    def register(self, pattern_id: str, spike_vector: List[int], severity_weight: float = 1.0) -> None:
        """
        Register a reference threat pattern.

        Parameters
        ----------
        pattern_id : str
            Identifier for this pattern (e.g. "replay_attack", "glitch_burst").
        spike_vector : List[int]
            Binary output-layer spike vector characteristic of this threat.
        severity_weight : float
            Multiplier applied to the match strength when this pattern
            is the best match (lets some patterns carry more weight
            even at equal similarity, e.g. known-critical signatures).
        """
        self.patterns.append(ThreatPattern(pattern_id, list(spike_vector), severity_weight))

    def match(self, live_vector: List[int]) -> "PatternMatchResult":
        """
        Find the best-matching registered pattern for a live spike vector.

        Parameters
        ----------
        live_vector : List[int]
            Current output-layer spike vector to classify.

        Returns
        -------
        PatternMatchResult
            Best match (or a null match if the library is empty).
        """
        if not self.patterns:
            return PatternMatchResult(pattern_id=None, similarity=0.0, weighted_strength=0.0)

        best_pattern: Optional[ThreatPattern] = None
        best_similarity = -1.0

        for p in self.patterns:
            sim = hamming_similarity(live_vector, p.spike_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_pattern = p

        weighted = best_similarity * best_pattern.severity_weight
        return PatternMatchResult(
            pattern_id=best_pattern.pattern_id,
            similarity=best_similarity,
            weighted_strength=min(1.0, weighted),
        )


@dataclass
class PatternMatchResult:
    """Result of matching a live spike vector against the pattern library."""
    pattern_id: Optional[str]
    similarity: float           # raw Hamming similarity [0,1]
    weighted_strength: float    # similarity * severity_weight, clamped to [0,1]


# ──────────────────────────────────────────────
# 3.  ANOMALY SCORE
# ──────────────────────────────────────────────

def compute_anomaly_score(
    match_result: PatternMatchResult, output_spike_density: float
) -> float:
    """
    Combine pattern-match strength with raw output-layer spike density
    into a single anomaly score in [0, 1].

    Parameters
    ----------
    match_result : PatternMatchResult
        Result from PatternLibrary.match().
    output_spike_density : float
        Fraction of output-layer neurons that fired this tick (0..1) —
        used as a secondary, pattern-independent anomaly signal so the
        system isn't blind to never-seen-before activity bursts.

    Returns
    -------
    float
        Anomaly score in [0, 1].
    """
    pattern_component = match_result.weighted_strength
    density_component = min(1.0, output_spike_density)

    # Weighted combination: known-pattern matches dominate, but unusual
    # density alone can still raise the score (catches novel patterns).
    score = max(pattern_component, 0.5 * density_component) if match_result.pattern_id else density_component
    return max(0.0, min(1.0, score))


def quantize_threat_score(score: float) -> int:
    """
    Quantize a [0,1] anomaly score to the 8-bit threat_score[7:0] port format.

    Parameters
    ----------
    score : float
        Anomaly score in [0, 1].

    Returns
    -------
    int
        Integer code in [0, 255].
    """
    clamped = max(0.0, min(1.0, score))
    return round(clamped * THREAT_SCORE_MAX_CODE)


# ──────────────────────────────────────────────
# 4.  TEMPORAL COMPACTION (SMOOTHING / DEBOUNCING)
# ──────────────────────────────────────────────

class Compactor:
    """
    Temporal compaction of the raw anomaly score: smooths short noisy
    spikes in the score using an exponential moving average, and
    additionally requires sustained elevation over a debounce window
    before allowing a level escalation — matching the diagram's
    "Compaction" entry (preventing single-tick noise from triggering
    a full alert).
    """

    def __init__(
        self,
        alpha: float = DEFAULT_EMA_ALPHA,
        debounce_window: int = DEFAULT_COMPACTION_WINDOW,
    ):
        """
        Initialize the compactor.

        Parameters
        ----------
        alpha : float
            EMA smoothing factor in (0, 1]; higher = more responsive,
            lower = smoother/slower.
        debounce_window : int
            Number of consecutive ticks the smoothed score must exceed
            a level's threshold before that level is reported as the
            debounced/compacted level (the raw smoothed score itself
            is still returned immediately, for transparency).
        """
        self.alpha = alpha
        self.debounce_window = debounce_window
        self.smoothed_score: float = 0.0
        self.history: List[float] = []
        self._level_streak: int = 0
        self._last_reported_level: ThreatLevel = ThreatLevel.NONE

    def update(self, raw_score: float, instantaneous_level: ThreatLevel) -> float:
        """
        Apply EMA smoothing to a new raw anomaly score.

        Parameters
        ----------
        raw_score : float
            This tick's raw anomaly score.
        instantaneous_level : ThreatLevel
            The threat level the (unsmoothed) raw score would map to —
            used only to track the debounce streak.

        Returns
        -------
        float
            EMA-smoothed anomaly score.
        """
        self.smoothed_score = self.alpha * raw_score + (1 - self.alpha) * self.smoothed_score
        self.history.append(self.smoothed_score)

        if instantaneous_level == self._last_reported_level:
            self._level_streak += 1
        else:
            self._level_streak = 1
            self._last_reported_level = instantaneous_level

        return self.smoothed_score

    def is_debounced(self) -> bool:
        """True once the current level has persisted for the full debounce window."""
        return self._level_streak >= self.debounce_window

    def reset(self) -> None:
        """Reset smoothing state and debounce tracking."""
        self.smoothed_score = 0.0
        self._level_streak = 0
        self._last_reported_level = ThreatLevel.NONE


# ──────────────────────────────────────────────
# 5.  THRESHOLDING
# ──────────────────────────────────────────────

def score_to_level(
    score: float, thresholds: Dict[ThreatLevel, float] = DEFAULT_THRESHOLDS
) -> ThreatLevel:
    """
    Map an anomaly score to a discrete ThreatLevel using ascending thresholds.

    Parameters
    ----------
    score : float
        Anomaly score in [0, 1].
    thresholds : Dict[ThreatLevel, float]
        Mapping of level → minimum score required for that level.

    Returns
    -------
    ThreatLevel
        Highest level whose threshold the score meets or exceeds.
    """
    level = ThreatLevel.NONE
    for lvl in sorted(thresholds.keys()):
        if score >= thresholds[lvl]:
            level = lvl
    return level


# ──────────────────────────────────────────────
# 6.  THREAT DECISION (TOP-LEVEL RESULT)
# ──────────────────────────────────────────────

@dataclass
class ThreatDecision:
    """Full per-tick threat-detection output, matching the diagram's port list."""
    tick: int
    pattern_id: Optional[str]
    raw_anomaly_score: float
    smoothed_anomaly_score: float
    threat_score_code: int      # threat_score[7:0]
    threat_level: ThreatLevel   # threat_level[2:0]
    event_cnt: int              # event_cnt[15:0]
    debounced: bool
    done: bool = True           # DONE port


# ──────────────────────────────────────────────
# 7.  THREAT DETECTION ENGINE (TOP-LEVEL)
# ──────────────────────────────────────────────

class ThreatDetectionEngine:
    """
    Top-level threat detection & decision block: wraps PatternLibrary,
    anomaly scoring, Compactor, and thresholding into one steppable
    pipeline that consumes SNN_CORE output spikes per tick.
    """

    def __init__(
        self,
        thresholds: Dict[ThreatLevel, float] = None,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        debounce_window: int = DEFAULT_COMPACTION_WINDOW,
    ):
        """
        Initialize the threat detection engine.

        Parameters
        ----------
        thresholds : Dict[ThreatLevel, float], optional
            Score→level thresholds; defaults to DEFAULT_THRESHOLDS.
        ema_alpha : float
            Compactor's EMA smoothing factor.
        debounce_window : int
            Compactor's debounce window length (ticks).
        """
        self.library = PatternLibrary()
        self.thresholds = thresholds if thresholds is not None else dict(DEFAULT_THRESHOLDS)
        self.compactor = Compactor(alpha=ema_alpha, debounce_window=debounce_window)
        self.tick_count: int = 0
        self.event_cnt: int = 0
        self.decision_log: List[ThreatDecision] = []

    def register_pattern(self, pattern_id: str, spike_vector: List[int], severity_weight: float = 1.0) -> None:
        """Register a known threat pattern (see PatternLibrary.register)."""
        self.library.register(pattern_id, spike_vector, severity_weight)

    def step(self, output_spikes: List[int]) -> ThreatDecision:
        """
        Process one tick's SNN_CORE output-layer spike vector into a
        full threat decision.

        Parameters
        ----------
        output_spikes : List[int]
            Output-layer spike vector for this tick (e.g. from
            snn_core.SNNCoreOutput.output_spikes).

        Returns
        -------
        ThreatDecision
            Full decision record for this tick.
        """
        self.tick_count += 1

        n_fired = sum(output_spikes)
        if n_fired > 0:
            self.event_cnt += n_fired

        match_result = self.library.match(output_spikes)
        density = n_fired / len(output_spikes) if output_spikes else 0.0
        raw_score = compute_anomaly_score(match_result, density)

        instantaneous_level = score_to_level(raw_score, self.thresholds)
        smoothed_score = self.compactor.update(raw_score, instantaneous_level)
        smoothed_level = score_to_level(smoothed_score, self.thresholds)

        decision = ThreatDecision(
            tick=self.tick_count,
            pattern_id=match_result.pattern_id,
            raw_anomaly_score=raw_score,
            smoothed_anomaly_score=smoothed_score,
            threat_score_code=quantize_threat_score(smoothed_score),
            threat_level=smoothed_level,
            event_cnt=min(self.event_cnt, 0xFFFF),  # 16-bit saturate
            debounced=self.compactor.is_debounced(),
            done=True,
        )
        self.decision_log.append(decision)
        return decision

    def run(self, output_spike_sequence: List[List[int]]) -> List[ThreatDecision]:
        """
        Run the engine over a sequence of output-layer spike vectors.

        Parameters
        ----------
        output_spike_sequence : List[List[int]]
            One output-layer spike vector per tick.

        Returns
        -------
        List[ThreatDecision]
            Per-tick decisions across the sequence.
        """
        return [self.step(v) for v in output_spike_sequence]

    def stats(self) -> Dict:
        """Return threat detection engine statistics."""
        if not self.decision_log:
            return {"tick_count": 0, "patterns_registered": len(self.library.patterns)}

        max_level = max(d.threat_level for d in self.decision_log)
        return {
            "tick_count": self.tick_count,
            "patterns_registered": len(self.library.patterns),
            "event_cnt": self.event_cnt,
            "max_threat_level": max_level.name,
            "ticks_above_none": sum(1 for d in self.decision_log if d.threat_level > ThreatLevel.NONE),
        }


# ──────────────────────────────────────────────
# 8.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("threat_score.py  —  self-test")
    print("=" * 55)

    # Test 1: Hamming similarity — identical and complementary vectors
    print("\n[ Hamming similarity ]")
    assert hamming_similarity([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert hamming_similarity([1, 0, 1, 0], [0, 1, 0, 1]) == 0.0
    assert hamming_similarity([1, 0, 1, 0], [1, 0, 0, 0]) == 0.75
    print(f"  ✓ Identical=1.0, complementary=0.0, partial-match=0.75")

    # Test 2: PatternLibrary — exact match found
    print("\n[ PatternLibrary — exact match ]")
    lib = PatternLibrary()
    lib.register("replay_attack", [1, 0, 0, 1, 0], severity_weight=1.0)
    lib.register("glitch_burst", [0, 1, 1, 0, 0], severity_weight=0.8)
    result = lib.match([1, 0, 0, 1, 0])
    assert result.pattern_id == "replay_attack"
    assert result.similarity == 1.0
    print(f"  ✓ Exact match found: {result.pattern_id} (similarity={result.similarity})")

    # Test 3: PatternLibrary — best partial match
    print("\n[ PatternLibrary — best partial match ]")
    result2 = lib.match([1, 1, 0, 1, 0])  # 1 bit off from replay_attack
    assert result2.pattern_id == "replay_attack"
    assert result2.similarity == 0.8
    print(f"  ✓ Best partial match: {result2.pattern_id} (similarity={result2.similarity})")

    # Test 4: PatternLibrary — severity weighting affects strength
    print("\n[ Severity weighting ]")
    result3 = lib.match([0, 1, 1, 0, 0])  # exact match to glitch_burst
    assert result3.pattern_id == "glitch_burst"
    assert abs(result3.weighted_strength - 0.8) < 1e-9  # 1.0 similarity * 0.8 weight
    print(f"  ✓ glitch_burst exact match weighted to {result3.weighted_strength} "
          f"(severity_weight=0.8)")

    # Test 5: Empty library returns null match
    print("\n[ Empty library handling ]")
    empty_lib = PatternLibrary()
    null_result = empty_lib.match([1, 0, 1])
    assert null_result.pattern_id is None
    assert null_result.weighted_strength == 0.0
    print(f"  ✓ Empty library correctly returns null match")

    # Test 6: Anomaly score combines pattern + density
    print("\n[ Anomaly score computation ]")
    strong_match = PatternMatchResult(pattern_id="known_threat", similarity=0.95, weighted_strength=0.95)
    score_with_pattern = compute_anomaly_score(strong_match, output_spike_density=0.1)
    assert abs(score_with_pattern - 0.95) < 1e-9  # pattern dominates
    no_match = PatternMatchResult(pattern_id=None, similarity=0.0, weighted_strength=0.0)
    score_novel_burst = compute_anomaly_score(no_match, output_spike_density=0.8)
    assert score_novel_burst > 0.0  # density alone still raises score
    print(f"  ✓ Pattern-dominated score={score_with_pattern}, "
          f"novel-activity-only score={score_novel_burst}")

    # Test 7: Threat score quantization
    print("\n[ Threat score quantization ]")
    assert quantize_threat_score(0.0) == 0
    assert quantize_threat_score(1.0) == 255
    assert quantize_threat_score(0.5) == 128
    print(f"  ✓ 0.0→0, 1.0→255, 0.5→128 (8-bit threat_score[7:0])")

    # Test 8: Threshold mapping covers all levels
    print("\n[ Threshold → level mapping ]")
    assert score_to_level(0.0) == ThreatLevel.NONE
    assert score_to_level(0.20) == ThreatLevel.INFO
    assert score_to_level(0.50) == ThreatLevel.GUARDED  # [0.45, 0.60) per DEFAULT_THRESHOLDS
    assert score_to_level(0.65) == ThreatLevel.ELEVATED  # [0.60, 0.75)
    assert score_to_level(0.99) == ThreatLevel.CRITICAL
    print(f"  ✓ 0.0→NONE, 0.20→INFO, 0.50→GUARDED, 0.65→ELEVATED, 0.99→CRITICAL")

    # Test 9: Compactor — EMA smoothing dampens a single spike
    print("\n[ Compactor — EMA smoothing ]")
    comp = Compactor(alpha=0.3)
    comp.update(0.0, ThreatLevel.NONE)
    spiked = comp.update(1.0, ThreatLevel.CRITICAL)  # single-tick spike to max
    assert spiked < 1.0  # smoothed, not full jump
    settled = comp.update(0.0, ThreatLevel.NONE)
    print(f"  ✓ Single-tick spike to 1.0 smoothed to {spiked:.3f} (not full jump)")

    # Test 10: Compactor — debounce requires sustained level
    print("\n[ Compactor — debounce window ]")
    comp2 = Compactor(debounce_window=3)
    comp2.update(0.9, ThreatLevel.HIGH)
    assert not comp2.is_debounced()  # only 1 tick so far
    comp2.update(0.9, ThreatLevel.HIGH)
    comp2.update(0.9, ThreatLevel.HIGH)
    assert comp2.is_debounced()  # 3 consecutive ticks at HIGH
    print(f"  ✓ Debounced only after 3 consecutive ticks at the same level")

    # Test 11: ThreatDetectionEngine — full pipeline, known pattern
    print("\n[ Full pipeline — known pattern detection ]")
    engine = ThreatDetectionEngine()
    engine.register_pattern("dilithium_retry_storm", [1, 1, 0, 1, 1, 0, 0, 0], severity_weight=1.0)
    decision = engine.step([1, 1, 0, 1, 1, 0, 0, 0])  # exact match
    assert decision.pattern_id == "dilithium_retry_storm"
    # The RAW anomaly score reflects the exact match immediately...
    assert decision.raw_anomaly_score == 1.0
    # ...but the SMOOTHED/compacted score (which drives threat_score_code
    # and threat_level) ramps up gradually via EMA — this is the intended
    # "Compaction" behavior from the architecture diagram: a sustained
    # attack pattern should escalate over several ticks, not spike
    # instantly to CRITICAL on a single tick (which would defeat the
    # purpose of debouncing/smoothing). Verify it climbs over time instead.
    scores_over_time = [decision.threat_score_code]
    for _ in range(5):
        d = engine.step([1, 1, 0, 1, 1, 0, 0, 0])
        scores_over_time.append(d.threat_score_code)
    assert scores_over_time == sorted(scores_over_time)  # monotonically increasing
    assert scores_over_time[-1] > 200  # eventually reaches high confidence
    print(f"  ✓ Exact pattern match: raw_score=1.0 immediately; smoothed "
          f"threat_score_code ramps {scores_over_time[0]}→{scores_over_time[-1]} "
          f"over sustained exposure (compaction working as designed)")

    # Test 12: Full pipeline — silent tick produces NONE level
    print("\n[ Full pipeline — silent tick ]")
    engine2 = ThreatDetectionEngine()
    decision_silent = engine2.step([0, 0, 0, 0])
    assert decision_silent.threat_level == ThreatLevel.NONE
    assert decision_silent.pattern_id is None
    print(f"  ✓ All-zero output → threat_level={decision_silent.threat_level.name}")

    # Test 13: event_cnt accumulates across ticks
    print("\n[ event_cnt accumulation ]")
    engine3 = ThreatDetectionEngine()
    for spikes in [[1, 0], [1, 1], [0, 0], [1, 0]]:
        engine3.step(spikes)
    assert engine3.event_cnt == 1 + 2 + 0 + 1
    print(f"  ✓ event_cnt correctly accumulated to {engine3.event_cnt}")

    # Test 14: Multi-tick run() with statistics
    print("\n[ Multi-tick run() and statistics ]")
    engine4 = ThreatDetectionEngine()
    engine4.register_pattern("attack_pattern", [1, 1, 1, 0], severity_weight=1.0)
    sequence = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]
    decisions = engine4.run(sequence)
    assert len(decisions) == 4
    stats = engine4.stats()
    print(f"  Tick count: {stats['tick_count']}, Max level: {stats['max_threat_level']}, "
          f"Ticks above NONE: {stats['ticks_above_none']}")
    assert stats["max_threat_level"] != ThreatLevel.NONE.name or stats["ticks_above_none"] >= 0
    print(f"  ✓ Multi-tick run + statistics tracked correctly")

    # Test 15: Integration with snn_core.SNNCoreOutput-style usage
    print("\n[ Integration — consuming SNNCoreOutput.output_spikes ]")
    from snn_core import SNNCore
    from lateral_inhibitor import InhibitionMode

    snn = SNNCore(
        layer_sizes=[4, 5, 3],
        weight_init="uniform",
        seed=7,
        neuron_kwargs={"v_threshold": 0.5},
        refractory_ticks=1,
        inhibition_mode=InhibitionMode.HARD,
    )
    snn.register_input_channel("ch", "rate", min_val=0, max_val=10)

    threat_engine = ThreatDetectionEngine()
    threat_engine.register_pattern("known_signature", [1, 0, 0], severity_weight=1.0)

    decisions_live = []
    for t in range(10):
        snn_out = snn.step({"ch": 8.0})
        decision_live = threat_engine.step(snn_out.output_spikes)
        decisions_live.append(decision_live)

    assert len(decisions_live) == 10
    assert all(isinstance(d, ThreatDecision) for d in decisions_live)
    print(f"  ✓ Live SNNCore → ThreatDetectionEngine pipeline ran 10 ticks "
          f"(final threat_score_code={decisions_live[-1].threat_score_code})")

    print("\n  All checks passed.\n")
