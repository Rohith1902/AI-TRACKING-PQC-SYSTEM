"""
alert_manager.py
==================
Alert Manager for the PQC-SNN SoC (FB_ALM — Alert & Output Subsystem).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Consumes threat decisions and tamper/zeroize events, applies severity     ║
║ scoring and rate limiting, and dispatches alerts with full audit trail.   ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Severity scoring: maps ThreatLevel + event source into a unified
    severity scale (per diagram's "Severity Scoring" entry)
  - Timestamp: monotonic tick-based timestamping of every alert
  - Alert Generation Logic: decides whether a threat/tamper/zeroize
    event should actually produce an alert (vs. be suppressed)
  - Rate Limiter: caps alert throughput per time window so a storm of
    threats cannot flood downstream consumers (UART/network/log)
  - Zeroize_Done_Ack: acknowledges completion of a zeroization sequence
    (from zeroize_fsm.py) so the system can confirm the secure-wipe
    response to a critical alert actually completed

Context (per architecture diagram, "ALERT MANAGER (FB_ALM)" box, inside
"4. ALERT & OUTPUT SUBSYSTEM"):
  Severity Scoring
  Timestamp
  Alert Generation Logic
  Rate Limiter
  Zeroize_Done_Ack
  Output ports (downstream, handled by other alert/ modules):
    UART Message, GPIO/LED, Interrupt to PS, Network Packet, Log to Memory

Pipeline (per tick):
  1. ThreatDecision (from threat_score.py) or TamperEvent (from
     tamper_detect.py) arrives
  2. compute_severity() maps it to a unified Severity level
  3. Rate limiter checks if this severity class has budget remaining
     this window; if not, the alert is suppressed (counted, not dropped
     silently — visible via stats())
  4. If accepted: timestamp, assign alert_id, append to alert log
  5. If a CRITICAL alert triggers a zeroize sequence elsewhere, the
     caller later reports completion via acknowledge_zeroize(), which
     is recorded against the originating alert

Integration points:
  - threat_score.ThreatDecision   → input (SNN-detected threats)
  - tamper_detect.TamperEvent      → input (physical tamper events)
  - zeroize_fsm.ZeroizeFSM          → acknowledged via acknowledge_zeroize()
  - alert_formatter.py (next file) → consumes AlertManager.pending_alerts()
  - log_writer.py (next file)      → consumes AlertManager's alert log

Matches alert_manager.sv (hardware RTL reference, FB_ALM).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : AEGIS-NEURO architecture diagram, §4 ALERT & OUTPUT SUBSYSTEM
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

# ──────────────────────────────────────────────
# 1.  SEVERITY SCALE
# ──────────────────────────────────────────────

class Severity(IntEnum):
    """Unified severity scale across all alert sources (threat/tamper/system)."""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertSource(IntEnum):
    """Origin of an alert, for routing/formatting downstream."""
    THREAT_DETECTION = 0   # from threat_score.ThreatDecision
    TAMPER = 1              # from tamper_detect.TamperEvent
    ZEROIZE = 2             # from zeroize_fsm completion/failure
    SECURE_BOOT = 3         # from secure_boot_model halt
    SYSTEM = 4              # generic/system-level


#: Rate limit budgets: max alerts per window, per severity class.
#: Lower severities are capped harder (to avoid noise flooding the
#: link); CRITICAL is intentionally generous since suppressing a real
#: critical alert is the worse failure mode.
DEFAULT_RATE_LIMITS: Dict[Severity, int] = {
    Severity.INFO: 5,
    Severity.LOW: 10,
    Severity.MEDIUM: 15,
    Severity.HIGH: 30,
    Severity.CRITICAL: 1000,
}

#: Rate limit window length, in ticks
DEFAULT_RATE_WINDOW_TICKS: int = 100


# ──────────────────────────────────────────────
# 2.  SEVERITY SCORING
# ──────────────────────────────────────────────

#: ThreatLevel (0-7, from threat_score.py) -> Severity (0-4) mapping
THREAT_LEVEL_TO_SEVERITY: Dict[int, Severity] = {
    0: Severity.INFO,      # NONE
    1: Severity.INFO,      # INFO
    2: Severity.LOW,       # LOW
    3: Severity.LOW,       # GUARDED
    4: Severity.MEDIUM,    # ELEVATED
    5: Severity.HIGH,      # HIGH
    6: Severity.HIGH,      # SEVERE
    7: Severity.CRITICAL,  # CRITICAL
}

#: TamperLevel (from tamper_detect.py, string-valued) -> Severity mapping
TAMPER_LEVEL_TO_SEVERITY: Dict[str, Severity] = {
    "benign": Severity.INFO,
    "warning": Severity.MEDIUM,
    "critical": Severity.CRITICAL,
}


def severity_from_threat_level(threat_level_value: int) -> Severity:
    """
    Map a threat_score.ThreatLevel integer value to unified Severity.

    Parameters
    ----------
    threat_level_value : int
        Integer value of a ThreatLevel enum member (0-7).

    Returns
    -------
    Severity
        Unified severity level.
    """
    return THREAT_LEVEL_TO_SEVERITY.get(threat_level_value, Severity.MEDIUM)


def severity_from_tamper_level(tamper_level_name: str) -> Severity:
    """
    Map a tamper_detect.TamperLevel name to unified Severity.

    Parameters
    ----------
    tamper_level_name : str
        String value of a TamperLevel enum member ("benign"/"warning"/"critical").

    Returns
    -------
    Severity
        Unified severity level.
    """
    return TAMPER_LEVEL_TO_SEVERITY.get(tamper_level_name, Severity.MEDIUM)


# ──────────────────────────────────────────────
# 3.  ALERT RECORD
# ──────────────────────────────────────────────

@dataclass
class Alert:
    """A single generated alert, with full audit metadata."""
    alert_id: int
    tick: int
    source: AlertSource
    severity: Severity
    description: str
    raw_score: Optional[float] = None       # e.g. threat_score_code or sensor reading
    zeroize_required: bool = False
    zeroize_acknowledged: bool = False
    zeroize_ack_tick: Optional[int] = None
    suppressed: bool = False                # True if logged-but-rate-limited


# ──────────────────────────────────────────────
# 4.  RATE LIMITER
# ──────────────────────────────────────────────

class RateLimiter:
    """
    Per-severity sliding-window rate limiter (simplified fixed-window
    model: counts reset every DEFAULT_RATE_WINDOW_TICKS ticks).
    """

    def __init__(
        self,
        limits: Dict[Severity, int] = None,
        window_ticks: int = DEFAULT_RATE_WINDOW_TICKS,
    ):
        """
        Initialize the rate limiter.

        Parameters
        ----------
        limits : Dict[Severity, int], optional
            Max alerts per window, per severity. Defaults to
            DEFAULT_RATE_LIMITS.
        window_ticks : int
            Window length in ticks before counters reset.
        """
        self.limits = dict(limits) if limits is not None else dict(DEFAULT_RATE_LIMITS)
        self.window_ticks = window_ticks
        self.window_start_tick: int = 0
        self.counts: Dict[Severity, int] = {s: 0 for s in Severity}
        self.suppressed_counts: Dict[Severity, int] = {s: 0 for s in Severity}

    def allow(self, severity: Severity, current_tick: int) -> bool:
        """
        Check (and account for) whether an alert of this severity is
        allowed under the current rate budget.

        Parameters
        ----------
        severity : Severity
            Severity of the candidate alert.
        current_tick : int
            Current simulation tick.

        Returns
        -------
        bool
            True if allowed (and counted), False if suppressed (and
            counted as suppressed).
        """
        # Roll window if expired
        if current_tick - self.window_start_tick >= self.window_ticks:
            self.window_start_tick = current_tick
            self.counts = {s: 0 for s in Severity}

        limit = self.limits.get(severity, 0)
        if self.counts[severity] < limit:
            self.counts[severity] += 1
            return True

        self.suppressed_counts[severity] += 1
        return False

    def stats(self) -> Dict:
        """Return rate limiter statistics."""
        return {
            "window_ticks": self.window_ticks,
            "current_counts": {s.name: c for s, c in self.counts.items()},
            "suppressed_counts": {s.name: c for s, c in self.suppressed_counts.items()},
        }


# ──────────────────────────────────────────────
# 5.  ALERT MANAGER (TOP-LEVEL)
# ──────────────────────────────────────────────

class AlertManager:
    """
    Top-level alert manager: severity scoring, timestamping, rate
    limiting, alert generation, and zeroize acknowledgement tracking.
    """

    def __init__(
        self,
        rate_limits: Dict[Severity, int] = None,
        rate_window_ticks: int = DEFAULT_RATE_WINDOW_TICKS,
        zeroize_trigger_severity: Severity = Severity.CRITICAL,
    ):
        """
        Initialize the alert manager.

        Parameters
        ----------
        rate_limits : Dict[Severity, int], optional
            Per-severity rate limit budgets.
        rate_window_ticks : int
            Rate limiter window length.
        zeroize_trigger_severity : Severity
            Minimum severity at which an alert is flagged as requiring
            a zeroization response (caller is responsible for actually
            triggering zeroize_fsm; this just tracks the requirement
            and its eventual acknowledgement).
        """
        self.rate_limiter = RateLimiter(rate_limits, rate_window_ticks)
        self.zeroize_trigger_severity = zeroize_trigger_severity

        self.tick: int = 0
        self._next_alert_id: int = 1
        self.alert_log: List[Alert] = []
        self._pending_zeroize_alert_ids: List[int] = []

    def _next_id(self) -> int:
        aid = self._next_alert_id
        self._next_alert_id += 1
        return aid

    def _emit(
        self,
        source: AlertSource,
        severity: Severity,
        description: str,
        raw_score: Optional[float] = None,
    ) -> Alert:
        """Internal: score, rate-limit, timestamp, and log one alert."""
        allowed = self.rate_limiter.allow(severity, self.tick)

        alert = Alert(
            alert_id=self._next_id(),
            tick=self.tick,
            source=source,
            severity=severity,
            description=description,
            raw_score=raw_score,
            zeroize_required=(severity >= self.zeroize_trigger_severity),
            suppressed=not allowed,
        )

        self.alert_log.append(alert)

        if alert.zeroize_required and not alert.suppressed:
            self._pending_zeroize_alert_ids.append(alert.alert_id)

        return alert

    def step(self, tick: Optional[int] = None) -> None:
        """
        Advance the manager's internal tick counter.

        Parameters
        ----------
        tick : int, optional
            Explicit tick value to set. If None, increments by 1.
        """
        self.tick = tick if tick is not None else self.tick + 1

    def process_threat_decision(self, decision) -> Alert:
        """
        Process a threat_score.ThreatDecision into an alert.

        Parameters
        ----------
        decision : ThreatDecision
            Decision object from threat_score.ThreatDetectionEngine.step().
            Duck-typed: expects .threat_level (int-valued enum),
            .pattern_id, .threat_score_code attributes.

        Returns
        -------
        Alert
            The generated (possibly suppressed) alert record.
        """
        severity = severity_from_threat_level(int(decision.threat_level))
        desc = (
            f"Threat detected: pattern={decision.pattern_id or 'unknown'}, "
            f"level={getattr(decision.threat_level, 'name', decision.threat_level)}"
        )
        return self._emit(
            AlertSource.THREAT_DETECTION, severity, desc,
            raw_score=float(decision.threat_score_code),
        )

    def process_tamper_event(self, event) -> Alert:
        """
        Process a tamper_detect.TamperEvent into an alert.

        Parameters
        ----------
        event : TamperEvent
            Event object from tamper_detect.TamperDetector.step().
            Duck-typed: expects .level (enum with .value), .channel,
            .reading, .reason attributes.

        Returns
        -------
        Alert
            The generated (possibly suppressed) alert record.
        """
        level_value = getattr(event.level, "value", str(event.level))
        severity = severity_from_tamper_level(level_value)
        channel_name = getattr(event.channel, "value", str(event.channel))
        desc = f"Tamper event: channel={channel_name}, reading={event.reading}, reason={event.reason}"
        return self._emit(
            AlertSource.TAMPER, severity, desc, raw_score=float(event.reading)
        )

    def process_boot_halt(self, halt_reason: str) -> Alert:
        """
        Process a secure_boot_model halt into a CRITICAL alert.

        Parameters
        ----------
        halt_reason : str
            Reason string from SecureBootChain.halt_reason.

        Returns
        -------
        Alert
            The generated alert record (always CRITICAL severity).
        """
        return self._emit(
            AlertSource.SECURE_BOOT, Severity.CRITICAL,
            f"Secure boot halted: {halt_reason}",
        )

    def raise_system_alert(self, description: str, severity: Severity = Severity.MEDIUM) -> Alert:
        """
        Raise a generic system-level alert.

        Parameters
        ----------
        description : str
            Human-readable alert description.
        severity : Severity
            Severity level for this alert.

        Returns
        -------
        Alert
            The generated (possibly suppressed) alert record.
        """
        return self._emit(AlertSource.SYSTEM, severity, description)

    def acknowledge_zeroize(self, alert_id: int) -> bool:
        """
        Acknowledge that a zeroization sequence triggered by a given
        alert has completed (Zeroize_Done_Ack).

        Parameters
        ----------
        alert_id : int
            ID of the alert whose required zeroization has completed.

        Returns
        -------
        bool
            True if the alert was found and pending, False if not
            found or not actually requiring zeroization.
        """
        for alert in self.alert_log:
            if alert.alert_id == alert_id:
                if not alert.zeroize_required:
                    return False
                alert.zeroize_acknowledged = True
                alert.zeroize_ack_tick = self.tick
                if alert_id in self._pending_zeroize_alert_ids:
                    self._pending_zeroize_alert_ids.remove(alert_id)
                return True
        return False

    def pending_zeroize_alerts(self) -> List[Alert]:
        """Return alerts that require zeroization but haven't been acknowledged yet."""
        return [a for a in self.alert_log if a.alert_id in self._pending_zeroize_alert_ids]

    def pending_alerts(self, since_tick: int = 0, include_suppressed: bool = False) -> List[Alert]:
        """
        Return alerts generated since a given tick, for downstream
        consumers (alert_formatter.py, log_writer.py).

        Parameters
        ----------
        since_tick : int
            Only return alerts with tick >= since_tick.
        include_suppressed : bool
            If False (default), rate-limited/suppressed alerts are
            excluded from the result (but remain in alert_log for audit).

        Returns
        -------
        List[Alert]
            Matching alerts, in chronological order.
        """
        return [
            a for a in self.alert_log
            if a.tick >= since_tick and (include_suppressed or not a.suppressed)
        ]

    def stats(self) -> Dict:
        """Return alert manager statistics."""
        by_severity: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        for a in self.alert_log:
            by_severity[a.severity.name] = by_severity.get(a.severity.name, 0) + 1
            by_source[a.source.name] = by_source.get(a.source.name, 0) + 1

        return {
            "tick": self.tick,
            "total_alerts": len(self.alert_log),
            "suppressed_alerts": sum(1 for a in self.alert_log if a.suppressed),
            "by_severity": by_severity,
            "by_source": by_source,
            "pending_zeroize": len(self._pending_zeroize_alert_ids),
            "rate_limiter": self.rate_limiter.stats(),
        }


# ──────────────────────────────────────────────
# 6.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("alert_manager.py  —  self-test")
    print("=" * 55)

    # Test 1: Severity mapping — threat levels
    print("\n[ Severity mapping — threat levels ]")
    assert severity_from_threat_level(0) == Severity.INFO    # NONE
    assert severity_from_threat_level(4) == Severity.MEDIUM  # ELEVATED
    assert severity_from_threat_level(7) == Severity.CRITICAL  # CRITICAL
    print(f"  ✓ ThreatLevel 0→INFO, 4→MEDIUM, 7→CRITICAL")

    # Test 2: Severity mapping — tamper levels
    print("\n[ Severity mapping — tamper levels ]")
    assert severity_from_tamper_level("benign") == Severity.INFO
    assert severity_from_tamper_level("warning") == Severity.MEDIUM
    assert severity_from_tamper_level("critical") == Severity.CRITICAL
    print(f"  ✓ benign→INFO, warning→MEDIUM, critical→CRITICAL")

    # Test 3: Basic alert generation
    print("\n[ Basic alert generation ]")
    mgr = AlertManager()
    alert = mgr.raise_system_alert("test system event", Severity.LOW)
    assert alert.alert_id == 1
    assert alert.severity == Severity.LOW
    assert not alert.suppressed
    print(f"  ✓ Generated alert #{alert.alert_id}: {alert.description}")

    # Test 4: Rate limiting — exceeding budget suppresses
    print("\n[ Rate limiting ]")
    mgr2 = AlertManager(rate_limits={Severity.LOW: 3}, rate_window_ticks=100)
    results = [mgr2.raise_system_alert(f"event_{i}", Severity.LOW).suppressed for i in range(5)]
    assert results == [False, False, False, True, True]
    print(f"  ✓ With limit=3: alerts 1-3 allowed, 4-5 suppressed: {results}")

    # Test 5: Rate limit window reset
    print("\n[ Rate limit window reset ]")
    mgr3 = AlertManager(rate_limits={Severity.LOW: 2}, rate_window_ticks=10)
    mgr3.raise_system_alert("a", Severity.LOW)
    mgr3.raise_system_alert("b", Severity.LOW)
    suppressed_before_reset = mgr3.raise_system_alert("c", Severity.LOW).suppressed
    assert suppressed_before_reset
    mgr3.step(tick=15)  # advance past window
    allowed_after_reset = not mgr3.raise_system_alert("d", Severity.LOW).suppressed
    assert allowed_after_reset
    print(f"  ✓ Rate limit window correctly resets after {mgr3.rate_limiter.window_ticks} ticks")

    # Test 6: CRITICAL alert flags zeroize_required
    print("\n[ Critical alert triggers zeroize requirement ]")
    mgr4 = AlertManager()
    crit_alert = mgr4.raise_system_alert("critical system fault", Severity.CRITICAL)
    assert crit_alert.zeroize_required
    assert not crit_alert.zeroize_acknowledged
    assert len(mgr4.pending_zeroize_alerts()) == 1
    print(f"  ✓ CRITICAL alert flagged zeroize_required, pending acknowledgement")

    # Test 7: Zeroize acknowledgement
    print("\n[ Zeroize_Done_Ack ]")
    acked = mgr4.acknowledge_zeroize(crit_alert.alert_id)
    assert acked
    assert crit_alert.zeroize_acknowledged
    assert crit_alert.zeroize_ack_tick == mgr4.tick
    assert len(mgr4.pending_zeroize_alerts()) == 0
    print(f"  ✓ Zeroize acknowledged at tick {crit_alert.zeroize_ack_tick}, "
          f"no longer pending")

    # Test 8: Acknowledge non-zeroize alert returns False
    print("\n[ Acknowledge non-critical alert rejected ]")
    low_alert = mgr4.raise_system_alert("minor event", Severity.LOW)
    ack_result = mgr4.acknowledge_zeroize(low_alert.alert_id)
    assert ack_result is False
    print(f"  ✓ Acknowledging a non-zeroize-required alert correctly returns False")

    # Test 9: Acknowledge unknown alert id
    print("\n[ Acknowledge unknown alert ID ]")
    assert mgr4.acknowledge_zeroize(99999) is False
    print(f"  ✓ Unknown alert ID correctly returns False")

    # Test 10: Integration — threat_score.ThreatDecision duck-typed input
    print("\n[ Integration — threat_score.ThreatDecision ]")
    from threat_score import ThreatDetectionEngine
    import hashlib

    engine = ThreatDetectionEngine()
    engine.register_pattern("dilithium_retry_storm", [1, 1, 0, 1], severity_weight=1.0)
    decision = engine.step([1, 1, 0, 1])  # exact match

    mgr5 = AlertManager()
    threat_alert = mgr5.process_threat_decision(decision)
    assert threat_alert.source == AlertSource.THREAT_DETECTION
    print(f"  ✓ Processed real ThreatDecision → severity={threat_alert.severity.name}, "
          f"desc={threat_alert.description}")

    # Test 11: Integration — tamper_detect.TamperEvent duck-typed input
    print("\n[ Integration — tamper_detect.TamperEvent ]")
    from tamper_detect import TamperDetector, SensorChannel

    detector = TamperDetector()
    events = detector.step({SensorChannel.VOLTAGE: 0.5})  # under-voltage, CRITICAL
    tamper_event = events[0]

    mgr6 = AlertManager()
    tamper_alert = mgr6.process_tamper_event(tamper_event)
    assert tamper_alert.severity == Severity.CRITICAL
    assert tamper_alert.zeroize_required
    print(f"  ✓ Processed real TamperEvent → severity={tamper_alert.severity.name}, "
          f"zeroize_required={tamper_alert.zeroize_required}")

    # Test 12: pending_alerts filtering
    print("\n[ pending_alerts() filtering ]")
    mgr7 = AlertManager(rate_limits={Severity.LOW: 1})
    mgr7.raise_system_alert("alert_a", Severity.LOW)       # allowed
    mgr7.raise_system_alert("alert_b", Severity.LOW)       # suppressed
    visible = mgr7.pending_alerts()
    all_including_suppressed = mgr7.pending_alerts(include_suppressed=True)
    assert len(visible) == 1
    assert len(all_including_suppressed) == 2
    print(f"  ✓ pending_alerts(): {len(visible)} visible, "
          f"{len(all_including_suppressed)} total (incl. suppressed)")

    # Test 13: Full multi-source scenario + statistics
    print("\n[ Full multi-source scenario ]")
    mgr8 = AlertManager()
    mgr8.process_threat_decision(decision)
    mgr8.process_tamper_event(tamper_event)
    mgr8.process_boot_halt("signature verification failed at stage pl_bitstream")
    mgr8.raise_system_alert("routine heartbeat", Severity.INFO)

    stats = mgr8.stats()
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  By source: {stats['by_source']}")
    print(f"  By severity: {stats['by_severity']}")
    print(f"  Pending zeroize: {stats['pending_zeroize']}")
    assert stats["total_alerts"] == 4
    assert stats["by_source"]["SECURE_BOOT"] == 1
    print(f"  ✓ Multi-source scenario tracked correctly")

    print("\n  All checks passed.\n")
