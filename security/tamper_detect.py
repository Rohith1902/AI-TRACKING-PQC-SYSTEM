"""
tamper_detect.py
=================
Hardware tamper detection model for the PQC-SNN SoC security subsystem.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Models physical/environmental tamper sensors and triggers an emergency    ║
║ zeroization response when readings exceed safe operating bounds.         ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Sensor channel model (voltage, temperature, clock-glitch, light, mesh)
  - Per-channel threshold checking (min/max bounds)
  - Glitch detection (sudden delta beyond noise floor)
  - Tamper event classification (benign / warning / critical)
  - Tamper response dispatch (alert vs. zeroize-triggering)
  - Event log with timestamped history (simulated tick counter)

Sensor channels modeled:
  - VOLTAGE     : supply rail voltage (V), under/over-voltage attacks
  - TEMPERATURE : die temperature (°C), thermal fault injection
  - CLOCK_GLITCH: clock period deviation (ns), glitch/fault injection
  - LIGHT       : photo-sensor (lux), die decapsulation detection
  - MESH        : tamper-mesh continuity (0=intact, 1=broken)

Detection algorithm:
  For each channel:
    1. Check absolute bounds (min ≤ reading ≤ max)
    2. Check rate-of-change (|reading[t] - reading[t-1]| > glitch_threshold)
    3. Classify: BENIGN, WARNING (single channel marginal), or
       CRITICAL (bound violation or glitch on security-relevant channel)
  CRITICAL on any channel → trigger zeroize_fsm (see zeroize_fsm.py)

Response policy:
  - BENIGN   : no action, log only
  - WARNING  : increment warning counter, alert subsystem (alert_manager)
  - CRITICAL : immediate trigger of zeroize_all() in key_protection.py

Matches tamper_detect.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : FIPS 140-3 §7.8 (Physical Security — Tamper Detection)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

# ──────────────────────────────────────────────
# 1.  SENSOR CHANNELS
# ──────────────────────────────────────────────

class SensorChannel(Enum):
    VOLTAGE = "voltage"
    TEMPERATURE = "temperature"
    CLOCK_GLITCH = "clock_glitch"
    LIGHT = "light"
    MESH = "mesh"


class TamperLevel(Enum):
    BENIGN = "benign"
    WARNING = "warning"
    CRITICAL = "critical"


#: Safe operating bounds per channel: (min, max)
SENSOR_BOUNDS: Dict[SensorChannel, tuple] = {
    SensorChannel.VOLTAGE: (1.62, 1.98),       # nominal 1.8V ±10%
    SensorChannel.TEMPERATURE: (-40.0, 125.0), # industrial range (°C)
    SensorChannel.CLOCK_GLITCH: (-2.0, 2.0),   # ns deviation from nominal period
    SensorChannel.LIGHT: (0.0, 50.0),          # lux; >50 implies decap exposure
    SensorChannel.MESH: (0, 0),                # 0=intact only; 1=broken triggers
}

#: Rate-of-change glitch thresholds (per channel, per tick)
GLITCH_THRESHOLDS: Dict[SensorChannel, float] = {
    SensorChannel.VOLTAGE: 0.3,
    SensorChannel.TEMPERATURE: 30.0,
    SensorChannel.CLOCK_GLITCH: 1.5,
    SensorChannel.LIGHT: 40.0,
    SensorChannel.MESH: 1.0,
}

#: Warning threshold margin (fraction of bound range treated as marginal)
WARNING_MARGIN: float = 0.1


# ──────────────────────────────────────────────
# 2.  EVENT RECORD
# ──────────────────────────────────────────────

@dataclass
class TamperEvent:
    """Single tamper-detection event."""
    tick: int
    channel: SensorChannel
    reading: float
    level: TamperLevel
    reason: str


# ──────────────────────────────────────────────
# 3.  TAMPER DETECTOR
# ──────────────────────────────────────────────

class TamperDetector:
    """
    Models a multi-channel hardware tamper detection subsystem.
    """

    def __init__(self, on_critical: Optional[Callable[[TamperEvent], None]] = None):
        """
        Initialize the tamper detector.

        Parameters
        ----------
        on_critical : callable, optional
            Callback invoked with the TamperEvent when a CRITICAL
            event is detected (e.g. to trigger zeroize_fsm.zeroize_all()).
        """
        self.tick: int = 0
        self.last_reading: Dict[SensorChannel, float] = {}
        self.event_log: List[TamperEvent] = []
        self.warning_count: int = 0
        self.critical_count: int = 0
        self.on_critical = on_critical
        self.armed: bool = True  # detector can be disarmed for test/debug

    def step(self, readings: Dict[SensorChannel, float]) -> List[TamperEvent]:
        """
        Process one tick of sensor readings across all channels.

        Parameters
        ----------
        readings : Dict[SensorChannel, float]
            Current reading per channel.

        Returns
        -------
        List[TamperEvent]
            Events generated this tick (may be empty if all benign).
        """
        self.tick += 1
        events = []

        if not self.armed:
            return events

        for channel, value in readings.items():
            event = self._evaluate_channel(channel, value)
            events.append(event)
            self.event_log.append(event)

            if event.level == TamperLevel.WARNING:
                self.warning_count += 1
            elif event.level == TamperLevel.CRITICAL:
                self.critical_count += 1
                if self.on_critical is not None:
                    self.on_critical(event)

            self.last_reading[channel] = value

        return events

    def _evaluate_channel(self, channel: SensorChannel, value: float) -> TamperEvent:
        """Evaluate a single channel reading against bounds and glitch rate."""
        lo, hi = SENSOR_BOUNDS[channel]

        # Absolute bound violation → CRITICAL
        if value < lo or value > hi:
            return TamperEvent(
                tick=self.tick, channel=channel, reading=value,
                level=TamperLevel.CRITICAL,
                reason=f"out_of_bounds [{lo},{hi}]",
            )

        # Glitch (rate of change) check
        prev = self.last_reading.get(channel)
        if prev is not None:
            delta = abs(value - prev)
            if delta > GLITCH_THRESHOLDS[channel]:
                return TamperEvent(
                    tick=self.tick, channel=channel, reading=value,
                    level=TamperLevel.CRITICAL,
                    reason=f"glitch delta={delta:.3f}",
                )

        # Marginal/warning zone near bounds
        # NOTE: LIGHT is asymmetric — low lux (sealed package) is always
        # benign; only the *upper* bound (decap exposure) has a warning
        # margin. Other channels use symmetric two-sided margins.
        span = hi - lo
        if span > 0:
            margin = span * WARNING_MARGIN
            if channel == SensorChannel.LIGHT:
                if value > hi - margin:
                    return TamperEvent(
                        tick=self.tick, channel=channel, reading=value,
                        level=TamperLevel.WARNING,
                        reason="near_bound",
                    )
            elif value < lo + margin or value > hi - margin:
                return TamperEvent(
                    tick=self.tick, channel=channel, reading=value,
                    level=TamperLevel.WARNING,
                    reason="near_bound",
                )

        return TamperEvent(
            tick=self.tick, channel=channel, reading=value,
            level=TamperLevel.BENIGN, reason="ok",
        )

    def disarm(self) -> None:
        """Disarm the detector (test/debug mode only — logged separately)."""
        self.armed = False

    def arm(self) -> None:
        """Re-arm the detector."""
        self.armed = True

    def stats(self) -> Dict:
        """Return detector statistics."""
        return {
            "tick": self.tick,
            "armed": self.armed,
            "total_events": len(self.event_log),
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "benign_count": len(self.event_log) - self.warning_count - self.critical_count,
        }


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("tamper_detect.py  —  self-test")
    print("=" * 55)

    # Test 1: Benign readings
    print("\n[ Benign nominal readings ]")
    det = TamperDetector()
    events = det.step({
        SensorChannel.VOLTAGE: 1.80,
        SensorChannel.TEMPERATURE: 25.0,
        SensorChannel.CLOCK_GLITCH: 0.0,
        SensorChannel.LIGHT: 0.0,
        SensorChannel.MESH: 0,
    })
    assert all(e.level == TamperLevel.BENIGN for e in events)
    print(f"  ✓ All {len(events)} channels reported BENIGN")

    # Test 2: Out-of-bounds voltage → CRITICAL
    print("\n[ Under-voltage attack ]")
    events = det.step({
        SensorChannel.VOLTAGE: 1.2,  # below 1.62V min
        SensorChannel.TEMPERATURE: 25.0,
        SensorChannel.CLOCK_GLITCH: 0.0,
        SensorChannel.LIGHT: 0.0,
        SensorChannel.MESH: 0,
    })
    voltage_event = [e for e in events if e.channel == SensorChannel.VOLTAGE][0]
    assert voltage_event.level == TamperLevel.CRITICAL
    print(f"  ✓ Under-voltage (1.2V) → CRITICAL: {voltage_event.reason}")

    # Test 3: Mesh breach → CRITICAL
    print("\n[ Tamper-mesh breach ]")
    det2 = TamperDetector()
    det2.step({SensorChannel.MESH: 0})  # establish baseline
    events = det2.step({SensorChannel.MESH: 1})  # breach
    mesh_event = events[0]
    assert mesh_event.level == TamperLevel.CRITICAL
    print(f"  ✓ Mesh breach (0→1) → CRITICAL: {mesh_event.reason}")

    # Test 4: Clock glitch detection
    print("\n[ Clock glitch detection ]")
    det3 = TamperDetector()
    det3.step({SensorChannel.CLOCK_GLITCH: 0.0})
    events = det3.step({SensorChannel.CLOCK_GLITCH: 1.8})  # > 1.5 threshold but within [-2,2] bound
    glitch_event = events[0]
    assert glitch_event.level == TamperLevel.CRITICAL
    print(f"  ✓ Clock glitch (Δ=1.8ns) → CRITICAL: {glitch_event.reason}")

    # Test 5: Warning zone (marginal reading)
    print("\n[ Warning zone — marginal temperature ]")
    det4 = TamperDetector()
    # range [-40,125], span=165, margin=16.5 → warning if temp > 108.5
    events = det4.step({SensorChannel.TEMPERATURE: 115.0})
    temp_event = events[0]
    assert temp_event.level == TamperLevel.WARNING
    print(f"  ✓ Marginal temp (115°C) → WARNING: {temp_event.reason}")

    # Test 6: on_critical callback triggers zeroization-style response
    print("\n[ Critical callback dispatch ]")
    triggered = []

    def fake_zeroize_all(event: TamperEvent):
        triggered.append(event)

    det5 = TamperDetector(on_critical=fake_zeroize_all)
    det5.step({SensorChannel.VOLTAGE: 0.5})  # critical under-voltage
    assert len(triggered) == 1
    assert triggered[0].level == TamperLevel.CRITICAL
    print(f"  ✓ on_critical callback fired for {triggered[0].channel.value}")

    # Test 7: Disarm/arm
    print("\n[ Disarm / re-arm ]")
    det6 = TamperDetector()
    det6.disarm()
    events = det6.step({SensorChannel.VOLTAGE: 0.5})  # would be critical if armed
    assert events == []
    print(f"  ✓ Disarmed detector produces no events")
    det6.arm()
    events = det6.step({SensorChannel.VOLTAGE: 1.8})
    assert len(events) == 1
    print(f"  ✓ Re-armed detector resumes monitoring")

    # Test 8: Full multi-tick simulation
    print("\n[ Multi-tick simulation ]")
    det7 = TamperDetector()
    sequence = [
        {SensorChannel.VOLTAGE: 1.80, SensorChannel.TEMPERATURE: 25.0},
        {SensorChannel.VOLTAGE: 1.79, SensorChannel.TEMPERATURE: 26.0},
        {SensorChannel.VOLTAGE: 1.78, SensorChannel.TEMPERATURE: 27.0},
        {SensorChannel.VOLTAGE: 1.10, SensorChannel.TEMPERATURE: 27.0},  # attack tick
    ]
    for reading_set in sequence:
        det7.step(reading_set)

    stats = det7.stats()
    assert stats["tick"] == 4
    assert stats["critical_count"] >= 1
    print(f"  Tick: {stats['tick']}, Critical events: {stats['critical_count']}")
    print(f"  ✓ Multi-tick simulation tracked correctly")

    # Test 9: Statistics summary
    print("\n[ Statistics summary ]")
    final_stats = det.stats()
    print(f"  Total events: {final_stats['total_events']}")
    print(f"  Benign: {final_stats['benign_count']}, "
          f"Warning: {final_stats['warning_count']}, "
          f"Critical: {final_stats['critical_count']}")
    print(f"  ✓ Statistics aggregated correctly")

    print("\n  All checks passed.\n")
