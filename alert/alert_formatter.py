"""
alert_formatter.py
====================
Alert message formatting for the PQC-SNN SoC (Alert & Output Subsystem).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Formats alert_manager.Alert records into the concrete output formats     ║
║ each downstream channel expects: UART, GPIO/LED, IRQ payload, network.   ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - UART message formatting (human-readable ASCII line, per diagram's
    "UART Message" output port)
  - GPIO/LED pattern encoding (severity → blink/solid pattern, per
    diagram's "GPIO/LED" output port)
  - Interrupt payload formatting (compact binary record for the PS
    interrupt controller, per diagram's "Interrupt to PS" output port)
  - Network packet formatting (structured binary frame for Ethernet
    transport, per diagram's "Network Packet" output port)
  - JSON formatting (for log_writer.py and general structured storage)
  - The architecture diagram's example message: "password detection
    happened" style human-readable summaries

Context (per architecture diagram, "OUTPUT INTERFACES" box, inside
"4. ALERT & OUTPUT SUBSYSTEM", and the "MESSAGE SENT TO SOFTWARE" box
at the bottom of the diagram which shows the literal example payload
"password detection happened" delivered via Interrupt / UART / Memory
Mapped Register):
  UART Message
  GPIO / LED
  Interrupt to PS
  Network Packet (Ethernet)
  Log to Memory  (handled by log_writer.py, the next file)

Design note on GPIO/LED patterns:
  Real hardware can't "blink" in a golden model — instead, each
  pattern is represented as a named ON/OFF duty sequence (a list of
  booleans) that an RTL testbench or LED driver can play back at a
  fixed tick rate. INFO get a slow single pulse; CRITICAL gets a fast
  continuous strobe, matching common alerting conventions.

Matches alert_formatter.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : AEGIS-NEURO architecture diagram, §4 ALERT & OUTPUT SUBSYSTEM
"""

from __future__ import annotations
import json
import struct
from dataclasses import dataclass
from typing import Dict, List

from alert_manager import Alert, Severity, AlertSource

# ──────────────────────────────────────────────
# 1.  GPIO / LED PATTERNS
# ──────────────────────────────────────────────

#: Named LED duty patterns per severity (True=on, False=off, one entry
#: per tick of a fixed playback rate defined by the caller/RTL testbench)
LED_PATTERNS: Dict[Severity, List[bool]] = {
    Severity.INFO:     [True, False, False, False, False, False, False, False],
    Severity.LOW:      [True, False, True, False, False, False, False, False],
    Severity.MEDIUM:   [True, False, True, False, True, False, False, False],
    Severity.HIGH:     [True, False, True, False, True, False, True, False],
    Severity.CRITICAL: [True, True, True, True, True, True, True, True],
}

#: GPIO pin assignment per severity (single-bit "this severity active" line)
GPIO_PIN_FOR_SEVERITY: Dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


# ──────────────────────────────────────────────
# 2.  UART MESSAGE FORMATTING
# ──────────────────────────────────────────────

def format_uart_message(alert: Alert) -> str:
    """
    Format an alert as a human-readable UART ASCII line.

    Parameters
    ----------
    alert : Alert
        Alert record from alert_manager.AlertManager.

    Returns
    -------
    str
        Single-line ASCII message, newline-terminated, suitable for
        direct transmission over a UART link.

    Examples
    --------
    "[T00042] CRITICAL  SECURE_BOOT  password detection happened\\n"
    """
    return (
        f"[T{alert.tick:05d}] "
        f"{alert.severity.name:<8} "
        f"{alert.source.name:<16} "
        f"{alert.description}\n"
    )


# ──────────────────────────────────────────────
# 3.  GPIO / LED FORMATTING
# ──────────────────────────────────────────────

@dataclass
class GpioLedOutput:
    """GPIO/LED output for one alert."""
    pin: int
    pattern: List[bool]
    severity: Severity


def format_gpio_led(alert: Alert) -> GpioLedOutput:
    """
    Format an alert as a GPIO pin assignment + LED blink pattern.

    Parameters
    ----------
    alert : Alert
        Alert record.

    Returns
    -------
    GpioLedOutput
        Pin number and on/off duty pattern for this alert's severity.
    """
    return GpioLedOutput(
        pin=GPIO_PIN_FOR_SEVERITY[alert.severity],
        pattern=list(LED_PATTERNS[alert.severity]),
        severity=alert.severity,
    )


# ──────────────────────────────────────────────
# 4.  INTERRUPT PAYLOAD FORMATTING
# ──────────────────────────────────────────────

#: Interrupt payload layout (big-endian, fixed 16 bytes):
#:   alert_id   : 4 bytes (uint32)
#:   tick        : 4 bytes (uint32)
#:   severity    : 1 byte  (uint8)
#:   source      : 1 byte  (uint8)
#:   zeroize_req : 1 byte  (0/1)
#:   reserved    : 1 byte  (0x00, padding)
#:   raw_score   : 4 bytes (float32, 0.0 if None)
IRQ_PAYLOAD_STRUCT = struct.Struct(">IIBBBBf")
IRQ_PAYLOAD_LEN: int = IRQ_PAYLOAD_STRUCT.size  # 16 bytes


def format_irq_payload(alert: Alert) -> bytes:
    """
    Format an alert as a compact binary interrupt payload for the PS
    interrupt controller (per diagram's "Interrupt to PS" port).

    Parameters
    ----------
    alert : Alert
        Alert record.

    Returns
    -------
    bytes
        Fixed-size (16-byte) binary payload.
    """
    return IRQ_PAYLOAD_STRUCT.pack(
        alert.alert_id & 0xFFFFFFFF,
        alert.tick & 0xFFFFFFFF,
        int(alert.severity) & 0xFF,
        int(alert.source) & 0xFF,
        1 if alert.zeroize_required else 0,
        0,  # reserved/padding
        float(alert.raw_score) if alert.raw_score is not None else 0.0,
    )


def parse_irq_payload(data: bytes) -> Dict:
    """
    Parse a binary interrupt payload back into its fields (for
    testbench verification / loopback checking).

    Parameters
    ----------
    data : bytes
        16-byte payload as produced by format_irq_payload().

    Returns
    -------
    Dict
        Parsed fields: alert_id, tick, severity, source, zeroize_required,
        raw_score.
    """
    alert_id, tick, severity, source, zeroize_req, _reserved, raw_score = (
        IRQ_PAYLOAD_STRUCT.unpack(data)
    )
    return {
        "alert_id": alert_id,
        "tick": tick,
        "severity": Severity(severity),
        "source": AlertSource(source),
        "zeroize_required": bool(zeroize_req),
        "raw_score": raw_score,
    }


# ──────────────────────────────────────────────
# 5.  NETWORK PACKET FORMATTING
# ──────────────────────────────────────────────

#: Network frame magic bytes (identifies AEGIS-NEURO alert frames on the wire)
NETWORK_FRAME_MAGIC: bytes = b"\xAE\x91"

#: Network frame layout (big-endian):
#:   magic       : 2 bytes
#:   version     : 1 byte (protocol version, currently 1)
#:   alert_id    : 4 bytes
#:   tick        : 4 bytes
#:   severity    : 1 byte
#:   source      : 1 byte
#:   zeroize_req : 1 byte
#:   desc_len    : 2 bytes (length of UTF-8 description that follows)
#:   description : variable length UTF-8 bytes
NETWORK_FRAME_VERSION: int = 1
_NETWORK_HEADER_STRUCT = struct.Struct(">2sBIIBBBH")


def format_network_packet(alert: Alert) -> bytes:
    """
    Format an alert as a structured binary network frame for Ethernet
    transport (per diagram's "Network Packet" output port).

    Parameters
    ----------
    alert : Alert
        Alert record.

    Returns
    -------
    bytes
        Variable-length frame: fixed header + UTF-8 description.
    """
    desc_bytes = alert.description.encode("utf-8")
    header = _NETWORK_HEADER_STRUCT.pack(
        NETWORK_FRAME_MAGIC,
        NETWORK_FRAME_VERSION,
        alert.alert_id & 0xFFFFFFFF,
        alert.tick & 0xFFFFFFFF,
        int(alert.severity) & 0xFF,
        int(alert.source) & 0xFF,
        1 if alert.zeroize_required else 0,
        len(desc_bytes) & 0xFFFF,
    )
    return header + desc_bytes


def parse_network_packet(data: bytes) -> Dict:
    """
    Parse a network frame back into its fields.

    Parameters
    ----------
    data : bytes
        Frame as produced by format_network_packet().

    Returns
    -------
    Dict
        Parsed fields, or raises ValueError if magic bytes don't match.
    """
    header_len = _NETWORK_HEADER_STRUCT.size
    magic, version, alert_id, tick, severity, source, zeroize_req, desc_len = (
        _NETWORK_HEADER_STRUCT.unpack(data[:header_len])
    )
    if magic != NETWORK_FRAME_MAGIC:
        raise ValueError(f"Invalid frame magic bytes: {magic!r}")

    description = data[header_len:header_len + desc_len].decode("utf-8")

    return {
        "version": version,
        "alert_id": alert_id,
        "tick": tick,
        "severity": Severity(severity),
        "source": AlertSource(source),
        "zeroize_required": bool(zeroize_req),
        "description": description,
    }


# ──────────────────────────────────────────────
# 6.  JSON FORMATTING (for log_writer.py / structured storage)
# ──────────────────────────────────────────────

def format_json(alert: Alert) -> str:
    """
    Format an alert as a JSON string, for log_writer.py and general
    structured storage/inspection.

    Parameters
    ----------
    alert : Alert
        Alert record.

    Returns
    -------
    str
        JSON-encoded alert record.
    """
    return json.dumps({
        "alert_id": alert.alert_id,
        "tick": alert.tick,
        "source": alert.source.name,
        "severity": alert.severity.name,
        "description": alert.description,
        "raw_score": alert.raw_score,
        "zeroize_required": alert.zeroize_required,
        "zeroize_acknowledged": alert.zeroize_acknowledged,
        "zeroize_ack_tick": alert.zeroize_ack_tick,
        "suppressed": alert.suppressed,
    })


# ──────────────────────────────────────────────
# 7.  MULTI-FORMAT DISPATCH HELPER
# ──────────────────────────────────────────────

def format_all(alert: Alert) -> Dict[str, object]:
    """
    Format an alert into every supported output format at once,
    convenient for a dispatcher that fans out to all live channels.

    Parameters
    ----------
    alert : Alert
        Alert record.

    Returns
    -------
    Dict[str, object]
        Mapping of channel name -> formatted payload:
        "uart" -> str, "gpio_led" -> GpioLedOutput, "irq" -> bytes,
        "network" -> bytes, "json" -> str.
    """
    return {
        "uart": format_uart_message(alert),
        "gpio_led": format_gpio_led(alert),
        "irq": format_irq_payload(alert),
        "network": format_network_packet(alert),
        "json": format_json(alert),
    }


# ──────────────────────────────────────────────
# 8.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("alert_formatter.py  —  self-test")
    print("=" * 55)

    from alert_manager import AlertManager

    # Build a real alert via the real AlertManager (not a hand-built mock)
    mgr = AlertManager()
    mgr.step(tick=42)
    alert = mgr.process_boot_halt("password detection happened")

    # Test 1: UART formatting
    print("\n[ UART message formatting ]")
    uart_msg = format_uart_message(alert)
    assert "CRITICAL" in uart_msg
    assert "password detection happened" in uart_msg
    assert uart_msg.endswith("\n")
    print(f"  ✓ UART line: {uart_msg.strip()!r}")

    # Test 2: GPIO/LED formatting
    print("\n[ GPIO/LED formatting ]")
    gpio = format_gpio_led(alert)
    assert gpio.pin == GPIO_PIN_FOR_SEVERITY[Severity.CRITICAL]
    assert gpio.pattern == LED_PATTERNS[Severity.CRITICAL]
    assert all(gpio.pattern)  # CRITICAL = solid strobe (all True)
    print(f"  ✓ GPIO pin={gpio.pin}, pattern={gpio.pattern}")

    # Test 3: Different severities produce different LED patterns
    print("\n[ LED pattern differentiation by severity ]")
    info_alert = mgr.raise_system_alert("routine check", Severity.INFO)
    info_gpio = format_gpio_led(info_alert)
    assert info_gpio.pattern != gpio.pattern
    assert sum(info_gpio.pattern) < sum(gpio.pattern)  # fewer on-ticks than CRITICAL
    print(f"  ✓ INFO pattern ({sum(info_gpio.pattern)} on-ticks) is sparser than "
          f"CRITICAL ({sum(gpio.pattern)} on-ticks)")

    # Test 4: IRQ payload — fixed size and roundtrip
    print("\n[ IRQ payload format/parse roundtrip ]")
    irq_bytes = format_irq_payload(alert)
    assert len(irq_bytes) == IRQ_PAYLOAD_LEN == 16
    parsed = parse_irq_payload(irq_bytes)
    assert parsed["alert_id"] == alert.alert_id
    assert parsed["tick"] == alert.tick
    assert parsed["severity"] == alert.severity
    assert parsed["source"] == alert.source
    assert parsed["zeroize_required"] == alert.zeroize_required
    print(f"  ✓ IRQ payload: {len(irq_bytes)} bytes, roundtrip fields match exactly")

    # Test 5: Network packet — magic bytes and roundtrip
    print("\n[ Network packet format/parse roundtrip ]")
    net_bytes = format_network_packet(alert)
    assert net_bytes[:2] == NETWORK_FRAME_MAGIC
    parsed_net = parse_network_packet(net_bytes)
    assert parsed_net["alert_id"] == alert.alert_id
    assert parsed_net["description"] == alert.description
    assert parsed_net["severity"] == alert.severity
    print(f"  ✓ Network frame: {len(net_bytes)} bytes "
          f"(header + {len(alert.description.encode('utf-8'))}-byte description)")

    # Test 6: Network packet rejects bad magic
    print("\n[ Network packet — invalid magic rejected ]")
    corrupted = b"\x00\x00" + net_bytes[2:]
    try:
        parse_network_packet(corrupted)
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ Corrupted magic bytes correctly rejected")

    # Test 7: JSON formatting — valid JSON, round-trippable
    print("\n[ JSON formatting ]")
    json_str = format_json(alert)
    parsed_json = json.loads(json_str)
    assert parsed_json["alert_id"] == alert.alert_id
    assert parsed_json["severity"] == "CRITICAL"
    assert parsed_json["description"] == "Secure boot halted: password detection happened"
    print(f"  ✓ JSON: {json_str}")

    # Test 8: Unicode description in network packet (UTF-8 safety)
    print("\n[ Unicode description handling ]")
    mgr.step(tick=43)
    unicode_alert = mgr.raise_system_alert("château firmware check — résumé", Severity.LOW)
    net_unicode = format_network_packet(unicode_alert)
    parsed_unicode = parse_network_packet(net_unicode)
    assert parsed_unicode["description"] == "château firmware check — résumé"
    print(f"  ✓ Unicode description survives format/parse roundtrip")

    # Test 9: format_all dispatch helper
    print("\n[ format_all() multi-channel dispatch ]")
    all_formats = format_all(alert)
    assert set(all_formats.keys()) == {"uart", "gpio_led", "irq", "network", "json"}
    assert isinstance(all_formats["uart"], str)
    assert isinstance(all_formats["irq"], bytes)
    print(f"  ✓ format_all() produced all 5 channel formats: {list(all_formats.keys())}")

    # Test 10: Integration — full pipeline from real ThreatDecision through formatting
    print("\n[ Full pipeline integration ]")
    from threat_score import ThreatDetectionEngine

    engine = ThreatDetectionEngine()
    engine.register_pattern("attack_sig", [1, 1, 0, 1], severity_weight=1.0)
    decision = engine.step([1, 1, 0, 1])

    mgr2 = AlertManager()
    threat_alert = mgr2.process_threat_decision(decision)
    formatted = format_all(threat_alert)
    assert "Threat detected" in formatted["uart"]
    print(f"  ✓ ThreatDecision → Alert → all formats: "
          f"UART={formatted['uart'].strip()!r}")

    print("\n  All checks passed.\n")
