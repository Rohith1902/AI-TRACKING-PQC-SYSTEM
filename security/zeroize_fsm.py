"""
zeroize_fsm.py
===============
Secure zeroization finite state machine for the PQC-SNN SoC security subsystem.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Coordinates emergency and routine key/state zeroization via a verified    ║
║ FSM, ensuring all sensitive memory is overwritten before completion.     ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Zeroization FSM (IDLE → TRIGGERED → WIPING → VERIFYING → DONE)
  - Multi-source trigger handling (tamper, command, watchdog timeout)
  - Memory region registry (what gets wiped, in what order)
  - Multi-pass overwrite (defense against memory remanence)
  - Post-wipe verification (read-back check)
  - Completion/failure reporting

FSM states:
  IDLE       — normal operation, no zeroization pending
  TRIGGERED  — trigger received, about to begin wipe sequence
  WIPING     — actively overwriting registered memory regions
  VERIFYING  — reading back regions to confirm all-zero
  DONE       — zeroization complete and verified
  FAILED     — verification failed (region not fully wiped)

Trigger sources:
  - TAMPER_CRITICAL : from tamper_detect.py on_critical callback
  - HOST_COMMAND    : explicit software-issued zeroize command
  - WATCHDOG_TIMEOUT: system watchdog expired without heartbeat
  - SELF_TEST_FAIL  : FIPS self-test failure (e.g. KAT mismatch)

Wipe sequence (per FIPS 140-3 zeroization requirements):
  1. For each registered region (in priority order):
       a. Overwrite with pass 1 pattern (0x00)
       b. Overwrite with pass 2 pattern (0xFF)
       c. Overwrite with pass 3 pattern (0x00)   — defense-in-depth
  2. Read back each region, confirm all bytes == 0x00
  3. Transition to DONE if all verified, else FAILED

Matches zeroize_fsm.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : FIPS 140-3 §7.9 (Zeroization Requirements)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

# ──────────────────────────────────────────────
# 1.  FSM STATES AND TRIGGERS
# ──────────────────────────────────────────────

class ZeroizeState(Enum):
    IDLE = "idle"
    TRIGGERED = "triggered"
    WIPING = "wiping"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"


class TriggerSource(Enum):
    TAMPER_CRITICAL = "tamper_critical"
    HOST_COMMAND = "host_command"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    SELF_TEST_FAIL = "self_test_fail"


#: Valid FSM transitions
VALID_TRANSITIONS: Dict[ZeroizeState, set] = {
    ZeroizeState.IDLE: {ZeroizeState.TRIGGERED},
    ZeroizeState.TRIGGERED: {ZeroizeState.WIPING},
    ZeroizeState.WIPING: {ZeroizeState.VERIFYING},
    ZeroizeState.VERIFYING: {ZeroizeState.DONE, ZeroizeState.FAILED},
    ZeroizeState.DONE: set(),    # terminal
    ZeroizeState.FAILED: set(),  # terminal (requires hard reset to recover)
}

#: Overwrite pass patterns (defense-in-depth multi-pass wipe)
WIPE_PATTERNS: List[int] = [0x00, 0xFF, 0x00]


# ──────────────────────────────────────────────
# 2.  MEMORY REGION REGISTRY
# ──────────────────────────────────────────────

@dataclass
class MemoryRegion:
    """A registered region of sensitive memory."""
    name: str
    data: bytearray
    priority: int = 0  # lower number = wiped first


# ──────────────────────────────────────────────
# 3.  ZEROIZE FSM
# ──────────────────────────────────────────────

class ZeroizeFSM:
    """
    Finite state machine coordinating secure zeroization of all
    registered sensitive memory regions.
    """

    def __init__(self):
        self.state: ZeroizeState = ZeroizeState.IDLE
        self.regions: Dict[str, MemoryRegion] = {}
        self.trigger_source: Optional[TriggerSource] = None
        self.wipe_log: List[Dict] = []
        self.transition_log: List[Dict] = []
        self.on_complete: Optional[Callable[[], None]] = None

    # ── Region registry ──────────────────────────

    def register_region(
        self, name: str, data: bytearray, priority: int = 0
    ) -> None:
        """
        Register a memory region to be wiped on zeroization.

        Parameters
        ----------
        name : str
            Region identifier (e.g. "kyber_sk", "dilithium_sk", "pepper").
        data : bytearray
            Mutable buffer representing the sensitive memory.
            Must be a bytearray (not bytes) so it can be overwritten in place.
        priority : int
            Wipe order priority; lower values wiped first.
        """
        if not isinstance(data, bytearray):
            raise TypeError("data must be a bytearray for in-place zeroization")

        self.regions[name] = MemoryRegion(name=name, data=data, priority=priority)

    def unregister_region(self, name: str) -> None:
        """Remove a region from the registry (e.g. after key rotation)."""
        self.regions.pop(name, None)

    # ── Trigger and execution ────────────────────

    def trigger(self, source: TriggerSource) -> None:
        """
        Trigger zeroization from IDLE state.

        Parameters
        ----------
        source : TriggerSource
            What caused this zeroization request.
        """
        if self.state != ZeroizeState.IDLE:
            # Already in progress or done — ignore re-trigger (idempotent)
            return

        self.trigger_source = source
        self._transition(ZeroizeState.TRIGGERED)

    def run(self) -> ZeroizeState:
        """
        Execute the full zeroization sequence synchronously
        (TRIGGERED → WIPING → VERIFYING → DONE/FAILED).

        Returns
        -------
        ZeroizeState
            Final state after execution (DONE or FAILED).
        """
        if self.state != ZeroizeState.TRIGGERED:
            raise RuntimeError(
                f"run() requires state TRIGGERED, got {self.state.value}"
            )

        self._transition(ZeroizeState.WIPING)
        self._wipe_all_regions()

        self._transition(ZeroizeState.VERIFYING)
        all_verified = self._verify_all_regions()

        if all_verified:
            self._transition(ZeroizeState.DONE)
            if self.on_complete is not None:
                self.on_complete()
        else:
            self._transition(ZeroizeState.FAILED)

        return self.state

    def _wipe_all_regions(self) -> None:
        """Overwrite all registered regions in priority order, multi-pass."""
        ordered = sorted(self.regions.values(), key=lambda r: r.priority)

        for region in ordered:
            for pass_num, pattern in enumerate(WIPE_PATTERNS, start=1):
                for i in range(len(region.data)):
                    region.data[i] = pattern

                self.wipe_log.append({
                    "region": region.name,
                    "pass": pass_num,
                    "pattern": hex(pattern),
                    "size": len(region.data),
                })

    def _verify_all_regions(self) -> bool:
        """Read back all regions, confirm they are all-zero (final pattern)."""
        all_ok = True
        for region in self.regions.values():
            verified = all(b == 0x00 for b in region.data)
            self.wipe_log.append({
                "region": region.name,
                "pass": "verify",
                "result": "ok" if verified else "FAIL",
            })
            if not verified:
                all_ok = False
        return all_ok

    def reset(self) -> None:
        """
        Reset FSM to IDLE after DONE (e.g. for next test cycle).

        Note: in real hardware, FAILED state typically requires a
        hard power-on reset and cannot be cleared by this method.
        """
        if self.state == ZeroizeState.DONE:
            self.state = ZeroizeState.IDLE
            self.trigger_source = None
        else:
            raise RuntimeError(
                f"Cannot reset from state {self.state.value}; "
                f"only DONE can transition back to IDLE"
            )

    # ── Internal helpers ─────────────────────────

    def _transition(self, new_state: ZeroizeState) -> None:
        if new_state not in VALID_TRANSITIONS[self.state]:
            raise ValueError(
                f"Invalid transition: {self.state.value} → {new_state.value}"
            )
        self.transition_log.append({
            "from": self.state.value, "to": new_state.value
        })
        self.state = new_state

    def stats(self) -> Dict:
        """Return FSM statistics."""
        return {
            "state": self.state.value,
            "trigger_source": self.trigger_source.value if self.trigger_source else None,
            "regions_registered": len(self.regions),
            "wipe_operations": len(self.wipe_log),
            "transitions": len(self.transition_log),
        }


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("zeroize_fsm.py  —  self-test")
    print("=" * 55)

    # Test 1: Initial state
    print("\n[ Initial state ]")
    fsm = ZeroizeFSM()
    assert fsm.state == ZeroizeState.IDLE
    print(f"  ✓ FSM starts in IDLE")

    # Test 2: Register regions
    print("\n[ Region registration ]")
    kyber_sk = bytearray(b"\xAB" * 32)
    dilithium_sk = bytearray(b"\xCD" * 64)
    pepper = bytearray(b"\xEF" * 32)

    fsm.register_region("kyber_sk", kyber_sk, priority=0)
    fsm.register_region("dilithium_sk", dilithium_sk, priority=1)
    fsm.register_region("pepper", pepper, priority=2)
    assert len(fsm.regions) == 3
    print(f"  ✓ Registered 3 memory regions")

    # Test 3: Trigger from tamper event
    print("\n[ Trigger (tamper critical) ]")
    fsm.trigger(TriggerSource.TAMPER_CRITICAL)
    assert fsm.state == ZeroizeState.TRIGGERED
    print(f"  ✓ State: IDLE → TRIGGERED")

    # Test 4: Run full sequence
    print("\n[ Execute zeroization sequence ]")
    final_state = fsm.run()
    assert final_state == ZeroizeState.DONE
    print(f"  ✓ Final state: {final_state.value}")

    # Test 5: Verify memory actually wiped
    print("\n[ Verify memory contents ]")
    assert all(b == 0x00 for b in kyber_sk)
    assert all(b == 0x00 for b in dilithium_sk)
    assert all(b == 0x00 for b in pepper)
    print(f"  ✓ All registered regions confirmed zeroed")

    # Test 6: Multi-pass wipe log
    print("\n[ Multi-pass wipe verification ]")
    wipe_passes = [w for w in fsm.wipe_log if w.get("region") == "kyber_sk" and "pass" in w and w["pass"] != "verify"]
    assert len(wipe_passes) == 3  # 3 passes per region
    patterns_used = [w["pattern"] for w in wipe_passes]
    assert patterns_used == ["0x0", "0xff", "0x0"]
    print(f"  ✓ 3-pass wipe confirmed: {patterns_used}")

    # Test 7: Reset and re-trigger via host command
    print("\n[ Reset and host-command trigger ]")
    fsm.reset()
    assert fsm.state == ZeroizeState.IDLE
    fsm.trigger(TriggerSource.HOST_COMMAND)
    final_state2 = fsm.run()
    assert final_state2 == ZeroizeState.DONE
    print(f"  ✓ Reset → re-triggered via HOST_COMMAND → DONE")

    # Test 8: Idempotent re-trigger (ignored if not IDLE)
    print("\n[ Idempotent re-trigger ]")
    fsm2 = ZeroizeFSM()
    region = bytearray(b"\x11" * 16)
    fsm2.register_region("test_region", region)
    fsm2.trigger(TriggerSource.SELF_TEST_FAIL)
    fsm2.trigger(TriggerSource.HOST_COMMAND)  # should be ignored (not IDLE)
    assert fsm2.trigger_source == TriggerSource.SELF_TEST_FAIL
    print(f"  ✓ Second trigger ignored while not IDLE")

    # Test 9: on_complete callback
    print("\n[ on_complete callback ]")
    callback_fired = []
    fsm3 = ZeroizeFSM()
    fsm3.on_complete = lambda: callback_fired.append(True)
    fsm3.register_region("cb_region", bytearray(b"\x99" * 8))
    fsm3.trigger(TriggerSource.WATCHDOG_TIMEOUT)
    fsm3.run()
    assert len(callback_fired) == 1
    print(f"  ✓ on_complete callback fired after DONE")

    # Test 10: Invalid transition rejected
    print("\n[ Invalid transition rejection ]")
    fsm4 = ZeroizeFSM()
    try:
        fsm4._transition(ZeroizeState.WIPING)  # IDLE → WIPING invalid
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ IDLE → WIPING correctly rejected (must go via TRIGGERED)")

    # Test 11: run() requires TRIGGERED state
    print("\n[ run() state precondition ]")
    fsm5 = ZeroizeFSM()
    try:
        fsm5.run()  # still IDLE
        assert False, "Should have raised"
    except RuntimeError:
        print(f"  ✓ run() correctly rejected from IDLE state")

    # Test 12: Statistics
    print("\n[ FSM statistics ]")
    stats = fsm.stats()
    print(f"  State: {stats['state']}")
    print(f"  Trigger source: {stats['trigger_source']}")
    print(f"  Regions: {stats['regions_registered']}")
    print(f"  Wipe operations: {stats['wipe_operations']}")
    print(f"  Transitions: {stats['transitions']}")
    print(f"  ✓ Statistics tracked correctly")

    print("\n  All checks passed.\n")
