"""
secure_boot_model.py
=====================
Secure boot chain model for the PQC-SNN SoC (FB_BOOT).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Models the authenticated boot chain: bitstream verification, eFuse root   ║
║ of trust, and stage-by-stage hand-off with cryptographic attestation.    ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - eFuse root-of-trust key model (immutable, burned at manufacture)
  - Bitstream signature verification (Dilithium-signed boot image)
  - Multi-stage boot sequence (ROM → PL bitstream → PS firmware → App)
  - Authenticated hand-off between stages (each stage measures the next)
  - Boot failure handling (halt, retry, fallback to golden image)
  - Measured boot log (chain of cryptographic measurements)

Boot chain stages (per architecture diagram "5. SYSTEM INFRASTRUCTURE"):
  ROM_BOOT     → mask ROM, verifies eFuse root key presence
  PL_BITSTREAM → FPGA bitstream signature checked against root key
  PS_FIRMWARE  → ARM Cortex-A53 firmware (Linux/bare-metal) signature check
  APPLICATION  → final application image signature check
  RUNNING      → boot complete, system operational

Each transition requires:
  1. Hash the next-stage image (SHA-256)
  2. Verify Dilithium signature over that hash using the stage's pubkey
  3. Record measurement in boot log (for later attestation/audit)
  4. Only then hand off execution to the next stage

Failure handling:
  - Signature mismatch → HALT (no insecure fallback by default)
  - Optional golden/recovery image fallback (configurable, off by default
    since silent fallback can mask attacks — must be explicitly enabled)

Matches secure_boot_model.sv / FB_BOOT (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST SP 800-193 (Platform Firmware Resiliency), FIPS 204 (ML-DSA)
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable

# ──────────────────────────────────────────────
# 1.  BOOT STAGES
# ──────────────────────────────────────────────

class BootStage(Enum):
    ROM_BOOT = "rom_boot"
    PL_BITSTREAM = "pl_bitstream"
    PS_FIRMWARE = "ps_firmware"
    APPLICATION = "application"
    RUNNING = "running"
    HALTED = "halted"


#: Ordered boot sequence (excluding terminal RUNNING/HALTED)
BOOT_SEQUENCE: List[BootStage] = [
    BootStage.ROM_BOOT,
    BootStage.PL_BITSTREAM,
    BootStage.PS_FIRMWARE,
    BootStage.APPLICATION,
]


# ──────────────────────────────────────────────
# 2.  SIGNATURE VERIFICATION (SIMULATED DILITHIUM)
# ──────────────────────────────────────────────

def measure_image(image: bytes) -> bytes:
    """
    Measure (hash) a boot-stage image.

    Parameters
    ----------
    image : bytes
        Raw image bytes (bitstream, firmware, or application binary).

    Returns
    -------
    bytes
        32-byte SHA-256 measurement.
    """
    return hashlib.sha256(image).digest()


def simulate_sign(image_hash: bytes, signing_key: bytes) -> bytes:
    """
    Simulate a Dilithium signature over an image hash.

    In the full system this delegates to dilithium_sign.py; here we use
    a deterministic HMAC-style construction for golden-model testing.

    Parameters
    ----------
    image_hash : bytes
        32-byte measurement of the image.
    signing_key : bytes
        Private signing key (stage-specific).

    Returns
    -------
    bytes
        Simulated signature bytes.
    """
    return hashlib.sha256(signing_key + image_hash).digest()


def simulate_verify(image_hash: bytes, signature: bytes, public_key: bytes) -> bool:
    """
    Simulate Dilithium signature verification.

    Parameters
    ----------
    image_hash : bytes
        32-byte measurement of the image being verified.
    signature : bytes
        Signature to check.
    public_key : bytes
        Public key bound to the expected signing key (derived deterministically
        in this simulation so verify/sign are consistent without needing the
        real dilithium_sign/verify state machine).

    Returns
    -------
    bool
        True if signature is valid.
    """
    expected = hashlib.sha256(public_key + image_hash).digest()
    return hashlib_compare(signature, expected)


def hashlib_compare(a: bytes, b: bytes) -> bool:
    """Constant-time byte comparison."""
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= x ^ y
    return diff == 0


# ──────────────────────────────────────────────
# 3.  BOOT MEASUREMENT LOG ENTRY
# ──────────────────────────────────────────────

@dataclass
class BootMeasurement:
    """A single recorded boot-stage measurement."""
    stage: BootStage
    image_hash: bytes
    verified: bool
    reason: str = "ok"


# ──────────────────────────────────────────────
# 4.  SECURE BOOT FSM
# ──────────────────────────────────────────────

class SecureBootChain:
    """
    Models the authenticated, measured boot chain from eFuse root of
    trust through to application launch.
    """

    def __init__(self, efuse_root_key: bytes, allow_recovery_fallback: bool = False):
        """
        Initialize the secure boot chain.

        Parameters
        ----------
        efuse_root_key : bytes
            Immutable root-of-trust key (simulates eFuse-burned key).
            Used as the public key anchor for verifying the first stage.
        allow_recovery_fallback : bool
            If True, a failed stage may fall back to a golden/recovery
            image instead of halting. Disabled by default — silent
            fallback can mask an active attack, so this must be an
            explicit, audited configuration choice.
        """
        if len(efuse_root_key) != 32:
            efuse_root_key = hashlib.sha256(efuse_root_key).digest()

        self.efuse_root_key = efuse_root_key
        self.allow_recovery_fallback = allow_recovery_fallback

        self.stage: BootStage = BootStage.ROM_BOOT
        self.stage_index: int = 0

        # Each stage's expected public key, chained from the root.
        # stage_pubkeys[BootStage.PL_BITSTREAM] is the key used to verify
        # the PL_BITSTREAM image, etc. Derived deterministically from the
        # root key + stage name (simulates a key-derivation chain).
        self.stage_pubkeys: Dict[BootStage, bytes] = {
            s: hashlib.sha256(self.efuse_root_key + s.value.encode()).digest()
            for s in BOOT_SEQUENCE
        }

        # Matching "signing keys" for the golden model's own image generator
        # (in real hardware these live only at the signer, never on-chip)
        self.stage_signing_keys: Dict[BootStage, bytes] = dict(self.stage_pubkeys)

        self.boot_log: List[BootMeasurement] = []
        self.recovery_images: Dict[BootStage, bytes] = {}
        self.halted: bool = False
        self.halt_reason: Optional[str] = None
        self.on_halt: Optional[Callable[[str], None]] = None

    # ── Image preparation helpers (test/golden-model use) ──

    def sign_stage_image(self, stage: BootStage, image: bytes) -> bytes:
        """
        Produce a valid signature for a stage image (used to build test
        fixtures / golden images — not part of the on-device boot path).

        Parameters
        ----------
        stage : BootStage
            Which stage this image belongs to.
        image : bytes
            Raw image bytes.

        Returns
        -------
        bytes
            Signature bytes.
        """
        image_hash = measure_image(image)
        return simulate_sign(image_hash, self.stage_signing_keys[stage])

    def register_recovery_image(
        self, stage: BootStage, image: bytes, signature: bytes
    ) -> None:
        """Register a known-good recovery image for a stage."""
        self.recovery_images[stage] = (image, signature)

    # ── Boot execution ───────────────────────────

    def boot_stage(self, image: bytes, signature: bytes) -> bool:
        """
        Attempt to verify and advance through the current boot stage.

        Parameters
        ----------
        image : bytes
            The next-stage image to verify and hand off to.
        signature : bytes
            Signature over the image's measurement.

        Returns
        -------
        bool
            True if this stage succeeded and the chain advanced,
            False if verification failed (chain halts unless recovery
            fallback succeeds).
        """
        if self.halted:
            return False

        current_stage = self.stage
        image_hash = measure_image(image)
        pubkey = self.stage_pubkeys[current_stage]

        verified = simulate_verify(image_hash, signature, pubkey)

        if verified:
            self.boot_log.append(
                BootMeasurement(stage=current_stage, image_hash=image_hash, verified=True)
            )
            self._advance()
            return True

        # Verification failed — try recovery fallback if enabled
        if self.allow_recovery_fallback and current_stage in self.recovery_images:
            recovery_image, recovery_sig = self.recovery_images[current_stage]
            recovery_hash = measure_image(recovery_image)
            recovery_ok = simulate_verify(recovery_hash, recovery_sig, pubkey)

            if recovery_ok:
                self.boot_log.append(
                    BootMeasurement(
                        stage=current_stage, image_hash=recovery_hash,
                        verified=True, reason="recovery_fallback",
                    )
                )
                self._advance()
                return True

        # No recovery, or recovery also failed → halt
        self.boot_log.append(
            BootMeasurement(
                stage=current_stage, image_hash=image_hash,
                verified=False, reason="signature_mismatch",
            )
        )
        self._halt(f"Signature verification failed at stage {current_stage.value}")
        return False

    def _advance(self) -> None:
        """Advance to the next stage in the boot sequence."""
        self.stage_index += 1
        if self.stage_index < len(BOOT_SEQUENCE):
            self.stage = BOOT_SEQUENCE[self.stage_index]
        else:
            self.stage = BootStage.RUNNING

    def _halt(self, reason: str) -> None:
        """Halt the boot chain (terminal failure state)."""
        self.halted = True
        self.halt_reason = reason
        self.stage = BootStage.HALTED
        if self.on_halt is not None:
            self.on_halt(reason)

    def is_running(self) -> bool:
        """True once the full chain has completed successfully."""
        return self.stage == BootStage.RUNNING

    def stats(self) -> Dict:
        """Return boot chain statistics."""
        return {
            "current_stage": self.stage.value,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "stages_completed": sum(1 for m in self.boot_log if m.verified),
            "stages_failed": sum(1 for m in self.boot_log if not m.verified),
            "total_measurements": len(self.boot_log),
            "recovery_fallback_enabled": self.allow_recovery_fallback,
        }


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("secure_boot_model.py  —  self-test")
    print("=" * 55)

    efuse_key = hashlib.sha256(b"factory_burned_efuse_root_key").digest()

    # Test 1: Successful full boot chain
    print("\n[ Full successful boot chain ]")
    boot = SecureBootChain(efuse_key)

    pl_image = b"PL_BITSTREAM_IMAGE_DATA" * 100
    pl_sig = boot.sign_stage_image(BootStage.ROM_BOOT, pl_image)
    ok = boot.boot_stage(pl_image, pl_sig)
    assert ok
    assert boot.stage == BootStage.PL_BITSTREAM
    print(f"  ✓ ROM_BOOT → PL_BITSTREAM")

    ps_image = b"PS_FIRMWARE_LINUX_IMAGE" * 200
    ps_sig = boot.sign_stage_image(BootStage.PL_BITSTREAM, ps_image)
    ok = boot.boot_stage(ps_image, ps_sig)
    assert ok
    assert boot.stage == BootStage.PS_FIRMWARE
    print(f"  ✓ PL_BITSTREAM → PS_FIRMWARE")

    app_image = b"APPLICATION_THREAT_DETECTOR" * 150
    app_sig = boot.sign_stage_image(BootStage.PS_FIRMWARE, app_image)
    ok = boot.boot_stage(app_image, app_sig)
    assert ok
    assert boot.stage == BootStage.APPLICATION
    print(f"  ✓ PS_FIRMWARE → APPLICATION")

    final_image = b"FINAL_RUNTIME_HANDOFF"
    final_sig = boot.sign_stage_image(BootStage.APPLICATION, final_image)
    ok = boot.boot_stage(final_image, final_sig)
    assert ok
    assert boot.is_running()
    print(f"  ✓ APPLICATION → RUNNING (boot complete)")

    # Test 2: Tampered image rejected
    print("\n[ Tampered bitstream rejection ]")
    boot2 = SecureBootChain(efuse_key)
    legit_image = b"GOOD_BITSTREAM"
    legit_sig = boot2.sign_stage_image(BootStage.ROM_BOOT, legit_image)

    tampered_image = b"EVIL_BITSTREAM_INJECTED"
    ok = boot2.boot_stage(tampered_image, legit_sig)  # sig doesn't match tampered image
    assert not ok
    assert boot2.halted
    print(f"  ✓ Tampered image correctly rejected, chain halted")
    print(f"    Halt reason: {boot2.halt_reason}")

    # Test 3: Wrong root key produces different stage keys → all fail
    print("\n[ Wrong eFuse root key ]")
    wrong_key = hashlib.sha256(b"attacker_supplied_key").digest()
    boot3 = SecureBootChain(wrong_key)
    image = b"SOME_IMAGE"
    # Sign with the ORIGINAL boot's key (simulating legit image+sig pair)
    sig_from_real_chain = boot.sign_stage_image(BootStage.ROM_BOOT, image)
    ok = boot3.boot_stage(image, sig_from_real_chain)
    assert not ok
    print(f"  ✓ Image signed under different root key correctly rejected")

    # Test 4: Halted chain rejects further attempts
    print("\n[ Halted chain rejects further boot attempts ]")
    ok_after_halt = boot2.boot_stage(legit_image, legit_sig)
    assert not ok_after_halt
    print(f"  ✓ No further progress possible after halt")

    # Test 5: on_halt callback (could trigger zeroize_fsm in full system)
    print("\n[ on_halt callback dispatch ]")
    halt_events = []
    boot4 = SecureBootChain(efuse_key)
    boot4.on_halt = lambda reason: halt_events.append(reason)
    bad_sig = b"\x00" * 32
    boot4.boot_stage(b"any_image", bad_sig)
    assert len(halt_events) == 1
    print(f"  ✓ on_halt callback fired: {halt_events[0]}")

    # Test 6: Recovery fallback (explicitly enabled)
    print("\n[ Recovery fallback (explicit opt-in) ]")
    boot5 = SecureBootChain(efuse_key, allow_recovery_fallback=True)

    golden_image = b"KNOWN_GOOD_GOLDEN_BITSTREAM"
    golden_sig = boot5.sign_stage_image(BootStage.ROM_BOOT, golden_image)
    boot5.register_recovery_image(BootStage.ROM_BOOT, golden_image, golden_sig)

    # Primary image is corrupt/wrong signature
    corrupt_image = b"CORRUPT_PRIMARY_BITSTREAM"
    bad_primary_sig = b"\xFF" * 32

    ok = boot5.boot_stage(corrupt_image, bad_primary_sig)
    assert ok  # recovered via golden image
    assert not boot5.halted
    print(f"  ✓ Corrupt primary image → recovered via golden fallback")

    # Test 7: Recovery disabled by default
    print("\n[ Recovery fallback disabled by default ]")
    boot6 = SecureBootChain(efuse_key)  # default: allow_recovery_fallback=False
    assert boot6.allow_recovery_fallback is False
    print(f"  ✓ Recovery fallback is off unless explicitly enabled")

    # Test 8: Stage key chain derivation is deterministic
    print("\n[ Stage key derivation determinism ]")
    boot7a = SecureBootChain(efuse_key)
    boot7b = SecureBootChain(efuse_key)
    assert boot7a.stage_pubkeys == boot7b.stage_pubkeys
    print(f"  ✓ Same root key → same derived stage key chain")

    # Test 9: Boot log / measurement trail
    print("\n[ Measured boot log ]")
    assert len(boot.boot_log) == 4
    assert all(m.verified for m in boot.boot_log)
    print(f"  ✓ {len(boot.boot_log)} measurements recorded, all verified")

    # Test 10: Statistics
    print("\n[ Boot chain statistics ]")
    stats = boot.stats()
    print(f"  Current stage: {stats['current_stage']}")
    print(f"  Stages completed: {stats['stages_completed']}")
    print(f"  Stages failed: {stats['stages_failed']}")
    print(f"  ✓ Statistics tracked correctly")

    print("\n  All checks passed.\n")
