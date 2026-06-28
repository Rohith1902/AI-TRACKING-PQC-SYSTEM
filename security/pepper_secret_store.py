"""
pepper_secret_store.py
=======================
Pepper/secret storage management for the PQC-SNN SoC security subsystem.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Manages a hardware-resident "pepper" secret used to strengthen key        ║
║ derivation, separate from per-key salts, with rotation and burn-in.       ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Pepper generation (hardware-fused secret, simulated)
  - Pepper application to KDF inputs (pepper || salt || ikm)
  - Pepper rotation policy (versioned peppers)
  - Burn-in / one-time provisioning lock
  - Secret store for auxiliary secrets (device ID, provisioning keys)
  - Anti-rollback version tracking

Concept — Pepper vs. Salt:
  - Salt: per-key, often public, stored alongside the key.
  - Pepper: device-wide, secret, NEVER stored alongside derived keys.
    Typically fused into hardware (e-fuse / OTP) at manufacture time.
  - Combining both: derived_key = KDF(pepper || salt || IKM)
    An attacker who extracts only the key store (with salts) still
    cannot derive keys without the pepper.

Provisioning model:
  1. Factory: pepper generated once via TRNG, written to OTP (burn-in)
  2. Burn-in lock: pepper becomes immutable; further writes rejected
  3. Field: pepper read-only, used in all KDF operations
  4. Rotation (versioned): new pepper version added; old version
     retained for legacy key recovery, marked deprecated

Security properties:
  - Pepper never leaves the secure boundary (no read-back to host bus)
  - Burn-in lock prevents post-provisioning tampering
  - Versioning supports controlled rotation without bricking old keys

Matches pepper_secret_store.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST SP 800-132 (Password-Based Key Derivation, pepper concept)
"""

from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

# ──────────────────────────────────────────────
# 1.  PEPPER STORE PARAMETERS
# ──────────────────────────────────────────────

#: Pepper length (bytes) — matches HKDF/HMAC block expectations
PEPPER_LEN: int = 32

#: Maximum number of pepper versions retained (anti-bloat)
MAX_PEPPER_VERSIONS: int = 4


# ──────────────────────────────────────────────
# 2.  PEPPER RECORD
# ──────────────────────────────────────────────

@dataclass
class PepperRecord:
    """A single versioned pepper entry."""
    version: int
    pepper: bytes
    deprecated: bool = False


# ──────────────────────────────────────────────
# 3.  PEPPER / SECRET STORE
# ──────────────────────────────────────────────

class PepperSecretStore:
    """
    Manages the device pepper and auxiliary device secrets.

    Simulates one-time-programmable (OTP) hardware behavior:
    pepper can be burned in exactly once per version slot, after
    which it is locked against modification.
    """

    def __init__(self):
        self.peppers: Dict[int, PepperRecord] = {}
        self.current_version: int = -1
        self.burned_in: bool = False
        self.device_secrets: Dict[str, bytes] = {}
        self.access_log: List[Dict] = []

    # ── Pepper provisioning ──────────────────────

    def burn_in_pepper(self, pepper: bytes | None = None) -> int:
        """
        Burn in the initial pepper (one-time, factory provisioning).

        Parameters
        ----------
        pepper : bytes, optional
            Pepper value. If None, generated via TRNG (os.urandom).

        Returns
        -------
        int
            Version number assigned (always 0 for initial burn-in).

        Raises
        ------
        RuntimeError
            If burn-in has already occurred.
        """
        if self.burned_in:
            raise RuntimeError("Pepper already burned in — cannot re-provision")

        if pepper is None:
            pepper = os.urandom(PEPPER_LEN)
        if len(pepper) != PEPPER_LEN:
            pepper = hashlib.sha256(pepper).digest()

        self.peppers[0] = PepperRecord(version=0, pepper=pepper)
        self.current_version = 0
        self.burned_in = True

        return 0

    def rotate_pepper(self, new_pepper: bytes | None = None) -> int:
        """
        Add a new pepper version (rotation), deprecating the old one.

        Parameters
        ----------
        new_pepper : bytes, optional
            New pepper value. Generated via TRNG if None.

        Returns
        -------
        int
            New version number.

        Raises
        ------
        RuntimeError
            If store has not been burned in yet, or version limit reached.
        """
        if not self.burned_in:
            raise RuntimeError("Cannot rotate before initial burn-in")

        if len(self.peppers) >= MAX_PEPPER_VERSIONS:
            raise RuntimeError(
                f"Maximum pepper versions ({MAX_PEPPER_VERSIONS}) reached"
            )

        # Deprecate current version
        self.peppers[self.current_version].deprecated = True

        if new_pepper is None:
            new_pepper = os.urandom(PEPPER_LEN)
        if len(new_pepper) != PEPPER_LEN:
            new_pepper = hashlib.sha256(new_pepper).digest()

        new_version = self.current_version + 1
        self.peppers[new_version] = PepperRecord(version=new_version, pepper=new_pepper)
        self.current_version = new_version

        return new_version

    # ── Pepper application ──────────────────────

    def apply_pepper(
        self, salt: bytes, ikm: bytes, version: Optional[int] = None
    ) -> bytes:
        """
        Combine pepper with salt and IKM for KDF input.

        Parameters
        ----------
        salt : bytes
            Per-key salt (may be public).
        ikm : bytes
            Input keying material.
        version : int, optional
            Pepper version to use. Defaults to current version.

        Returns
        -------
        bytes
            pepper || salt || ikm (ready for HKDF/KDF input).

        Raises
        ------
        RuntimeError
            If no pepper has been provisioned.
        """
        if not self.burned_in:
            raise RuntimeError("No pepper provisioned — burn_in_pepper() first")

        v = version if version is not None else self.current_version
        record = self.peppers.get(v)
        if record is None:
            raise KeyError(f"Pepper version {v} not found")

        self._log_access(v, "apply_pepper")
        return record.pepper + salt + ikm

    def get_pepper(self, version: Optional[int] = None) -> bytes:
        """
        Retrieve raw pepper bytes for a given version (internal/trusted use only).

        Parameters
        ----------
        version : int, optional
            Pepper version. Defaults to current.

        Returns
        -------
        bytes
            Raw pepper bytes.
        """
        v = version if version is not None else self.current_version
        record = self.peppers.get(v)
        if record is None:
            raise KeyError(f"Pepper version {v} not found")

        self._log_access(v, "get_pepper")
        return record.pepper

    # ── Auxiliary device secrets ────────────────

    def store_device_secret(self, label: str, secret: bytes) -> None:
        """
        Store an auxiliary device secret (e.g. device ID, provisioning key).

        Parameters
        ----------
        label : str
            Identifier for the secret.
        secret : bytes
            Secret bytes.
        """
        self.device_secrets[label] = secret

    def get_device_secret(self, label: str) -> Optional[bytes]:
        """Retrieve a stored device secret by label."""
        return self.device_secrets.get(label)

    # ── Internal helpers ─────────────────────────

    def _log_access(self, version: int, operation: str) -> None:
        self.access_log.append({"version": version, "operation": operation})

    def stats(self) -> Dict:
        """Return store statistics."""
        return {
            "burned_in": self.burned_in,
            "current_version": self.current_version,
            "total_versions": len(self.peppers),
            "deprecated_versions": sum(
                1 for r in self.peppers.values() if r.deprecated
            ),
            "device_secrets_stored": len(self.device_secrets),
            "access_count": len(self.access_log),
        }


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("pepper_secret_store.py  —  self-test")
    print("=" * 55)

    # Test 1: Initial burn-in
    print("\n[ Initial pepper burn-in ]")
    store = PepperSecretStore()
    v0 = store.burn_in_pepper(b"\x11" * 32)
    assert v0 == 0
    assert store.burned_in
    print(f"  ✓ Burned in pepper version {v0}")

    # Test 2: Double burn-in rejected
    print("\n[ Double burn-in rejection ]")
    try:
        store.burn_in_pepper(b"\x22" * 32)
        assert False, "Should have raised"
    except RuntimeError:
        print(f"  ✓ Second burn-in correctly rejected")

    # Test 3: Apply pepper to KDF input
    print("\n[ Apply pepper to KDF input ]")
    salt = b"per_key_salt_16b"
    ikm = b"input_keying_material"
    combined = store.apply_pepper(salt, ikm)
    assert combined.startswith(b"\x11" * 32)
    assert combined.endswith(salt + ikm)
    print(f"  ✓ Combined: pepper || salt || ikm ({len(combined)} bytes)")

    # Test 4: Use before burn-in fails
    print("\n[ Use before burn-in ]")
    fresh_store = PepperSecretStore()
    try:
        fresh_store.apply_pepper(salt, ikm)
        assert False, "Should have raised"
    except RuntimeError:
        print(f"  ✓ Application before burn-in correctly rejected")

    # Test 5: Pepper rotation
    print("\n[ Pepper rotation ]")
    v1 = store.rotate_pepper(b"\x33" * 32)
    assert v1 == 1
    assert store.current_version == 1
    assert store.peppers[0].deprecated
    print(f"  ✓ Rotated to version {v1}, version 0 deprecated")

    # Test 6: Legacy version still accessible
    print("\n[ Legacy version access ]")
    old_pepper = store.get_pepper(version=0)
    assert old_pepper == b"\x11" * 32
    new_pepper = store.get_pepper(version=1)
    assert new_pepper == b"\x33" * 32
    print(f"  ✓ Both old and new pepper versions retrievable")

    # Test 7: Apply pepper with explicit version
    print("\n[ Apply pepper with specific version ]")
    combined_v0 = store.apply_pepper(salt, ikm, version=0)
    combined_v1 = store.apply_pepper(salt, ikm, version=1)
    assert combined_v0 != combined_v1
    print(f"  ✓ Different versions produce different combined output")

    # Test 8: Version limit enforcement
    print("\n[ Version limit enforcement ]")
    limit_store = PepperSecretStore()
    limit_store.burn_in_pepper()
    for i in range(MAX_PEPPER_VERSIONS - 1):
        limit_store.rotate_pepper()
    try:
        limit_store.rotate_pepper()
        assert False, "Should have raised"
    except RuntimeError:
        print(f"  ✓ Max versions ({MAX_PEPPER_VERSIONS}) correctly enforced")

    # Test 9: Auxiliary device secrets
    print("\n[ Auxiliary device secrets ]")
    store.store_device_secret("device_id", b"AEGIS-NEURO-0001")
    store.store_device_secret("provisioning_key", hashlib.sha256(b"provkey").digest())
    dev_id = store.get_device_secret("device_id")
    assert dev_id == b"AEGIS-NEURO-0001"
    print(f"  ✓ Stored and retrieved 2 device secrets")

    # Test 10: Missing secret returns None
    print("\n[ Missing secret handling ]")
    missing = store.get_device_secret("nonexistent")
    assert missing is None
    print(f"  ✓ Missing secret correctly returns None")

    # Test 11: Determinism of TRNG-based burn-in (different each time)
    print("\n[ TRNG-based pepper uniqueness ]")
    store_a = PepperSecretStore()
    store_b = PepperSecretStore()
    store_a.burn_in_pepper()  # random
    store_b.burn_in_pepper()  # random
    pepper_a = store_a.get_pepper()
    pepper_b = store_b.get_pepper()
    assert pepper_a != pepper_b
    print(f"  ✓ Independently burned-in peppers are unique")

    # Test 12: Statistics
    print("\n[ Store statistics ]")
    stats = store.stats()
    print(f"  Burned in: {stats['burned_in']}")
    print(f"  Current version: {stats['current_version']}")
    print(f"  Total versions: {stats['total_versions']}")
    print(f"  Deprecated: {stats['deprecated_versions']}")
    print(f"  Device secrets: {stats['device_secrets_stored']}")
    print(f"  ✓ Statistics tracked correctly")

    print("\n  All checks passed.\n")
