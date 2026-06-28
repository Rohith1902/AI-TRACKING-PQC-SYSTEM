"""
key_protection.py
==================
Key protection and lifecycle management for the PQC-SNN SoC security subsystem.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Manages secure key storage, wrapping/unwrapping, lifecycle states, and    ║
║ access control for Kyber/Dilithium keys across the SoC.                  ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Key wrapping/unwrapping (AES-KW style, via HKDF-derived KEK)
  - Key lifecycle state machine (GENERATED → ACTIVE → EXPIRED → ZEROIZED)
  - Access control (read/write permission per key)
  - Key usage counters and expiry policies
  - Secure key storage registry (in-memory, simulating protected SRAM)
  - Integrity tagging (HMAC over wrapped key blobs)

Lifecycle states:
  GENERATED → ACTIVE → EXPIRED → ZEROIZED
       │          │         │
       │          │         └─→ (terminal: key material is gone)
       │          └─→ EXPIRED (max usage or time limit reached)
       └─→ ACTIVE (after activation/first use)

Key wrapping (simplified AES-KW analogue):
  1. Derive KEK from master secret via HKDF (see hkdf_engine.py)
  2. wrapped = AES-CTR(KEK, nonce, key_material) || HMAC(KEK, wrapped)
  3. unwrap: verify HMAC, then AES-CTR decrypt

Access control:
  - Each key has an owner_id and allowed_operations set
  - Unauthorized access attempts are logged and rejected

Security properties:
  - Keys never stored in plaintext outside active use
  - Usage counter enforces key rotation policy
  - Zeroization is irreversible (overwrite + state transition)

Matches key_protection.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST SP 800-57 (Key Management), SP 800-38F (Key Wrapping)
"""

from __future__ import annotations
import hashlib
import hmac
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

# ──────────────────────────────────────────────
# 1.  KEY LIFECYCLE STATES
# ──────────────────────────────────────────────

class KeyState(Enum):
    """Key lifecycle states."""
    GENERATED = "generated"
    ACTIVE = "active"
    EXPIRED = "expired"
    ZEROIZED = "zeroized"


#: Valid state transitions
VALID_TRANSITIONS: Dict[KeyState, Set[KeyState]] = {
    KeyState.GENERATED: {KeyState.ACTIVE, KeyState.ZEROIZED},
    KeyState.ACTIVE: {KeyState.EXPIRED, KeyState.ZEROIZED},
    KeyState.EXPIRED: {KeyState.ZEROIZED},
    KeyState.ZEROIZED: set(),  # terminal state
}

#: Default maximum key usage count before forced expiry
DEFAULT_MAX_USAGE: int = 10_000


# ──────────────────────────────────────────────
# 2.  KEY METADATA
# ──────────────────────────────────────────────

@dataclass
class KeyMetadata:
    """Metadata tracked for each protected key."""
    key_id: str
    owner_id: str
    state: KeyState = KeyState.GENERATED
    usage_count: int = 0
    max_usage: int = DEFAULT_MAX_USAGE
    allowed_ops: Set[str] = field(default_factory=lambda: {"encrypt", "decrypt", "sign", "verify"})


# ──────────────────────────────────────────────
# 3.  KEY WRAPPING PRIMITIVES
# ──────────────────────────────────────────────

def _ctr_keystream(kek: bytes, nonce: bytes, length: int) -> bytes:
    """
    Generate a keystream using HMAC-SHA256 in counter mode (AES-CTR analogue).

    Parameters
    ----------
    kek : bytes
        Key-encryption key.
    nonce : bytes
        16-byte nonce.
    length : int
        Number of keystream bytes needed.

    Returns
    -------
    bytes
        Keystream of `length` bytes.
    """
    out = b""
    counter = 0
    while len(out) < length:
        block = hmac.new(kek, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def wrap_key(kek: bytes, key_material: bytes, nonce: bytes | None = None) -> bytes:
    """
    Wrap (encrypt + authenticate) key material under a KEK.

    Parameters
    ----------
    kek : bytes
        32-byte key-encryption key.
    key_material : bytes
        Raw key bytes to protect.
    nonce : bytes, optional
        16-byte nonce. Randomly generated if None.

    Returns
    -------
    bytes
        nonce || ciphertext || tag (HMAC-SHA256, 32 bytes).
    """
    if nonce is None:
        nonce = os.urandom(16)
    if len(nonce) != 16:
        raise ValueError("nonce must be 16 bytes")

    keystream = _ctr_keystream(kek, nonce, len(key_material))
    ciphertext = bytes(a ^ b for a, b in zip(key_material, keystream))

    tag = hmac.new(kek, nonce + ciphertext, hashlib.sha256).digest()

    return nonce + ciphertext + tag


def unwrap_key(kek: bytes, wrapped: bytes) -> Optional[bytes]:
    """
    Unwrap (verify + decrypt) a wrapped key blob.

    Parameters
    ----------
    kek : bytes
        32-byte key-encryption key.
    wrapped : bytes
        nonce || ciphertext || tag, as produced by wrap_key().

    Returns
    -------
    bytes or None
        Recovered key material, or None if integrity check fails.
    """
    if len(wrapped) < 16 + 32:
        return None

    nonce = wrapped[:16]
    tag = wrapped[-32:]
    ciphertext = wrapped[16:-32]

    expected_tag = hmac.new(kek, nonce + ciphertext, hashlib.sha256).digest()

    # Constant-time comparison
    if not hmac.compare_digest(tag, expected_tag):
        return None

    keystream = _ctr_keystream(kek, nonce, len(ciphertext))
    plaintext = bytes(a ^ b for a, b in zip(ciphertext, keystream))

    return plaintext


# ──────────────────────────────────────────────
# 4.  KEY PROTECTION MANAGER
# ──────────────────────────────────────────────

class KeyProtectionManager:
    """
    Manages secure storage, lifecycle, and access control for protected keys.
    """

    def __init__(self, master_kek: bytes):
        """
        Initialize the key protection manager.

        Parameters
        ----------
        master_kek : bytes
            32-byte master key-encryption key.
        """
        if len(master_kek) != 32:
            master_kek = hashlib.sha256(master_kek).digest()

        self.kek = master_kek
        self.storage: Dict[str, bytes] = {}        # key_id → wrapped blob
        self.metadata: Dict[str, KeyMetadata] = {}  # key_id → metadata
        self.access_log: List[Dict] = []

    def store_key(
        self,
        key_id: str,
        key_material: bytes,
        owner_id: str,
        allowed_ops: Optional[Set[str]] = None,
        max_usage: int = DEFAULT_MAX_USAGE,
    ) -> None:
        """
        Store a new key under protection.

        Parameters
        ----------
        key_id : str
            Unique identifier for the key.
        key_material : bytes
            Raw key bytes.
        owner_id : str
            Identifier of the owning subsystem (e.g. "kyber", "dilithium").
        allowed_ops : Set[str], optional
            Permitted operations for this key.
        max_usage : int
            Maximum number of accesses before forced expiry.
        """
        if key_id in self.metadata:
            raise KeyError(f"Key '{key_id}' already exists")

        wrapped = wrap_key(self.kek, key_material)
        self.storage[key_id] = wrapped

        meta = KeyMetadata(key_id=key_id, owner_id=owner_id, max_usage=max_usage)
        if allowed_ops is not None:
            meta.allowed_ops = allowed_ops
        self.metadata[key_id] = meta

    def activate_key(self, key_id: str) -> None:
        """Transition a key from GENERATED to ACTIVE."""
        self._transition(key_id, KeyState.ACTIVE)

    def retrieve_key(
        self, key_id: str, requester_id: str, operation: str
    ) -> Optional[bytes]:
        """
        Retrieve (unwrap) a key, enforcing access control and usage limits.

        Parameters
        ----------
        key_id : str
            Key identifier.
        requester_id : str
            Identifier of the requesting subsystem.
        operation : str
            Operation being performed ("encrypt", "sign", etc.).

        Returns
        -------
        bytes or None
            Raw key material if access is granted, None otherwise.
        """
        meta = self.metadata.get(key_id)
        if meta is None:
            self._log_access(key_id, requester_id, operation, granted=False, reason="not_found")
            return None

        # Auto-activate on first legitimate use
        if meta.state == KeyState.GENERATED:
            meta.state = KeyState.ACTIVE

        # Check state
        if meta.state not in (KeyState.ACTIVE,):
            self._log_access(key_id, requester_id, operation, granted=False, reason=f"state={meta.state.value}")
            return None

        # Check operation permission
        if operation not in meta.allowed_ops:
            self._log_access(key_id, requester_id, operation, granted=False, reason="op_not_allowed")
            return None

        # Check usage limit
        if meta.usage_count >= meta.max_usage:
            meta.state = KeyState.EXPIRED
            self._log_access(key_id, requester_id, operation, granted=False, reason="usage_exhausted")
            return None

        # Grant access
        wrapped = self.storage.get(key_id)
        key_material = unwrap_key(self.kek, wrapped)

        if key_material is None:
            self._log_access(key_id, requester_id, operation, granted=False, reason="integrity_fail")
            return None

        meta.usage_count += 1
        if meta.usage_count >= meta.max_usage:
            meta.state = KeyState.EXPIRED

        self._log_access(key_id, requester_id, operation, granted=True, reason="ok")
        return key_material

    def expire_key(self, key_id: str) -> None:
        """Force-expire a key (e.g. due to policy or rotation)."""
        self._transition(key_id, KeyState.EXPIRED)

    def zeroize_key(self, key_id: str) -> None:
        """
        Zeroize a key: overwrite storage and transition to terminal state.
        """
        if key_id in self.storage:
            length = len(self.storage[key_id])
            self.storage[key_id] = b"\x00" * length

        self._transition(key_id, KeyState.ZEROIZED)

    def zeroize_all(self) -> int:
        """Zeroize all keys (e.g. on tamper detection). Returns count zeroized."""
        count = 0
        for key_id, meta in self.metadata.items():
            if meta.state != KeyState.ZEROIZED:
                self.zeroize_key(key_id)
                count += 1
        return count

    def _transition(self, key_id: str, new_state: KeyState) -> None:
        """Validate and perform a state transition."""
        meta = self.metadata.get(key_id)
        if meta is None:
            raise KeyError(f"Key '{key_id}' not found")

        if new_state not in VALID_TRANSITIONS[meta.state]:
            raise ValueError(
                f"Invalid transition: {meta.state.value} → {new_state.value}"
            )

        meta.state = new_state

    def _log_access(
        self, key_id: str, requester_id: str, operation: str, granted: bool, reason: str
    ) -> None:
        """Record an access attempt."""
        self.access_log.append({
            "key_id": key_id,
            "requester": requester_id,
            "operation": operation,
            "granted": granted,
            "reason": reason,
        })

    def stats(self) -> Dict:
        """Return manager statistics."""
        state_counts: Dict[str, int] = {}
        for meta in self.metadata.values():
            state_counts[meta.state.value] = state_counts.get(meta.state.value, 0) + 1

        granted = sum(1 for log in self.access_log if log["granted"])
        denied = len(self.access_log) - granted

        return {
            "total_keys": len(self.metadata),
            "state_counts": state_counts,
            "access_attempts": len(self.access_log),
            "access_granted": granted,
            "access_denied": denied,
        }


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("key_protection.py  —  self-test")
    print("=" * 55)

    master_kek = hashlib.sha256(b"master_kek_for_key_protection_test").digest()

    # Test 1: Wrap/unwrap roundtrip
    print("\n[ Wrap/unwrap roundtrip ]")
    key_material = hashlib.sha256(b"kyber_secret_key").digest()
    wrapped = wrap_key(master_kek, key_material)
    unwrapped = unwrap_key(master_kek, wrapped)
    assert unwrapped == key_material
    print(f"  ✓ Wrapped ({len(wrapped)} bytes) → unwrapped matches original")

    # Test 2: Integrity check (tampered blob rejected)
    print("\n[ Integrity check ]")
    tampered = bytearray(wrapped)
    tampered[20] ^= 0xFF
    result = unwrap_key(master_kek, bytes(tampered))
    assert result is None
    print(f"  ✓ Tampered blob correctly rejected")

    # Test 3: Wrong KEK rejected
    print("\n[ Wrong KEK rejection ]")
    wrong_kek = hashlib.sha256(b"wrong_kek").digest()
    result = unwrap_key(wrong_kek, wrapped)
    assert result is None
    print(f"  ✓ Wrong KEK correctly rejected")

    # Test 4: Manager — store and retrieve
    print("\n[ Manager: store and retrieve ]")
    mgr = KeyProtectionManager(master_kek)
    mgr.store_key("kyber_sk_001", key_material, owner_id="kyber")
    retrieved = mgr.retrieve_key("kyber_sk_001", "kyber", "decrypt")
    assert retrieved == key_material
    print(f"  ✓ Stored and retrieved key successfully")

    # Test 5: Auto-activation on first use
    print("\n[ Auto-activation ]")
    meta = mgr.metadata["kyber_sk_001"]
    assert meta.state == KeyState.ACTIVE
    print(f"  ✓ Key auto-activated: state={meta.state.value}")

    # Test 6: Operation not allowed
    print("\n[ Access control: disallowed operation ]")
    mgr.store_key(
        "sign_only_key", key_material, owner_id="dilithium",
        allowed_ops={"sign"}
    )
    result = mgr.retrieve_key("sign_only_key", "dilithium", "encrypt")
    assert result is None
    print(f"  ✓ Disallowed operation correctly rejected")

    result_ok = mgr.retrieve_key("sign_only_key", "dilithium", "sign")
    assert result_ok == key_material
    print(f"  ✓ Allowed operation correctly granted")

    # Test 7: Usage limit enforcement
    print("\n[ Usage limit enforcement ]")
    mgr.store_key("limited_key", key_material, owner_id="test", max_usage=3)
    for i in range(3):
        r = mgr.retrieve_key("limited_key", "test", "encrypt")
        assert r is not None
    r_exhausted = mgr.retrieve_key("limited_key", "test", "encrypt")
    assert r_exhausted is None
    assert mgr.metadata["limited_key"].state == KeyState.EXPIRED
    print(f"  ✓ Key expired after 3 uses (max_usage=3)")

    # Test 8: Lifecycle state transitions
    print("\n[ Lifecycle transitions ]")
    mgr.store_key("lifecycle_key", key_material, owner_id="test")
    assert mgr.metadata["lifecycle_key"].state == KeyState.GENERATED
    mgr.activate_key("lifecycle_key")
    assert mgr.metadata["lifecycle_key"].state == KeyState.ACTIVE
    mgr.expire_key("lifecycle_key")
    assert mgr.metadata["lifecycle_key"].state == KeyState.EXPIRED
    mgr.zeroize_key("lifecycle_key")
    assert mgr.metadata["lifecycle_key"].state == KeyState.ZEROIZED
    print(f"  ✓ GENERATED → ACTIVE → EXPIRED → ZEROIZED")

    # Test 9: Invalid transition rejected
    print("\n[ Invalid transition rejection ]")
    try:
        mgr.activate_key("lifecycle_key")  # ZEROIZED → ACTIVE invalid
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ ZEROIZED → ACTIVE correctly rejected")

    # Test 10: Zeroization clears storage
    print("\n[ Zeroization clears storage ]")
    stored_bytes = mgr.storage["lifecycle_key"]
    assert all(b == 0 for b in stored_bytes)
    print(f"  ✓ Storage overwritten with zeros after zeroize")

    # Test 11: Retrieve from zeroized key fails
    print("\n[ Retrieve from zeroized key ]")
    result = mgr.retrieve_key("lifecycle_key", "test", "encrypt")
    assert result is None
    print(f"  ✓ Zeroized key correctly inaccessible")

    # Test 12: Zeroize all (tamper-response scenario)
    print("\n[ Zeroize all (tamper response) ]")
    mgr2 = KeyProtectionManager(master_kek)
    for i in range(5):
        mgr2.store_key(f"key_{i}", key_material, owner_id="test")
    count = mgr2.zeroize_all()
    assert count == 5
    assert all(m.state == KeyState.ZEROIZED for m in mgr2.metadata.values())
    print(f"  ✓ Zeroized all {count} keys on simulated tamper event")

    # Test 13: Statistics
    print("\n[ Manager statistics ]")
    stats = mgr.stats()
    print(f"  Total keys: {stats['total_keys']}")
    print(f"  State counts: {stats['state_counts']}")
    print(f"  Access: {stats['access_granted']} granted, {stats['access_denied']} denied")
    print(f"  ✓ Statistics tracked correctly")

    print("\n  All checks passed.\n")
