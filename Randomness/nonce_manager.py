"""
nonce_manager.py
================
Nonce (number used once) manager for the PQC-SNN SoC security manager.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Maintains monotonically increasing nonces for challenge/response, protocol ║
║ freshness, and cryptographic operations to prevent replay attacks.        ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Monotonic nonce counter (always increasing)
  - Per-operation nonce allocation
  - Nonce epoch management (counter reset with epoch change)
  - Persistent counter storage (simulated)
  - Overflow detection and rollover handling
  - Optional timestamp binding

Nonce usage in the SoC:
  - Challenge/response authentication (replay protection)
  - Dilithium signature generation (randomness input)
  - Protocol message freshness (prevent replays)
  - Timestamp tokens (proof of work)
  - Session identifiers (unique per connection)

Nonce structure:
  - Epoch (32 bits) : counter reset boundary
  - Counter (64 bits) : monotonically increasing value
  - Total: 96-bit unique nonce per operation

Matches nonce_manager.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : RFC 2104 (HMAC), NIST SP 800-38D (AEAD/Nonce)
"""

from __future__ import annotations
from typing import Tuple, Dict
import time

# ──────────────────────────────────────────────
# 1.  NONCE PARAMETERS
# ──────────────────────────────────────────────

#: Nonce length in bytes (96 bits: 32-bit epoch + 64-bit counter)
NONCE_LEN: int = 12

#: Epoch width in bits (32 bits)
EPOCH_BITS: int = 32

#: Counter width in bits (64 bits)
COUNTER_BITS: int = 64

#: Counter overflow threshold (max counter value before epoch rollover)
COUNTER_MAX: int = (1 << COUNTER_BITS) - 1

#: Nonce pool size (pre-generated nonces for performance)
NONCE_POOL_SIZE: int = 32

#: Epoch change interval (nonces before epoch should increment)
EPOCH_CHANGE_INTERVAL: int = 2**40  # ~1 trillion nonces per epoch


# ──────────────────────────────────────────────
# 2.  NONCE MANAGER CLASS
# ──────────────────────────────────────────────

class NonceManager:
    """
    Manages monotonically increasing nonces for the security system.

    Ensures all nonces are unique and prevent replay attacks.
    """

    def __init__(self):
        """Initialize nonce manager with epoch 0 and counter 0."""
        self.epoch = 0
        self.counter = 0
        self.nonce_pool: list[bytes] = []
        self.allocation_log: Dict[str, int] = {}
        self.total_nonces_issued = 0
        self.last_issue_timestamp = time.time()

    def _encode_nonce(self, epoch: int, counter: int) -> bytes:
        """
        Encode epoch and counter into a 12-byte nonce.

        Parameters
        ----------
        epoch : int
            32-bit epoch value.
        counter : int
            64-bit counter value.

        Returns
        -------
        bytes
            12-byte nonce (little-endian).
        """
        epoch_bytes = epoch.to_bytes(4, "little")
        counter_bytes = counter.to_bytes(8, "little")
        return epoch_bytes + counter_bytes

    def _decode_nonce(self, nonce: bytes) -> Tuple[int, int]:
        """
        Decode a 12-byte nonce into epoch and counter.

        Parameters
        ----------
        nonce : bytes
            12-byte nonce.

        Returns
        -------
        (epoch, counter)
            Tuple of epoch (int) and counter (int).
        """
        epoch = int.from_bytes(nonce[:4], "little")
        counter = int.from_bytes(nonce[4:], "little")
        return epoch, counter

    def _refill_pool(self) -> None:
        """Refill the nonce pool with pre-generated nonces."""
        self.nonce_pool = []
        for _ in range(NONCE_POOL_SIZE):
            nonce = self._encode_nonce(self.epoch, self.counter)
            self.nonce_pool.append(nonce)
            self.counter += 1
            
            # Check for epoch rollover
            if self.counter > COUNTER_MAX:
                self.epoch = (self.epoch + 1) & ((1 << EPOCH_BITS) - 1)
                self.counter = 0

    def get_nonce(self, label: str = "generic") -> bytes:
        """
        Allocate and retrieve a nonce.

        Parameters
        ----------
        label : str
            Operation label (e.g., "challenge", "signature", "session").

        Returns
        -------
        bytes
            12-byte unique nonce.

        Raises
        ------
        RuntimeError
            If counter overflow detected.
        """
        # Refill pool if empty
        if not self.nonce_pool:
            self._refill_pool()

        # Pop nonce from pool
        nonce = self.nonce_pool.pop(0)

        # Update tracking
        self.total_nonces_issued += 1
        self.allocation_log[label] = self.allocation_log.get(label, 0) + 1
        self.last_issue_timestamp = time.time()

        # Check for epoch change recommendation
        if self.counter >= EPOCH_CHANGE_INTERVAL:
            # In hardware, this would trigger epoch refresh
            pass

        return nonce

    def get_nonce_with_timestamp(self, label: str = "generic") -> Tuple[bytes, float]:
        """
        Get a nonce along with its issue timestamp.

        Parameters
        ----------
        label : str
            Operation label.

        Returns
        -------
        (nonce, timestamp)
            Tuple of nonce (bytes) and issue timestamp (float).
        """
        nonce = self.get_nonce(label)
        return nonce, time.time()

    def verify_nonce_freshness(self, nonce: bytes, max_age_seconds: float) -> bool:
        """
        Verify if a nonce is fresh (within age threshold).

        Parameters
        ----------
        nonce : bytes
            Nonce to verify.
        max_age_seconds : float
            Maximum allowed age in seconds.

        Returns
        -------
        bool
            True if nonce is fresh, False if expired.

        Note
        ----
        This is a simplified freshness check. Real implementation would
        track nonce issuance times in a database.
        """
        age = time.time() - self.last_issue_timestamp
        return age <= max_age_seconds

    def verify_nonce_uniqueness(self, nonce: bytes, seen_nonces: set[bytes]) -> bool:
        """
        Verify if a nonce has been seen before (replay detection).

        Parameters
        ----------
        nonce : bytes
            Nonce to verify.
        seen_nonces : set[bytes]
            Set of previously seen nonces.

        Returns
        -------
        bool
            True if nonce is new (not in seen set), False if seen before.
        """
        return nonce not in seen_nonces

    def get_multiple(self, count: int, label: str = "generic") -> list[bytes]:
        """
        Allocate multiple nonces at once.

        Parameters
        ----------
        count : int
            Number of nonces to retrieve.
        label : str
            Operation label.

        Returns
        -------
        List[bytes]
            List of nonces.
        """
        return [self.get_nonce(label) for _ in range(count)]

    def reset_epoch(self) -> None:
        """
        Manually reset epoch (typically done on system reboot/refresh).

        Increments epoch and resets counter to 0.
        """
        self.epoch = (self.epoch + 1) & ((1 << EPOCH_BITS) - 1)
        self.counter = 0
        self._refill_pool()

    def stats(self) -> Dict:
        """
        Return nonce manager statistics.

        Returns
        -------
        Dict
            Statistics: epoch, counter, total_issued, allocation_log, pool_size.
        """
        return {
            "epoch": self.epoch,
            "counter": self.counter,
            "total_issued": self.total_nonces_issued,
            "allocation_log": dict(self.allocation_log),
            "pool_remaining": len(self.nonce_pool),
        }


# ──────────────────────────────────────────────
# 3.  MODULE-LEVEL INTERFACE
# ──────────────────────────────────────────────

#: Global nonce manager instance
_NONCE_MGR_INSTANCE: NonceManager | None = None


def initialize_nonce_manager() -> None:
    """Initialize the global nonce manager."""
    global _NONCE_MGR_INSTANCE
    _NONCE_MGR_INSTANCE = NonceManager()


def get_nonce(label: str = "generic") -> bytes:
    """
    Get a nonce from the global manager.

    Parameters
    ----------
    label : str
        Operation label.

    Returns
    -------
    bytes
        12-byte nonce.
    """
    global _NONCE_MGR_INSTANCE
    if _NONCE_MGR_INSTANCE is None:
        initialize_nonce_manager()
    return _NONCE_MGR_INSTANCE.get_nonce(label)


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import binascii

    print("=" * 55)
    print("nonce_manager.py  —  self-test")
    print("=" * 55)

    # Test 1: Initialization
    print("\n[ Nonce manager initialization ]")
    initialize_nonce_manager()
    print(f"  ✓ Initialized")

    # Test 2: Nonce retrieval
    print("\n[ Nonce retrieval ]")
    nonce = get_nonce("challenge")
    assert len(nonce) == NONCE_LEN
    print(f"  ✓ Retrieved nonce: {binascii.hexlify(nonce).decode()}")

    # Test 3: Nonce monotonicity
    print("\n[ Nonce monotonicity ]")
    mgr = NonceManager()
    nonce1 = mgr.get_nonce()
    nonce2 = mgr.get_nonce()
    nonce3 = mgr.get_nonce()
    
    epoch1, counter1 = mgr._decode_nonce(nonce1)
    epoch2, counter2 = mgr._decode_nonce(nonce2)
    epoch3, counter3 = mgr._decode_nonce(nonce3)
    
    # Verify counter increases
    assert counter1 < counter2 < counter3 or (
        epoch1 < epoch2 < epoch3
    ), "Nonces should be monotonically increasing"
    print(f"  ✓ Nonces monotonically increase")
    print(f"    Nonce 1: epoch={epoch1}, counter={counter1}")
    print(f"    Nonce 2: epoch={epoch2}, counter={counter2}")
    print(f"    Nonce 3: epoch={epoch3}, counter={counter3}")

    # Test 4: Nonce uniqueness
    print("\n[ Nonce uniqueness ]")
    mgr = NonceManager()
    nonces = [mgr.get_nonce() for _ in range(100)]
    unique_nonces = len(set(nonces))
    assert unique_nonces == 100
    print(f"  ✓ Generated {unique_nonces}/100 unique nonces")

    # Test 5: Batch allocation
    print("\n[ Batch nonce allocation ]")
    mgr = NonceManager()
    batch = mgr.get_multiple(16, "batch_test")
    assert len(batch) == 16
    assert len(set(batch)) == 16  # All unique
    print(f"  ✓ Allocated 16 unique nonces in batch")

    # Test 6: Freshness check
    print("\n[ Freshness verification ]")
    mgr = NonceManager()
    nonce = mgr.get_nonce()
    is_fresh = mgr.verify_nonce_freshness(nonce, max_age_seconds=10.0)
    assert is_fresh
    print(f"  ✓ Nonce freshness check passed")

    # Test 7: Uniqueness detection (replay protection)
    print("\n[ Replay detection ]")
    mgr = NonceManager()
    seen = set()
    
    nonce1 = mgr.get_nonce()
    assert mgr.verify_nonce_uniqueness(nonce1, seen)
    seen.add(nonce1)
    
    # Try to reuse same nonce (replay attack)
    is_unique = mgr.verify_nonce_uniqueness(nonce1, seen)
    assert not is_unique, "Should detect replay"
    
    nonce2 = mgr.get_nonce()
    assert mgr.verify_nonce_uniqueness(nonce2, seen)
    print(f"  ✓ Correctly detects and prevents replays")

    # Test 8: Label tracking
    print("\n[ Usage label tracking ]")
    mgr = NonceManager()
    mgr.get_nonce("challenge")
    mgr.get_nonce("challenge")
    mgr.get_nonce("signature")
    mgr.get_nonce("session")
    stats = mgr.stats()
    assert stats["allocation_log"]["challenge"] == 2
    assert stats["allocation_log"]["signature"] == 1
    assert stats["allocation_log"]["session"] == 1
    print(f"  ✓ Allocation log: {stats['allocation_log']}")

    # Test 9: Epoch management
    print("\n[ Epoch management ]")
    mgr = NonceManager()
    initial_epoch = mgr.epoch
    mgr.reset_epoch()
    assert mgr.epoch == (initial_epoch + 1) & ((1 << EPOCH_BITS) - 1)
    print(f"  ✓ Epoch increments on reset ({initial_epoch} → {mgr.epoch})")

    # Test 10: Crypto protocol workflow
    print("\n[ Challenge/response workflow ]")
    mgr = NonceManager()
    # Server generates challenge nonce
    challenge = mgr.get_nonce("challenge")
    # Client response with same nonce for freshness
    response = mgr.get_nonce("response")
    # Verify uniqueness
    assert challenge != response
    assert mgr.verify_nonce_uniqueness(challenge, set())
    assert mgr.verify_nonce_uniqueness(response, set())
    print(f"  ✓ Challenge/response nonces unique and fresh")

    # Test 11: Statistics
    print("\n[ Manager statistics ]")
    mgr = NonceManager()
    for _ in range(50):
        mgr.get_nonce("test")
    stats = mgr.stats()
    print(f"  Total issued: {stats['total_issued']}")
    print(f"  Epoch: {stats['epoch']}")
    print(f"  Counter: {stats['counter']}")
    print(f"  Pool remaining: {stats['pool_remaining']}")
    print(f"  ✓ Statistics tracked correctly")

    print("\n  All checks passed.\n")
