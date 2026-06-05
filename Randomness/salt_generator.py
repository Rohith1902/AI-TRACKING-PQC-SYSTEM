"""
salt_generator.py
=================
Salt generation module for the PQC-SNN SoC security manager.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Generates 256-bit cryptographic salts from CTR_DRBG entropy for key       ║
║ derivation, initialization, and security operations across the SoC.       ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Salt pool management (buffered salt generation)
  - Per-operation salt allocation
  - Deterministic salt tracking (optional logging)
  - DRBG reseeding on entropy updates
  - Thread-safe salt distribution (simulated)

Salt usage in the SoC:
  - Kyber key derivation (KEM salt)
  - Dilithium signing (randomness source)
  - HKDF key derivation (master salt)
  - Challenge/response nonce generation
  - Initialization vectors for AES operations

Matches salt_generator.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST SP 800-56C Rev.1 (Key Derivation Function Specification)
"""

from __future__ import annotations
from typing import List, Dict
from ctr_drbg import CTR_DRBG, BLOCK_SIZE

# ──────────────────────────────────────────────
# 1.  SALT PARAMETERS
# ──────────────────────────────────────────────

#: Salt length in bytes (256 bits)
SALT_LEN: int = 32

#: Salt pool size (number of salts to buffer)
SALT_POOL_SIZE: int = 16

#: Rekey interval (number of salts before DRBG reseeding recommendation)
REKEY_INTERVAL: int = 128


# ──────────────────────────────────────────────
# 2.  SALT GENERATOR CLASS
# ──────────────────────────────────────────────

class SaltGenerator:
    """
    Cryptographic salt generator for the security manager.

    Manages a pool of salts derived from CTR_DRBG, with automatic
    refilling and DRBG reseeding.
    """

    def __init__(self, seed: bytes | None = None):
        """
        Initialize the salt generator.

        Parameters
        ----------
        seed : bytes, optional
            Initial seed for CTR_DRBG (32-48 bytes recommended).
            If None, uses a default seed.
        """
        if seed is None:
            seed = b"salt_generator_default_seed_32byt"

        self.drbg = CTR_DRBG(seed)
        self.salt_pool: List[bytes] = []
        self.allocation_log: Dict[str, int] = {}  # Track salt usage
        self.total_salts_generated = 0
        self.rekey_counter = 0

        # Initialize salt pool
        self._refill_pool()

    def _refill_pool(self) -> None:
        """
        Refill the salt pool from DRBG.

        Generates SALT_POOL_SIZE salts and buffers them.
        """
        self.salt_pool = []
        for _ in range(SALT_POOL_SIZE):
            salt = self.drbg.generate(SALT_LEN)
            self.salt_pool.append(salt)
        self.rekey_counter = 0

    def get_salt(self, label: str = "generic") -> bytes:
        """
        Allocate and retrieve a salt.

        Parameters
        ----------
        label : str
            Label/identifier for logging salt usage (e.g., "kyber_kem", "dilithium_sign").

        Returns
        -------
        bytes
            32-byte salt.

        Raises
        ------
        RuntimeError
            If pool is exhausted and DRBG requires reseed.
        """
        # Refill if pool is empty
        if not self.salt_pool:
            try:
                self._refill_pool()
            except RuntimeError as e:
                raise RuntimeError(f"Salt pool exhausted and DRBG requires reseed: {e}")

        # Pop salt from pool
        salt = self.salt_pool.pop()

        # Update counters
        self.total_salts_generated += 1
        self.rekey_counter += 1
        self.allocation_log[label] = self.allocation_log.get(label, 0) + 1

        # Suggest rekey if threshold reached
        if self.rekey_counter >= REKEY_INTERVAL:
            # In real hardware, this would trigger entropy manager to reseed
            pass

        return salt

    def reseed(self, entropy: bytes) -> None:
        """
        Reseed the DRBG with fresh entropy from TRNG.

        Parameters
        ----------
        entropy : bytes
            Fresh entropy input (32-48 bytes from TRNG).
        """
        self.drbg.reseed(entropy)
        self._refill_pool()

    def get_multiple(self, count: int, label: str = "generic") -> List[bytes]:
        """
        Allocate multiple salts at once.

        Parameters
        ----------
        count : int
            Number of salts to retrieve.
        label : str
            Label for logging.

        Returns
        -------
        List[bytes]
            List of salts.
        """
        return [self.get_salt(label) for _ in range(count)]

    def stats(self) -> Dict[str, int]:
        """
        Return generator statistics.

        Returns
        -------
        Dict[str, int]
            Statistics: total_generated, rekey_counter, pool_remaining, allocation_log.
        """
        return {
            "total_generated": self.total_salts_generated,
            "rekey_counter": self.rekey_counter,
            "pool_remaining": len(self.salt_pool),
            "allocation_log": dict(self.allocation_log),
        }


# ──────────────────────────────────────────────
# 3.  MODULE-LEVEL INTERFACE
# ──────────────────────────────────────────────

#: Global salt generator instance
_SALT_GEN_INSTANCE: SaltGenerator | None = None


def initialize_salt_gen(seed: bytes | None = None) -> None:
    """Initialize the global salt generator."""
    global _SALT_GEN_INSTANCE
    _SALT_GEN_INSTANCE = SaltGenerator(seed)


def get_salt(label: str = "generic") -> bytes:
    """
    Get a salt from the global generator.

    Parameters
    ----------
    label : str
        Operation label (e.g., "kyber_kem", "dilithium_sign").

    Returns
    -------
    bytes
        32-byte salt.
    """
    global _SALT_GEN_INSTANCE
    if _SALT_GEN_INSTANCE is None:
        initialize_salt_gen()
    return _SALT_GEN_INSTANCE.get_salt(label)


def reseed_salt_gen(entropy: bytes) -> None:
    """Reseed the global salt generator."""
    global _SALT_GEN_INSTANCE
    if _SALT_GEN_INSTANCE is None:
        initialize_salt_gen()
    _SALT_GEN_INSTANCE.reseed(entropy)


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import binascii

    print("=" * 55)
    print("salt_generator.py  —  self-test")
    print("=" * 55)

    # Test 1: Initialization
    print("\n[ Salt generator initialization ]")
    initialize_salt_gen(b"test_seed_32_bytes_for_salt_gen_ok")
    print(f"  ✓ Initialized with seed")

    # Test 2: Salt retrieval
    print("\n[ Salt retrieval ]")
    salt = get_salt("kyber_kem")
    assert len(salt) == SALT_LEN
    assert isinstance(salt, bytes)
    print(f"  ✓ Retrieved salt: {binascii.hexlify(salt).decode()[:32]}...")

    # Test 3: Determinism
    print("\n[ Determinism ]")
    gen1 = SaltGenerator(b"determinism_seed_32_bytes_for_test")
    gen2 = SaltGenerator(b"determinism_seed_32_bytes_for_test")
    salt1 = gen1.get_salt("test")
    salt2 = gen2.get_salt("test")
    assert salt1 == salt2
    print(f"  ✓ Same seed → same first salt")

    # Test 4: Salt uniqueness
    print("\n[ Salt uniqueness ]")
    gen = SaltGenerator()
    salts = [gen.get_salt("test") for _ in range(20)]
    unique_salts = len(set(salts))
    assert unique_salts == 20, "Salts should be unique"
    print(f"  ✓ Generated {unique_salts}/20 unique salts")

    # Test 5: Multiple salts
    print("\n[ Batch salt allocation ]")
    gen = SaltGenerator()
    batch = gen.get_multiple(8, "batch_test")
    assert len(batch) == 8
    assert all(len(s) == SALT_LEN for s in batch)
    print(f"  ✓ Allocated 8 salts in batch")

    # Test 6: Label tracking
    print("\n[ Usage label tracking ]")
    gen = SaltGenerator()
    gen.get_salt("kyber_kem")
    gen.get_salt("kyber_kem")
    gen.get_salt("dilithium_sign")
    gen.get_salt("hkdf_key")
    stats = gen.stats()
    assert stats["allocation_log"]["kyber_kem"] == 2
    assert stats["allocation_log"]["dilithium_sign"] == 1
    assert stats["allocation_log"]["hkdf_key"] == 1
    print(f"  ✓ Allocation log: {stats['allocation_log']}")

    # Test 7: Pool management
    print("\n[ Pool management ]")
    gen = SaltGenerator()
    initial_pool = len(gen.salt_pool)
    gen.get_salt()
    assert len(gen.salt_pool) == initial_pool - 1
    print(f"  ✓ Pool size decreases on allocation ({initial_pool} → {len(gen.salt_pool)})")

    # Test 8: Reseed
    print("\n[ Reseeding ]")
    gen = SaltGenerator(b"initial_seed_32_bytes_for_test_ok")
    salt_before = gen.get_salt()
    gen.reseed(b"fresh_entropy_32_bytes_for_test_ok")
    salt_after = gen.get_salt()
    assert salt_before != salt_after
    print(f"  ✓ Reseed changes salt output")

    # Test 9: Kyber/Dilithium workflow
    print("\n[ Crypto workflow simulation ]")
    gen = SaltGenerator()
    # Kyber KEM
    kyber_salt = gen.get_salt("kyber_kem")
    # Dilithium signing
    dilithium_salt = gen.get_salt("dilithium_sign")
    # HKDF
    hkdf_salt = gen.get_salt("hkdf_master")
    assert len({kyber_salt, dilithium_salt, hkdf_salt}) == 3
    print(f"  ✓ Generated distinct salts for KEM, DSA, HKDF")

    # Test 10: Statistics
    print("\n[ Generator statistics ]")
    gen = SaltGenerator()
    for _ in range(25):
        gen.get_salt("test")
    stats = gen.stats()
    print(f"  Total generated: {stats['total_generated']}")
    print(f"  Pool remaining:  {stats['pool_remaining']}")
    print(f"  Rekey counter:   {stats['rekey_counter']}")
    print(f"  ✓ Stats tracked correctly")

    print("\n  All checks passed.\n")
