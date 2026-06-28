"""
hkdf_engine.py
==============
HKDF key derivation engine for the PQC-SNN SoC security subsystem.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Implements RFC 5869 HKDF (Extract-then-Expand) for deriving cryptographic ║
║ keys from shared secrets, salts, and context info across PQC operations.  ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - HKDF-Extract  : PRK = HMAC-SHA256(salt, IKM)
  - HKDF-Expand   : OKM = T(1) || T(2) || ... || T(n)
  - HKDF          : Full Extract + Expand pipeline
  - Multi-key derivation (derive multiple keys from one IKM)
  - Context binding (info parameter for domain separation)
  - Key hierarchy support (master → session → operation keys)

Algorithm (RFC 5869):
  Extract:
    PRK = HMAC-Hash(salt, IKM)
    salt: optional (defaults to HashLen zeros)
    IKM : input keying material (e.g. Kyber shared secret)

  Expand:
    T(0) = empty
    T(i) = HMAC-Hash(PRK, T(i-1) || info || i)
    OKM  = T(1) || T(2) || ... truncated to L bytes

Usage in AEGIS-NEURO:
  - Kyber shared secret → session key via HKDF
  - Master key → operation-specific subkeys
  - PQC hybrid mode: combine classical + PQC secrets
  - SNN feature keys: derive keys for SNN threat context

Security:
  - HKDF is IND-CPA secure when IKM has sufficient entropy
  - info binding prevents cross-context key reuse
  - Max output: 255 × HashLen = 8160 bytes (SHA-256)

Matches hkdf_engine.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : RFC 5869 (HMAC-based Key Derivation Function)
"""

from __future__ import annotations
import hashlib
import hmac
from typing import Dict, List

# ──────────────────────────────────────────────
# 1.  HKDF PARAMETERS
# ──────────────────────────────────────────────

#: Hash function for HMAC (SHA-256)
HASH_ALGO: str = "sha256"

#: Hash output length (bytes)
HASH_LEN: int = 32

#: Maximum OKM length (255 × HashLen per RFC 5869)
MAX_OKM_LEN: int = 255 * HASH_LEN

#: Default salt (HashLen zero bytes per RFC 5869 §2.2)
DEFAULT_SALT: bytes = b"\x00" * HASH_LEN


# ──────────────────────────────────────────────
# 2.  HMAC PRIMITIVE
# ──────────────────────────────────────────────

def hmac_hash(key: bytes, data: bytes) -> bytes:
    """
    Compute HMAC-SHA256(key, data).

    Parameters
    ----------
    key : bytes
        HMAC key.
    data : bytes
        Input data.

    Returns
    -------
    bytes
        32-byte HMAC output.
    """
    return hmac.new(key, data, hashlib.sha256).digest()


# ──────────────────────────────────────────────
# 3.  HKDF EXTRACT
# ──────────────────────────────────────────────

def hkdf_extract(salt: bytes | None, ikm: bytes) -> bytes:
    """
    HKDF-Extract: derive pseudorandom key (PRK) from IKM.

    Parameters
    ----------
    salt : bytes or None
        Optional salt. If None, uses HashLen zero bytes.
    ikm : bytes
        Input keying material (e.g. Kyber shared secret).

    Returns
    -------
    bytes
        32-byte pseudorandom key (PRK).
    """
    if salt is None or len(salt) == 0:
        salt = DEFAULT_SALT

    return hmac_hash(salt, ikm)


# ──────────────────────────────────────────────
# 4.  HKDF EXPAND
# ──────────────────────────────────────────────

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    HKDF-Expand: derive output keying material (OKM) from PRK.

    Parameters
    ----------
    prk : bytes
        Pseudorandom key from HKDF-Extract (>= HashLen bytes).
    info : bytes
        Context and application-specific info (may be empty).
    length : int
        Desired output length in bytes (<= 255 * HashLen).

    Returns
    -------
    bytes
        `length` bytes of output keying material.

    Raises
    ------
    ValueError
        If length exceeds maximum allowed.
    """
    if length > MAX_OKM_LEN:
        raise ValueError(
            f"Requested length {length} exceeds max {MAX_OKM_LEN}"
        )

    t = b""
    okm = b""
    counter = 1

    while len(okm) < length:
        t = hmac_hash(prk, t + info + bytes([counter]))
        okm += t
        counter += 1

    return okm[:length]


# ──────────────────────────────────────────────
# 5.  HKDF (FULL PIPELINE)
# ──────────────────────────────────────────────

def hkdf(
    ikm: bytes,
    length: int,
    salt: bytes | None = None,
    info: bytes = b"",
) -> bytes:
    """
    Full HKDF: Extract then Expand.

    Parameters
    ----------
    ikm : bytes
        Input keying material.
    length : int
        Desired output length in bytes.
    salt : bytes or None
        Optional salt value.
    info : bytes
        Context info for domain separation.

    Returns
    -------
    bytes
        `length` bytes of derived key material.
    """
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)


# ──────────────────────────────────────────────
# 6.  HKDF ENGINE CLASS
# ──────────────────────────────────────────────

class HKDFEngine:
    """
    Stateful HKDF engine for key hierarchy management.

    Supports multi-key derivation and tracks key usage.
    """

    def __init__(self, master_ikm: bytes, master_salt: bytes | None = None):
        """
        Initialize HKDF engine with master IKM.

        Parameters
        ----------
        master_ikm : bytes
            Master input keying material.
        master_salt : bytes or None
            Master salt (optional).
        """
        self.master_prk = hkdf_extract(master_salt, master_ikm)
        self.derived_keys: Dict[str, bytes] = {}
        self.derivation_log: List[Dict] = []

    def derive(self, label: str, length: int = 32) -> bytes:
        """
        Derive a key bound to a label.

        Parameters
        ----------
        label : str
            Context label (e.g. "kyber_session", "dilithium_sign").
        length : int
            Output key length in bytes.

        Returns
        -------
        bytes
            Derived key material.
        """
        info = label.encode("utf-8")
        key = hkdf_expand(self.master_prk, info, length)

        self.derived_keys[label] = key
        self.derivation_log.append({"label": label, "length": length})

        return key

    def derive_multiple(self, labels: List[str], length: int = 32) -> Dict[str, bytes]:
        """
        Derive keys for multiple labels at once.

        Parameters
        ----------
        labels : List[str]
            List of context labels.
        length : int
            Output length per key.

        Returns
        -------
        Dict[str, bytes]
            Mapping label → derived key.
        """
        return {label: self.derive(label, length) for label in labels}

    def stats(self) -> Dict:
        """Return engine statistics."""
        return {
            "total_derived": len(self.derivation_log),
            "unique_labels": len(self.derived_keys),
            "labels": list(self.derived_keys.keys()),
        }


# ──────────────────────────────────────────────
# 7.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("hkdf_engine.py  —  self-test")
    print("=" * 55)

    # Test 1: HMAC primitive
    print("\n[ HMAC-SHA256 primitive ]")
    h = hmac_hash(b"key", b"data")
    assert len(h) == HASH_LEN
    assert h == hmac_hash(b"key", b"data")   # deterministic
    print(f"  ✓ HMAC-SHA256: {h.hex()[:32]}...")

    # Test 2: HKDF-Extract
    print("\n[ HKDF-Extract ]")
    ikm  = b"kyber_shared_secret_32_bytes_ok_"
    salt = b"aegis_neuro_salt_32_bytes_ok_000"
    prk  = hkdf_extract(salt, ikm)
    assert len(prk) == HASH_LEN
    print(f"  ✓ PRK ({len(prk)} bytes): {prk.hex()[:32]}...")

    # Test 3: Default salt (None)
    print("\n[ HKDF-Extract with no salt ]")
    prk2 = hkdf_extract(None, ikm)
    assert len(prk2) == HASH_LEN
    assert prk2 != prk    # different salt → different PRK
    print(f"  ✓ PRK (no salt): {prk2.hex()[:32]}...")

    # Test 4: HKDF-Expand
    print("\n[ HKDF-Expand ]")
    for length in [16, 32, 64, 128]:
        okm = hkdf_expand(prk, b"test_info", length)
        assert len(okm) == length
    print(f"  ✓ Expand: 16, 32, 64, 128 bytes all correct")

    # Test 5: Full HKDF pipeline
    print("\n[ Full HKDF pipeline ]")
    okm = hkdf(ikm, 32, salt, b"aegis_session_key")
    assert len(okm) == 32
    print(f"  ✓ OKM ({len(okm)} bytes): {okm.hex()[:32]}...")

    # Test 6: Determinism
    print("\n[ Determinism ]")
    okm1 = hkdf(ikm, 32, salt, b"info")
    okm2 = hkdf(ikm, 32, salt, b"info")
    assert okm1 == okm2
    print(f"  ✓ Same inputs → same OKM")

    # Test 7: Context separation via info
    print("\n[ Context separation ]")
    k_enc = hkdf(ikm, 32, salt, b"encryption_key")
    k_mac = hkdf(ikm, 32, salt, b"mac_key")
    k_iv  = hkdf(ikm, 16, salt, b"iv")
    assert k_enc != k_mac
    assert k_enc != k_iv[:32]
    print(f"  ✓ Different info → different keys")

    # Test 8: Kyber shared secret → session key
    print("\n[ Kyber shared secret → session key ]")
    kyber_ss = hashlib.sha256(b"kyber_shared_secret").digest()
    kem_salt = hashlib.sha256(b"kem_salt").digest()
    session_key = hkdf(kyber_ss, 32, kem_salt, b"kyber_kem_session")
    assert len(session_key) == 32
    print(f"  ✓ Kyber SS → 32-byte session key")

    # Test 9: Key hierarchy with HKDFEngine
    print("\n[ HKDFEngine key hierarchy ]")
    master_ikm = hashlib.sha256(b"master_secret").digest()
    engine = HKDFEngine(master_ikm, salt)

    keys = engine.derive_multiple(
        ["kyber_session", "dilithium_sign", "snn_feature", "aes_enc", "aes_mac"]
    )
    assert len(keys) == 5
    assert len(set(v for v in keys.values())) == 5  # all unique
    print(f"  ✓ Derived 5 unique keys from master IKM")

    # Test 10: Statistics
    print("\n[ Engine statistics ]")
    stats = engine.stats()
    assert stats["total_derived"] == 5
    print(f"  Labels: {stats['labels']}")
    print(f"  ✓ Statistics tracked correctly")

    # Test 11: PQC hybrid key derivation
    print("\n[ PQC hybrid key derivation ]")
    classical_secret = hashlib.sha256(b"ecdh_shared_secret").digest()
    pqc_secret       = hashlib.sha256(b"kyber_shared_secret").digest()
    hybrid_ikm       = classical_secret + pqc_secret
    hybrid_key       = hkdf(hybrid_ikm, 32, salt, b"hybrid_pqc_classical")
    assert len(hybrid_key) == 32
    print(f"  ✓ Hybrid classical+PQC key derived ({len(hybrid_key)} bytes)")

    print("\n  All checks passed.\n")

import hashlib
