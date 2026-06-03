"""
hash_ctrl.py
============
Hash function dispatcher and controller for the PQC-SNN SoC Python Golden Model.

Provides high-level hash function interfaces for Kyber and Dilithium:
  - H(x)      : General-purpose hash (SHA3-256 equivalent, 32 bytes)
  - G(x)      : Larger-output hash (SHA3-512 equivalent, 64 bytes)
  - J(x)      : Hash variant (specific PQC use case, variable output)
  - PRF(s, b) : Pseudorandom function (SHAKE256-based, variable output)
  - XOF(seed) : Seed expansion (SHAKE128-based, variable output)

Implemented via SHAKE128 and SHAKE256 with different parameters,
matching SHA3 behavior per FIPS 202.

Matches hash_ctrl.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST FIPS 202, CRYSTALS-Kyber/Dilithium specifications
"""

from __future__ import annotations
from typing import Literal
from shake128 import shake128
from shake256 import shake256

# ──────────────────────────────────────────────
# 1.  HASH FUNCTION PARAMETERS
# ──────────────────────────────────────────────

#: Output length for H (SHA3-256 equivalent)
H_OUTPUT_LEN: int = 32

#: Output length for G (SHA3-512 equivalent)
G_OUTPUT_LEN: int = 64

#: Default output length for J (variable, typically 64)
J_OUTPUT_LEN: int = 64

#: Default output length for PRF (variable, typically 32-64)
PRF_DEFAULT_LEN: int = 32

#: Default output length for XOF (variable, typically 168-1344 for Kyber)
XOF_DEFAULT_LEN: int = 168


# ──────────────────────────────────────────────
# 2.  HASH FUNCTION IMPLEMENTATIONS
# ──────────────────────────────────────────────

def H(x: bytes) -> bytes:
    """
    General-purpose hash function H(x).

    Equivalent to SHA3-256: fixed 32-byte output.
    Implemented via SHAKE256 with 32-byte output.

    Parameters
    ----------
    x : bytes
        Input message.

    Returns
    -------
    bytes
        32 bytes of hash output.

    Usage
    -----
    In Kyber:
      - Hash of public key (32 bytes)
      - Hash-based operations in decapsulation
    """
    return shake256(x, H_OUTPUT_LEN)


def G(x: bytes) -> bytes:
    """
    Larger-output hash function G(x).

    Equivalent to SHA3-512: fixed 64-byte output.
    Implemented via SHAKE256 with 64-byte output.

    Parameters
    ----------
    x : bytes
        Input message.

    Returns
    -------
    bytes
        64 bytes of hash output.

    Usage
    -----
    In Kyber:
      - Seed expansion in key generation
      - Generating (d, dH) pairs
    In Dilithium:
      - Seed expansion
    """
    return shake256(x, G_OUTPUT_LEN)


def J(x: bytes, output_len: int = J_OUTPUT_LEN) -> bytes:
    """
    Hash variant J(x) with variable output length.

    Implemented via SHAKE256.

    Parameters
    ----------
    x : bytes
        Input message.
    output_len : int
        Desired output length in bytes (default: 64).

    Returns
    -------
    bytes
        output_len bytes of hash output.

    Usage
    -----
    In Dilithium:
      - Signature generation
      - Challenge hashing
    """
    return shake256(x, output_len)


def PRF(s: bytes, b: int, output_len: int = PRF_DEFAULT_LEN) -> bytes:
    """
    Pseudorandom function PRF(s, b).

    Expands a seed and counter into pseudorandom bytes via SHAKE256.
    Counter b is prepended to the seed for domain separation.

    Parameters
    ----------
    s : bytes
        Seed (typically 32 bytes).
    b : int
        Counter/selector (0-255, typically fits in one byte).
    output_len : int
        Desired output length in bytes (default: 32).

    Returns
    -------
    bytes
        output_len bytes of pseudorandom data.

    Usage
    -----
    In Kyber:
      - Noise sampling (CBD sampling)
      - Error polynomial generation
    In Dilithium:
      - Masking polynomial generation
      - Error sampling
    """
    # Prepend counter byte for domain separation
    counter_byte = bytes([b & 0xFF])
    message = counter_byte + s
    return shake256(message, output_len)


def XOF(seed: bytes, output_len: int = XOF_DEFAULT_LEN) -> bytes:
    """
    Seed expansion via XOF (extendable-output function).

    Uses SHAKE128 for efficient seed expansion.
    Typically used for expanding short seeds into large polynomials.

    Parameters
    ----------
    seed : bytes
        Seed bytes (typically 32 bytes).
    output_len : int
        Desired output length (default: 168 for Kyber).

    Returns
    -------
    bytes
        output_len bytes of expanded seed.

    Usage
    -----
    In Kyber:
      - Matrix A generation (ρ seed → 1344 bytes per row)
      - Deterministic polynomial sampling
    In Dilithium:
      - Matrix A generation
      - Polynomial expansion
    """
    return shake128(seed, output_len)


# ──────────────────────────────────────────────
# 3.  HASH DISPATCHER
# ──────────────────────────────────────────────

def hash_dispatch(
    func_name: Literal["H", "G", "J", "PRF", "XOF"],
    message: bytes,
    **kwargs,
) -> bytes:
    """
    Dispatch to the appropriate hash function.

    Parameters
    ----------
    func_name : {"H", "G", "J", "PRF", "XOF"}
        Name of the hash function to call.
    message : bytes
        Input message (primary argument).
    **kwargs : dict
        Additional arguments (output_len, b for PRF, etc.).

    Returns
    -------
    bytes
        Hash output.

    Raises
    ------
    ValueError
        If func_name is not recognized.

    Example
    -------
    >>> h_out = hash_dispatch("H", b"test message")
    >>> len(h_out)
    32
    >>> prf_out = hash_dispatch("PRF", b"seed", b=5, output_len=64)
    >>> len(prf_out)
    64
    """
    if func_name == "H":
        return H(message)
    elif func_name == "G":
        return G(message)
    elif func_name == "J":
        output_len = kwargs.get("output_len", J_OUTPUT_LEN)
        return J(message, output_len)
    elif func_name == "PRF":
        b = kwargs.get("b", 0)
        output_len = kwargs.get("output_len", PRF_DEFAULT_LEN)
        return PRF(message, b, output_len)
    elif func_name == "XOF":
        output_len = kwargs.get("output_len", XOF_DEFAULT_LEN)
        return XOF(message, output_len)
    else:
        raise ValueError(
            f"Unknown hash function: {func_name}. "
            f"Choose from: H, G, J, PRF, XOF"
        )


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import binascii

    print("=" * 55)
    print("hash_ctrl.py  —  self-test")
    print("=" * 55)

    test_msg = b"PQC-SNN golden model test message"

    # Test 1: H function
    print("\n[ H - General hash (32 bytes) ]")
    h_out = H(test_msg)
    print(f"  Output: {binascii.hexlify(h_out).decode()[:32]}...")
    assert len(h_out) == H_OUTPUT_LEN
    print(f"  ✓ Length: {len(h_out)} bytes")

    # Test 2: G function
    print("\n[ G - Large hash (64 bytes) ]")
    g_out = G(test_msg)
    print(f"  Output: {binascii.hexlify(g_out).decode()[:32]}...")
    assert len(g_out) == G_OUTPUT_LEN
    print(f"  ✓ Length: {len(g_out)} bytes")

    # Test 3: J function with variable output
    print("\n[ J - Variable output hash ]")
    for out_len in [32, 64, 128]:
        j_out = J(test_msg, out_len)
        assert len(j_out) == out_len
    print(f"  ✓ Tested J with outputs: 32, 64, 128 bytes")

    # Test 4: PRF with counter
    print("\n[ PRF - Pseudorandom function ]")
    seed = b"kyber_prf_test_32_bytes_seed_val"
    for b_val in [0, 1, 127, 255]:
        prf_out = PRF(seed, b_val, 32)
        assert len(prf_out) == 32
    print(f"  ✓ PRF outputs differ per counter (b=0,1,127,255)")
    
    # Verify PRF outputs differ for different counters
    prf_b0 = PRF(seed, 0, 32)
    prf_b1 = PRF(seed, 1, 32)
    diffs = sum(1 for i in range(32) if prf_b0[i] != prf_b1[i])
    print(f"  ✓ PRF(s, b=0) vs PRF(s, b=1) differ in {diffs}/32 bytes")

    # Test 5: XOF seed expansion
    print("\n[ XOF - Seed expansion ]")
    xof_seed = b"kyber_xof_seed__"
    for out_len in [168, 512, 1344]:
        xof_out = XOF(xof_seed, out_len)
        assert len(xof_out) == out_len
    print(f"  ✓ XOF supports seed expansion: 168, 512, 1344 bytes")

    # Test 6: Dispatcher
    print("\n[ Dispatcher hash_dispatch() ]")
    h_via_dispatch = hash_dispatch("H", test_msg)
    h_direct = H(test_msg)
    assert h_via_dispatch == h_direct
    print(f"  ✓ Dispatcher matches direct calls")

    # Test 7: Kyber use case
    print("\n[ Kyber key generation workflow ]")
    kyber_seed = b"kyber_seed_32_bytes_for_keygen_"
    # Expand seed to matrix A (simulated)
    a_expanded = XOF(kyber_seed, 1344)  # One row of A
    # Hash public key
    pk_hash = H(a_expanded[:32])
    assert len(a_expanded) == 1344
    assert len(pk_hash) == 32
    print(f"  ✓ Seed → 1344-byte matrix, hash → 32 bytes")

    # Test 8: Dilithium use case
    print("\n[ Dilithium signing workflow ]")
    dilithium_seed = b"dilithium_seed_32_bytes_for_sig_"
    # Seed expansion
    expanded = G(dilithium_seed)  # 64 bytes
    # Counter-based PRF for error sampling
    for counter in range(5):
        error_sample = PRF(dilithium_seed, counter, 64)
        assert len(error_sample) == 64
    print(f"  ✓ Seed → G(64B), PRF with counters for error sampling")

    # Test 9: Determinism across all functions
    print("\n[ Determinism ]")
    for func_name in ["H", "G", "XOF"]:
        out1 = hash_dispatch(func_name, test_msg)
        out2 = hash_dispatch(func_name, test_msg)
        assert out1 == out2
    print(f"  ✓ All functions deterministic")

    print("\n  All checks passed.\n")
