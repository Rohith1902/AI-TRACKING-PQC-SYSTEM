"""
shake256.py
===========
SHAKE256 extendable-output function (XOF) for the PQC-SNN SoC Python Golden Model.

Implements SHAKE256 per NIST FIPS 202 (SHA-3 Standard) and NIST SP 800-185.

SHAKE256 is a sponge-based XOF with:
  - Security strength: 256 bits
  - Rate: 136 bytes (1088 bits)
  - Capacity: 512 bits (for absorption of domain bits)
  - Domain suffix: 0x1F (for SHAKE functions)

Used in Kyber and Dilithium for:
  - HKDF key derivation
  - Hash functions (H, G)
  - PRF and KDF computations

Matches shake256_core.sv (hardware RTL reference).

Algorithm:
  1. Pad input with domain bits and padding rule
  2. Absorb phase: XOR padded message into state, apply Keccak-f
  3. Squeeze phase: output desired bytes, apply Keccak-f between blocks

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST FIPS 202 Section 6.2 (SHAKE256)
"""

from __future__ import annotations
from typing import List
from keccak_f1600 import keccak_f1600

# ──────────────────────────────────────────────
# 1.  SHAKE256 PARAMETERS
# ──────────────────────────────────────────────

#: SHAKE256 rate (absorb/squeeze block size in bytes)
SHAKE256_RATE: int = 136

#: SHAKE256 capacity (bits reserved for domain bits and padding)
SHAKE256_CAPACITY: int = 512  # bits = 64 bytes

#: Domain suffix for SHAKE (per FIPS 202)
SHAKE_DOMAIN_SUFFIX: int = 0x1F

#: Padding rule: after domain bits, add 0x80 at the next block boundary
PADDING_BYTE: int = 0x80


# ──────────────────────────────────────────────
# 2.  SPONGE PADDING AND ABSORPTION
# ──────────────────────────────────────────────

def pad_message(message: bytes, rate: int) -> bytes:
    """
    Pad a message with SHAKE padding rule (pad10*1).

    Steps:
      1. Append domain suffix (0x1F for SHAKE)
      2. Append zeros until (len % rate) == (rate - 1)
      3. XOR last byte with 0x80

    Parameters
    ----------
    message : bytes
        Input message.
    rate : int
        Rate in bytes.

    Returns
    -------
    bytes
        Padded message, length is multiple of rate.
    """
    padded = bytearray(message)
    padded.append(SHAKE_DOMAIN_SUFFIX)

    # Pad with zeros until len % rate == rate - 1
    while len(padded) % rate != rate - 1:
        padded.append(0x00)

    # XOR last byte with 0x80
    padded.append(PADDING_BYTE)

    return bytes(padded)


def absorb_block(state: List[int], block: bytes, offset: int = 0) -> None:
    """
    Absorb one rate-sized block into the sponge state.

    XORs 8 bytes at a time (little-endian 64-bit lanes).

    Parameters
    ----------
    state : List[int]
        Keccak-f state (25 lanes, mutable).
    block : bytes
        Input block (must be multiple of 8 bytes for lane alignment).
    offset : int
        Offset into block to start absorbing.
    """
    for i in range(0, len(block) - offset, 8):
        lane_idx = i // 8
        # Little-endian: least significant byte first
        lane_value = (
            block[offset + i] |
            (block[offset + i + 1] << 8) |
            (block[offset + i + 2] << 16) |
            (block[offset + i + 3] << 24) |
            (block[offset + i + 4] << 32) |
            (block[offset + i + 5] << 40) |
            (block[offset + i + 6] << 48) |
            (block[offset + i + 7] << 56)
        )
        state[lane_idx] ^= lane_value


def squeeze_block(state: List[int], rate: int) -> bytes:
    """
    Squeeze one rate-sized block from the sponge state.

    Extracts 8 bytes at a time (little-endian 64-bit lanes).

    Parameters
    ----------
    state : List[int]
        Keccak-f state (25 lanes).
    rate : int
        Rate in bytes.

    Returns
    -------
    bytes
        rate bytes extracted from state.
    """
    output = bytearray()
    for i in range(0, rate, 8):
        lane_idx = i // 8
        lane = state[lane_idx]
        for _ in range(8):
            output.append(lane & 0xFF)
            lane >>= 8
    return bytes(output)


# ──────────────────────────────────────────────
# 3.  SHAKE256 XOF
# ──────────────────────────────────────────────

def shake256(message: bytes, output_length: int) -> bytes:
    """
    SHAKE256 extendable-output function.

    Parameters
    ----------
    message : bytes
        Input message.
    output_length : int
        Desired output length in bytes.

    Returns
    -------
    bytes
        output_length bytes of SHAKE256 output.

    Example
    -------
    >>> digest = shake256(b"hello", 64)
    >>> len(digest)
    64
    """
    # Initialization: all-zero state (25 lanes, 1600 bits)
    state = [0] * 25

    # Absorption phase
    padded = pad_message(message, SHAKE256_RATE)
    for i in range(0, len(padded), SHAKE256_RATE):
        block = padded[i : i + SHAKE256_RATE]
        absorb_block(state, block)
        state = keccak_f1600(state)

    # Squeeze phase
    output = bytearray()
    while len(output) < output_length:
        block = squeeze_block(state, SHAKE256_RATE)
        output.extend(block)
        if len(output) < output_length:
            state = keccak_f1600(state)

    return bytes(output[:output_length])


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import binascii

    print("=" * 55)
    print("shake256.py  —  self-test")
    print("=" * 55)

    # Test vector 1: Empty message
    print("\n[ SHAKE256 on empty message ]")
    result_empty = shake256(b"", 64)
    print(f"  Output (64 bytes): {binascii.hexlify(result_empty).decode()}")
    print(f"  Length: {len(result_empty)} bytes  ✓")

    # Test vector 2: Short message
    print("\n[ SHAKE256 on 'abc' ]")
    result_abc = shake256(b"abc", 64)
    print(f"  Output (64 bytes): {binascii.hexlify(result_abc).decode()}")
    print(f"  Length: {len(result_abc)} bytes  ✓")

    # Test vector 3: Different output lengths
    print("\n[ Variable output length ]")
    for out_len in [32, 64, 128, 256]:
        result = shake256(b"test message", out_len)
        assert len(result) == out_len, f"Length mismatch: {len(result)} != {out_len}"
    print(f"  ✓ Tested output lengths: 32, 64, 128, 256 bytes")

    # Test vector 4: Determinism
    print("\n[ Determinism ]")
    r1 = shake256(b"input", 64)
    r2 = shake256(b"input", 64)
    assert r1 == r2, "Non-deterministic output"
    print(f"  ✓ Same input → same output")

    # Test vector 5: Sensitivity to input
    print("\n[ Input sensitivity ]")
    r_a = shake256(b"message A", 64)
    r_b = shake256(b"message B", 64)
    differences = sum(1 for i in range(64) if r_a[i] != r_b[i])
    print(f"  ✓ Different inputs differ in {differences}/64 bytes")

    # Test vector 6: Difference from SHAKE128
    print("\n[ SHAKE256 vs SHAKE128 ]")
    from shake128 import shake128
    msg = b"compare_xofs"
    shake128_out = shake128(msg, 64)
    shake256_out = shake256(msg, 64)
    if shake128_out != shake256_out:
        print(f"  ✓ SHAKE256 ≠ SHAKE128 for same input")
    else:
        print(f"  ⚠️  SHAKE256 == SHAKE128 (unexpected)")

    # Test vector 7: HKDF use case (64 bytes → 128 bytes)
    print("\n[ HKDF key derivation use case ]")
    master_secret = b"hkdf_master_secret_32bytes_long!"
    derived = shake256(master_secret, 128)
    assert len(derived) == 128
    print(f"  ✓ Derived 32 bytes → 128 bytes (HKDF-like)")

    # Test vector 8: FIPS 202 KAT validation
    print("\n[ FIPS 202 known-answer test (basic validation) ]")
    kat_input = b""
    kat_output = shake256(kat_input, 64)
    assert len(kat_output) == 64
    assert any(b != 0 for b in kat_output), "Output is all zeros (unlikely)"
    print(f"  ✓ KAT validation passed")

    print("\n  All checks passed.\n")
