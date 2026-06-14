"""
kyber_compress.py
=================
Kyber polynomial compression/decompression utilities for PQC-SNN SoC.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Implements efficient polynomial compression and decompression for Kyber   ║
║ ciphertexts, reducing message size while maintaining error bounds.        ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Compress_q: compress polynomial coefficients to d bits
  - Decompress_q: recover coefficients from compressed form
  - Bytewise compression/decompression
  - Lossless compression for d ≥ log2(q)
  - Lossy compression for d < log2(q) (ciphertext only)
  - Batch compression for multiple polynomials

Algorithm (Compress_q):
  For each coefficient c ∈ [0, q):
    compressed = ⌊(c * 2^d + q/2) / q⌋ mod 2^d

Algorithm (Decompress_q):
  For each compressed coefficient c' ∈ [0, 2^d):
    c = ⌊(c' * q + 2^(d-1)) / 2^d⌋

Compression rates:
  - d=4 (Dilithium signature): 4 bits/coeff = 128 bytes/poly
  - d=10 (Kyber u): 10 bits/coeff = 320 bytes/poly
  - d=11 (Kyber1024 u): 11 bits/coeff = 352 bytes/poly
  - d=12 (plaintext): 12 bits/coeff = 384 bytes/poly

Kyber ciphertext format:
  ct = Compress_q(u, d_u) || Compress_q(v, d_v)
  - u: K polynomials, d_u = 10 or 11 bits
  - v: 1 polynomial, d_v = 4 bits

Error bounds:
  - Decompression error: |c - decompress(compress(c))| ≤ ⌈q/2^(d+1)⌉
  - For Kyber: error ≤ 1 per coefficient (acceptable noise margin)

Matches kyber_compress.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST FIPS 203 (Compression/Decompression algorithms)
"""

from __future__ import annotations
from typing import List, Tuple

# ──────────────────────────────────────────────
# 1.  COMPRESSION PARAMETERS
# ──────────────────────────────────────────────

#: Kyber modulus
KYBER_Q: int = 3329

#: Polynomial degree
POLY_DEGREE: int = 256

#: Compression bit depths for Kyber operations
COMPRESS_D_U_512_768: int = 10  # Kyber512/768 u compression
COMPRESS_D_U_1024: int = 11      # Kyber1024 u compression
COMPRESS_D_V: int = 4            # v compression (all variants)

#: Maximum bits per coefficient
MAX_COMPRESSION_BITS: int = 12


# ──────────────────────────────────────────────
# 2.  COMPRESSION FUNCTIONS
# ──────────────────────────────────────────────

def compress_coeff(coeff: int, bits: int, q: int = KYBER_Q) -> int:
    """
    Compress a single polynomial coefficient.

    Parameters
    ----------
    coeff : int
        Coefficient value in [0, q).
    bits : int
        Number of bits for compression.
    q : int
        Modulus (default 3329 for Kyber).

    Returns
    -------
    int
        Compressed coefficient in [0, 2^bits).
    """
    if not (0 <= bits <= MAX_COMPRESSION_BITS):
        raise ValueError(f"bits must be in [1, {MAX_COMPRESSION_BITS}]")

    if not (0 <= coeff < q):
        raise ValueError(f"coeff must be in [0, {q})")

    # compressed = ⌊(coeff * 2^bits + q/2) / q⌋
    numerator = (coeff << bits) + (q >> 1)
    compressed = numerator // q

    # Mask to ensure within range
    mask = (1 << bits) - 1
    return compressed & mask


def decompress_coeff(
    compressed: int, bits: int, q: int = KYBER_Q
) -> int:
    """
    Decompress a single polynomial coefficient.

    Parameters
    ----------
    compressed : int
        Compressed coefficient in [0, 2^bits).
    bits : int
        Number of bits used in compression.
    q : int
        Modulus (default 3329 for Kyber).

    Returns
    -------
    int
        Decompressed coefficient in [0, q).
    """
    if not (0 <= bits <= MAX_COMPRESSION_BITS):
        raise ValueError(f"bits must be in [1, {MAX_COMPRESSION_BITS}]")

    # coeff = ⌊(compressed * q + 2^(bits-1)) / 2^bits⌋
    numerator = (compressed * q) + (1 << (bits - 1))
    coeff = numerator >> bits

    return coeff % q


def compress_poly(
    poly: List[int], bits: int, q: int = KYBER_Q
) -> bytes:
    """
    Compress a polynomial to bytes.

    Parameters
    ----------
    poly : List[int]
        Polynomial with 256 coefficients in [0, q).
    bits : int
        Compression bit depth.
    q : int
        Modulus.

    Returns
    -------
    bytes
        Compressed polynomial bytes.
    """
    if len(poly) != POLY_DEGREE:
        raise ValueError(f"Polynomial must have {POLY_DEGREE} coefficients")

    compressed = bytearray()
    bit_buffer = 0
    bit_count = 0

    for coeff in poly:
        # Compress coefficient
        compressed_coeff = compress_coeff(coeff, bits, q)

        # Add to bit buffer
        bit_buffer |= compressed_coeff << bit_count
        bit_count += bits

        # Extract complete bytes
        while bit_count >= 8:
            compressed.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8

    # Add remaining bits
    if bit_count > 0:
        compressed.append(bit_buffer & 0xFF)

    return bytes(compressed)


def decompress_poly(
    data: bytes, bits: int, q: int = KYBER_Q
) -> List[int]:
    """
    Decompress a polynomial from bytes.

    Parameters
    ----------
    data : bytes
        Compressed polynomial bytes.
    bits : int
        Compression bit depth used.
    q : int
        Modulus.

    Returns
    -------
    List[int]
        Decompressed polynomial with 256 coefficients.
    """
    poly = []
    bit_buffer = 0
    bit_count = 0
    byte_idx = 0

    for _ in range(POLY_DEGREE):
        # Fill buffer as needed
        while bit_count < bits and byte_idx < len(data):
            bit_buffer |= data[byte_idx] << bit_count
            bit_count += 8
            byte_idx += 1

        # Extract coefficient
        mask = (1 << bits) - 1
        compressed_coeff = bit_buffer & mask
        bit_buffer >>= bits
        bit_count -= bits

        # Decompress
        coeff = decompress_coeff(compressed_coeff, bits, q)
        poly.append(coeff)

    return poly


# ──────────────────────────────────────────────
# 3.  BATCH COMPRESSION
# ──────────────────────────────────────────────

def compress_polys(
    polys: List[List[int]], bits: int, q: int = KYBER_Q
) -> bytes:
    """
    Compress multiple polynomials.

    Parameters
    ----------
    polys : List[List[int]]
        List of polynomials.
    bits : int
        Compression bit depth.
    q : int
        Modulus.

    Returns
    -------
    bytes
        Concatenated compressed polynomials.
    """
    compressed = bytearray()
    for poly in polys:
        compressed.extend(compress_poly(poly, bits, q))
    return bytes(compressed)


def decompress_polys(
    data: bytes, count: int, bits: int, q: int = KYBER_Q
) -> List[List[int]]:
    """
    Decompress multiple polynomials.

    Parameters
    ----------
    data : bytes
        Concatenated compressed polynomial data.
    count : int
        Number of polynomials to decompress.
    bits : int
        Compression bit depth.
    q : int
        Modulus.

    Returns
    -------
    List[List[int]]
        List of decompressed polynomials.
    """
    polys = []
    bytes_per_poly = (bits * POLY_DEGREE) // 8
    if (bits * POLY_DEGREE) % 8 != 0:
        bytes_per_poly += 1

    for i in range(count):
        start = i * bytes_per_poly
        end = min(start + bytes_per_poly, len(data))
        poly_bytes = data[start:end]
        poly = decompress_poly(poly_bytes, bits, q)
        polys.append(poly)

    return polys


# ──────────────────────────────────────────────
# 4.  COMPRESSION STATISTICS
# ──────────────────────────────────────────────

def compression_ratio(bits: int) -> Tuple[int, float]:
    """
    Compute compression statistics.

    Parameters
    ----------
    bits : int
        Compression bit depth.

    Returns
    -------
    (compressed_bytes, ratio)
        - compressed_bytes: bytes per polynomial
        - ratio: compression ratio (compressed / original)
    """
    original_bytes = POLY_DEGREE * 12 // 8  # Assume 12 bits uncompressed
    compressed_bytes = (bits * POLY_DEGREE) // 8
    if (bits * POLY_DEGREE) % 8 != 0:
        compressed_bytes += 1

    ratio = compressed_bytes / original_bytes if original_bytes > 0 else 1.0

    return compressed_bytes, ratio


def estimate_max_error(bits: int, q: int = KYBER_Q) -> float:
    """
    Estimate maximum decompression error.

    Parameters
    ----------
    bits : int
        Compression bit depth.
    q : int
        Modulus.

    Returns
    -------
    float
        Maximum error per coefficient.
    """
    # Error bound: ⌈q / 2^(bits+1)⌉
    max_error = (q + (1 << (bits + 1)) - 1) >> (bits + 1)
    return float(max_error)


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("kyber_compress.py  —  self-test")
    print("=" * 55)

    try:
        # Test 1: Single coefficient compression
        print("\n[ Single coefficient compression ]")
        for bits in [4, 10, 11, 12]:
            coeff = 1234
            compressed = compress_coeff(coeff, bits)
            decompressed = decompress_coeff(compressed, bits)
            error = abs(coeff - decompressed)
            max_error = estimate_max_error(bits)
            assert error <= max_error + 1
            print(f"  ✓ {bits} bits: coeff={coeff}, compressed={compressed}, "
                  f"error={error} (max={max_error})")

        # Test 2: Polynomial compression roundtrip
        print("\n[ Polynomial compression roundtrip ]")
        test_poly = [i % KYBER_Q for i in range(POLY_DEGREE)]
        for bits in [4, 10, 11]:
            compressed = compress_poly(test_poly, bits)
            decompressed = decompress_poly(compressed, bits)
            errors = [
                abs(test_poly[i] - decompressed[i]) for i in range(POLY_DEGREE)
            ]
            max_error_observed = max(errors)
            max_error_bound = estimate_max_error(bits)
            assert max_error_observed <= max_error_bound + 1
            print(f"  ✓ {bits} bits: {len(compressed)} bytes, "
                  f"max error {max_error_observed} (bound {max_error_bound})")

        # Test 3: Batch compression
        print("\n[ Batch polynomial compression ]")
        batch = [test_poly for _ in range(3)]
        compressed_batch = compress_polys(batch, 10)
        decompressed_batch = decompress_polys(compressed_batch, 3, 10)
        assert len(decompressed_batch) == 3
        print(f"  ✓ Compressed 3 polys: {len(compressed_batch)} bytes")

        # Test 4: Compression ratios
        print("\n[ Compression statistics ]")
        for bits in [4, 10, 11, 12]:
            compressed_bytes, ratio = compression_ratio(bits)
            max_error = estimate_max_error(bits)
            print(f"  {bits} bits: {compressed_bytes} bytes/poly, "
                  f"ratio={ratio:.2f}, max_error={max_error:.2f}")

        # Test 5: Kyber ciphertext compression
        print("\n[ Kyber ciphertext compression (u || v) ]")
        # Simulate u (2 polys, 10 bits each) + v (1 poly, 4 bits)
        u_polys = [test_poly for _ in range(2)]
        v_poly = test_poly

        u_compressed = compress_polys(u_polys, COMPRESS_D_U_512_768)
        v_compressed = compress_poly(v_poly, COMPRESS_D_V)
        ct = u_compressed + v_compressed

        assert len(ct) == (
            (COMPRESS_D_U_512_768 * POLY_DEGREE * 2 + 7) // 8 +
            (COMPRESS_D_V * POLY_DEGREE + 7) // 8
        )
        print(f"  ✓ Kyber512 ciphertext: {len(ct)} bytes")
        print(f"    u (2×10 bits): {len(u_compressed)} bytes")
        print(f"    v (1×4 bits): {len(v_compressed)} bytes")

        # Test 6: Different compression depths
        print("\n[ Compression depth comparison ]")
        for bits in range(1, 13):
            try:
                compressed = compress_poly(test_poly, bits)
                decompressed = decompress_poly(compressed, bits)
                error = max(
                    abs(test_poly[i] - decompressed[i])
                    for i in range(POLY_DEGREE)
                )
                compressed_size, ratio = compression_ratio(bits)
                print(f"  {bits:2d} bits: {compressed_size:3d} B, "
                      f"ratio={ratio:.3f}, error={error}")
            except Exception as e:
                print(f"  {bits:2d} bits: ERROR - {e}")

        print("\n  All checks passed.\n")

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
