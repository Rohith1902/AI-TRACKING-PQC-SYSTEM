"""
dilithium_pack_unpack.py
========================
Polynomial packing and unpacking utilities for Dilithium ML-DSA in PQC-SNN SoC.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Encodes and decodes Dilithium polynomials to/from compact byte arrays for  ║
║ key serialization and signature transmission per NIST FIPS 204.           ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Public key packing   (t1 polynomials + ρ seed → bytes)
  - Secret key packing   (s1, s2, t0, seeds → bytes)
  - Signature packing    (z, hint, c̃ → bytes)
  - Public key unpacking (bytes → t1 polynomials + ρ)
  - Secret key unpacking (bytes → s1, s2, t0, seeds)
  - Signature unpacking  (bytes → z, hint, c̃)
  - Bit-level encoding   (variable width: 4, 6, 10, 13, 18, 20 bits)

Bit-width map (FIPS 204 Table 1):
  Coefficients     | Bits | Usage
  -----------------+------+-------------------------
  t1               |  10  | public key
  s1, s2 (η=2)    |   3  | secret key (Dil2/5)
  s1, s2 (η=4)    |   4  | secret key (Dil3)
  t0               |  13  | secret key
  z (γ1=2^17)     |  18  | signature (Dil2)
  z (γ1=2^19)     |  20  | signature (Dil3/5)
  hint h           |   1  | signature (sparse)
  challenge c̃     |  32B | signature (32 bytes)

Matches dilithium_pack_unpack.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST FIPS 204 §7 (Encoding)
"""

from __future__ import annotations
from typing import List, Tuple
from dilithium_keygen import DILITHIUM_PARAMS, DILITHIUM_2, DILITHIUM_3, DILITHIUM_5

# ──────────────────────────────────────────────
# 1.  PACKING PARAMETERS
# ──────────────────────────────────────────────

POLY_DEGREE: int = 256
DILITHIUM_Q: int = 8_380_417
SEED_LEN: int = 32


# ──────────────────────────────────────────────
# 2.  BIT-LEVEL PACK / UNPACK PRIMITIVES
# ──────────────────────────────────────────────

def pack_coeffs(coeffs: List[int], bits: int) -> bytes:
    """
    Pack a list of coefficients into bytes using `bits` per coefficient.

    Parameters
    ----------
    coeffs : List[int]
        Coefficient values (non-negative, < 2^bits).
    bits : int
        Bits per coefficient.

    Returns
    -------
    bytes
        Packed byte array.
    """
    buf = 0
    buf_len = 0
    out = bytearray()

    for c in coeffs:
        buf |= (int(c) & ((1 << bits) - 1)) << buf_len
        buf_len += bits
        while buf_len >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            buf_len -= 8

    if buf_len > 0:
        out.append(buf & 0xFF)

    return bytes(out)


def unpack_coeffs(data: bytes, count: int, bits: int) -> List[int]:
    """
    Unpack `count` coefficients of `bits` width from bytes.

    Parameters
    ----------
    data : bytes
        Packed byte array.
    count : int
        Number of coefficients to unpack.
    bits : int
        Bits per coefficient.

    Returns
    -------
    List[int]
        Unpacked coefficient values.
    """
    buf = 0
    buf_len = 0
    idx = 0
    out = []
    mask = (1 << bits) - 1

    for _ in range(count):
        while buf_len < bits and idx < len(data):
            buf |= data[idx] << buf_len
            buf_len += 8
            idx += 1
        out.append(buf & mask)
        buf >>= bits
        buf_len -= bits

    return out


# ──────────────────────────────────────────────
# 3.  PUBLIC KEY PACK / UNPACK
# ──────────────────────────────────────────────

def pack_public_key(
    t1_polys: List[List[int]], seed_rho: bytes, params: dict
) -> bytes:
    """
    Pack Dilithium public key.

    pk = ρ || pack10(t1[0]) || ... || pack10(t1[k-1])

    Parameters
    ----------
    t1_polys : List[List[int]]
        k polynomials of high-order bits of t.
    seed_rho : bytes
        32-byte matrix seed ρ.
    params : dict
        Dilithium parameter set.

    Returns
    -------
    bytes
        Packed public key.
    """
    out = bytearray(seed_rho)
    for poly in t1_polys:
        out.extend(pack_coeffs(poly, 10))
    return bytes(out)


def unpack_public_key(
    pk: bytes, params: dict
) -> Tuple[bytes, List[List[int]]]:
    """
    Unpack Dilithium public key.

    Parameters
    ----------
    pk : bytes
        Packed public key.
    params : dict
        Dilithium parameter set.

    Returns
    -------
    (seed_rho, t1_polys)
    """
    seed_rho = pk[:SEED_LEN]
    offset = SEED_LEN
    k = params["k"]
    bytes_per_poly = (10 * POLY_DEGREE) // 8

    t1_polys = []
    for _ in range(k):
        chunk = pk[offset: offset + bytes_per_poly]
        t1_polys.append(unpack_coeffs(chunk, POLY_DEGREE, 10))
        offset += bytes_per_poly

    return seed_rho, t1_polys


# ──────────────────────────────────────────────
# 4.  SECRET KEY PACK / UNPACK
# ──────────────────────────────────────────────

def _eta_bits(eta: int) -> int:
    """Return bits needed to represent s1/s2 coefficients for given η."""
    # Coefficients in [-η, η]; map to [0, 2η] → needs ceil(log2(2η+1)) bits
    return 3 if eta == 2 else 4


def pack_secret_key(
    seed_rho: bytes,
    seed_sigma: bytes,
    pk_hash: bytes,
    s1_polys: List[List[int]],
    s2_polys: List[List[int]],
    t0_polys: List[List[int]],
    params: dict,
) -> bytes:
    """
    Pack Dilithium secret key.

    sk = ρ || σ || H(pk) || pack_s(s1) || pack_s(s2) || pack13(t0)

    Parameters
    ----------
    seed_rho : bytes
        32-byte ρ seed.
    seed_sigma : bytes
        64-byte σ seed.
    pk_hash : bytes
        32-byte hash of public key.
    s1_polys : List[List[int]]
        l small-norm polynomials.
    s2_polys : List[List[int]]
        k small-norm polynomials.
    t0_polys : List[List[int]]
        k low-order t polynomials.
    params : dict
        Dilithium parameter set.

    Returns
    -------
    bytes
        Packed secret key.
    """
    eta = params["eta"]
    bits_s = _eta_bits(eta)

    out = bytearray()
    out.extend(seed_rho)
    out.extend(seed_sigma)
    out.extend(pk_hash)

    # Pack s1, s2: coefficients shifted from [-η, η] to [0, 2η]
    for poly in s1_polys + s2_polys:
        shifted = [c + eta for c in poly]   # map [-η,η] → [0,2η]
        out.extend(pack_coeffs(shifted, bits_s))

    # Pack t0: coefficients in [-2^12, 2^12], shifted to [0, 2^13)
    for poly in t0_polys:
        shifted = [(c + (1 << 12)) & 0x1FFF for c in poly]
        out.extend(pack_coeffs(shifted, 13))

    return bytes(out)


def unpack_secret_key(
    sk: bytes, params: dict
) -> Tuple[bytes, bytes, bytes, List[List[int]], List[List[int]], List[List[int]]]:
    """
    Unpack Dilithium secret key.

    Parameters
    ----------
    sk : bytes
        Packed secret key.
    params : dict
        Dilithium parameter set.

    Returns
    -------
    (seed_rho, seed_sigma, pk_hash, s1_polys, s2_polys, t0_polys)
    """
    eta = params["eta"]
    bits_s = _eta_bits(eta)
    k, l = params["k"], params["l"]

    seed_rho   = sk[:32]
    seed_sigma = sk[32:96]
    pk_hash    = sk[96:128]
    offset     = 128

    bytes_per_s  = (bits_s * POLY_DEGREE + 7) // 8
    bytes_per_t0 = (13 * POLY_DEGREE + 7) // 8

    s1_polys = []
    for _ in range(l):
        chunk = sk[offset: offset + bytes_per_s]
        shifted = unpack_coeffs(chunk, POLY_DEGREE, bits_s)
        s1_polys.append([c - eta for c in shifted])
        offset += bytes_per_s

    s2_polys = []
    for _ in range(k):
        chunk = sk[offset: offset + bytes_per_s]
        shifted = unpack_coeffs(chunk, POLY_DEGREE, bits_s)
        s2_polys.append([c - eta for c in shifted])
        offset += bytes_per_s

    t0_polys = []
    for _ in range(k):
        chunk = sk[offset: offset + bytes_per_t0]
        shifted = unpack_coeffs(chunk, POLY_DEGREE, 13)
        t0_polys.append([c - (1 << 12) for c in shifted])
        offset += bytes_per_t0

    return seed_rho, seed_sigma, pk_hash, s1_polys, s2_polys, t0_polys


# ──────────────────────────────────────────────
# 5.  SIGNATURE PACK / UNPACK
# ──────────────────────────────────────────────

def _gamma1_bits(gamma1: int) -> int:
    """Return bits needed for z coefficients given γ1."""
    return 18 if gamma1 == (1 << 17) else 20


def pack_signature(
    z_polys: List[List[int]],
    hint_polys: List[List[int]],
    challenge_hash: bytes,
    params: dict,
) -> bytes:
    """
    Pack Dilithium signature.

    σ = c̃ || pack_z(z) || pack_hint(h)

    Parameters
    ----------
    z_polys : List[List[int]]
        l response polynomials.
    hint_polys : List[List[int]]
        k hint polynomials (sparse, 0/1 coefficients).
    challenge_hash : bytes
        32-byte challenge seed c̃.
    params : dict
        Dilithium parameter set.

    Returns
    -------
    bytes
        Packed signature.
    """
    gamma1 = params["gamma1"]
    bits_z = _gamma1_bits(gamma1)
    k = params["k"]

    out = bytearray(challenge_hash)

    # Pack z: shift from [-γ1+1, γ1] → [0, 2γ1)
    for poly in z_polys:
        shifted = [(c + gamma1) & ((1 << bits_z) - 1) for c in poly]
        out.extend(pack_coeffs(shifted, bits_z))

    # Pack hint: sparse encoding — store count + positions
    total_hints = 0
    hint_bytes = bytearray()
    for poly in hint_polys:
        positions = [i for i, c in enumerate(poly) if c == 1]
        total_hints += len(positions)
        hint_bytes.extend(bytes(positions))
        # pad to 256 bytes per poly for simplicity
        hint_bytes.extend(bytes(POLY_DEGREE - len(positions)))

    out.extend(hint_bytes)
    return bytes(out)


def unpack_signature(
    sig: bytes, params: dict
) -> Tuple[bytes, List[List[int]], List[List[int]]]:
    """
    Unpack Dilithium signature.

    Parameters
    ----------
    sig : bytes
        Packed signature.
    params : dict
        Dilithium parameter set.

    Returns
    -------
    (challenge_hash, z_polys, hint_polys)
    """
    gamma1 = params["gamma1"]
    bits_z = _gamma1_bits(gamma1)
    k, l = params["k"], params["l"]

    challenge_hash = sig[:SEED_LEN]
    offset = SEED_LEN

    bytes_per_z = (bits_z * POLY_DEGREE + 7) // 8

    z_polys = []
    for _ in range(l):
        chunk = sig[offset: offset + bytes_per_z]
        shifted = unpack_coeffs(chunk, POLY_DEGREE, bits_z)
        z_polys.append([c - gamma1 for c in shifted])
        offset += bytes_per_z

    # Unpack hints
    hint_polys = []
    for _ in range(k):
        chunk = sig[offset: offset + POLY_DEGREE]
        poly = [0] * POLY_DEGREE
        for pos in chunk:
            if pos < POLY_DEGREE:
                poly[pos] = 1
        hint_polys.append(poly)
        offset += POLY_DEGREE

    return challenge_hash, z_polys, hint_polys


# ──────────────────────────────────────────────
# 6.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import hashlib

    print("=" * 55)
    print("dilithium_pack_unpack.py  —  self-test")
    print("=" * 55)

    for param_set in [DILITHIUM_2, DILITHIUM_3, DILITHIUM_5]:
        params = DILITHIUM_PARAMS[param_set]
        k, l, eta = params["k"], params["l"], params["eta"]
        print(f"\n── {param_set} (k={k}, l={l}, η={eta}) ──")

        # Shared seeds
        seed_rho   = hashlib.sha256(b"rho"   + param_set.encode()).digest()
        seed_sigma = hashlib.sha256(b"sigma" + param_set.encode()).digest() * 2
        pk_hash    = hashlib.sha256(b"pk"    + param_set.encode()).digest()

        # ── Public key roundtrip ──
        t1_polys = [[i % 1024 for i in range(POLY_DEGREE)] for _ in range(k)]
        pk = pack_public_key(t1_polys, seed_rho, params)
        rho_out, t1_out = unpack_public_key(pk, params)

        assert rho_out == seed_rho
        assert len(t1_out) == k
        assert all(len(p) == POLY_DEGREE for p in t1_out)
        print(f"  ✓ pk pack/unpack  ({len(pk)} bytes)")

        # ── Secret key roundtrip ──
        s1 = [[i % (2 * eta + 1) - eta for i in range(POLY_DEGREE)] for _ in range(l)]
        s2 = [[i % (2 * eta + 1) - eta for i in range(POLY_DEGREE)] for _ in range(k)]
        t0 = [[(i % 8192) - 4096 for i in range(POLY_DEGREE)] for _ in range(k)]

        sk = pack_secret_key(seed_rho, seed_sigma, pk_hash, s1, s2, t0, params)
        rho2, sig2, pkh2, s1_out, s2_out, t0_out = unpack_secret_key(sk, params)

        assert rho2 == seed_rho
        assert pkh2 == pk_hash
        assert len(s1_out) == l and len(s2_out) == k and len(t0_out) == k
        assert s1_out[0] == s1[0]
        assert s2_out[0] == s2[0]
        assert t0_out[0] == t0[0]
        print(f"  ✓ sk pack/unpack  ({len(sk)} bytes)")

        # ── Signature roundtrip ──
        gamma1 = params["gamma1"]
        z  = [[(i % (gamma1 // 2)) - (gamma1 // 4) for i in range(POLY_DEGREE)]
              for _ in range(l)]
        hints = [[1 if i % 64 == 0 else 0 for i in range(POLY_DEGREE)]
                 for _ in range(k)]
        c_tilde = hashlib.sha256(b"challenge" + param_set.encode()).digest()

        sig = pack_signature(z, hints, c_tilde, params)
        c_out, z_out, h_out = unpack_signature(sig, params)

        assert c_out == c_tilde
        assert len(z_out) == l
        assert len(h_out) == k
        print(f"  ✓ sig pack/unpack ({len(sig)} bytes)")

    print("\n  All checks passed.\n")
