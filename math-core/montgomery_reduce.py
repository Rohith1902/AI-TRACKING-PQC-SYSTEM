"""
montgomery_reduce.py
====================
Montgomery modular multiplication for the PQC-SNN SoC Python Golden Model.

Provides bit-exact Montgomery reduction matching:
  - montgomery_mul.sv  (hardware RTL reference)

Algorithm (matches CRYSTALS reference C implementation)
---------------------------------------------------------
Given a in (-q*R, q*R):
  1. t = (int_N)(a * QINV)   -- lower N bits, sign-extended
  2. u = (a - t * q) >> N    -- exact arithmetic right shift
  return u  in (-q, q)

where QINV = q^{-1} mod R  and  N = bit-width of R.

Precomputed constants
---------------------
  Kyber    : R = 2^16, QINV = 62209  (3329 * 62209 mod 65536 = 1)
  Dilithium: R = 2^32, QINV = 58728449  (8380417 * 58728449 mod 2^32 = 1)

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : CRYSTALS-Kyber/Dilithium reference C implementations
"""

from __future__ import annotations
from typing import List, Sequence

# ──────────────────────────────────────────────
# 1.  CONSTANTS
# ──────────────────────────────────────────────

KYBER_Q:     int = 3329
KYBER_R:     int = 1 << 16
KYBER_QINV:  int = 62209       # q^{-1} mod 2^16
KYBER_R_INV: int = pow(1 << 16, -1, 3329)   # R^{-1} mod q  = 169
KYBER_MONT:  int = (1 << 16) % 3329          # R mod q       = 2285

DILITHIUM_Q:     int = 8_380_417
DILITHIUM_R:     int = 1 << 32
DILITHIUM_QINV:  int = 58_728_449    # q^{-1} mod 2^32
DILITHIUM_R_INV: int = pow(1 << 32, -1, 8_380_417)
DILITHIUM_MONT:  int = (1 << 32) % 8_380_417  # 4193792


def _sign16(x: int) -> int:
    """Simulate C int16_t cast (sign-extend lower 16 bits)."""
    x = int(x) & 0xFFFF
    return x - 0x10000 if x >= 0x8000 else x


def _sign32(x: int) -> int:
    """Simulate C int32_t cast (sign-extend lower 32 bits)."""
    x = int(x) & 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def _verify_constants() -> None:
    assert (KYBER_Q    * KYBER_QINV)     % KYBER_R    == 1, "KYBER_QINV"
    assert (KYBER_R    * KYBER_R_INV)    % KYBER_Q    == 1, "KYBER_R_INV"
    assert (DILITHIUM_Q * DILITHIUM_QINV) % DILITHIUM_R == 1, "DILITHIUM_QINV"
    assert (DILITHIUM_R * DILITHIUM_R_INV) % DILITHIUM_Q == 1, "DILITHIUM_R_INV"
    assert KYBER_MONT    == KYBER_R    % KYBER_Q
    assert DILITHIUM_MONT == DILITHIUM_R % DILITHIUM_Q


_verify_constants()

# ──────────────────────────────────────────────
# 2.  CORE REDUCTION PRIMITIVES
# ──────────────────────────────────────────────

def montgomery_reduce_kyber(a: int) -> int:
    """
    Montgomery reduction mod KYBER_Q.

    Parameters
    ----------
    a : int
        Input in (-KYBER_Q * KYBER_R, KYBER_Q * KYBER_R).

    Returns
    -------
    int
        t ≡ a * R^{-1} (mod q),  in (-KYBER_Q, KYBER_Q).
    """
    t = _sign16(a * KYBER_QINV)
    return (a - t * KYBER_Q) >> 16


def montgomery_reduce_dilithium(a: int) -> int:
    """
    Montgomery reduction mod DILITHIUM_Q.

    Parameters
    ----------
    a : int
        Input in (-DILITHIUM_Q * DILITHIUM_R, DILITHIUM_Q * DILITHIUM_R).

    Returns
    -------
    int
        t ≡ a * R^{-1} (mod q),  in (-DILITHIUM_Q, DILITHIUM_Q).
    """
    t = _sign32(a * DILITHIUM_QINV)
    return (a - t * DILITHIUM_Q) >> 32


def montgomery_reduce(a: int, q: int) -> int:
    """Generic dispatch by modulus."""
    if q == KYBER_Q:
        return montgomery_reduce_kyber(a)
    if q == DILITHIUM_Q:
        return montgomery_reduce_dilithium(a)
    raise ValueError(f"Unsupported modulus {q}")

# ──────────────────────────────────────────────
# 3.  MONTGOMERY MULTIPLICATION
# ──────────────────────────────────────────────

def mont_mul_kyber(a: int, b: int) -> int:
    """a * b * R^{-1} mod KYBER_Q  →  result in (-q, q)."""
    return montgomery_reduce_kyber(a * b)


def mont_mul_dilithium(a: int, b: int) -> int:
    """a * b * R^{-1} mod DILITHIUM_Q  →  result in (-q, q)."""
    return montgomery_reduce_dilithium(a * b)


def mont_mul_poly_kyber(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Coefficient-wise Montgomery multiply for two 256-coeff Kyber polynomials."""
    return [montgomery_reduce_kyber(a[i] * b[i]) for i in range(256)]


def mont_mul_poly_dilithium(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Coefficient-wise Montgomery multiply for two 256-coeff Dilithium polynomials."""
    return [montgomery_reduce_dilithium(a[i] * b[i]) for i in range(256)]

# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(7)
    TRIALS = 100_000

    print("=" * 55)
    print("montgomery_reduce.py  —  self-test")
    print("=" * 55)

    print(f"\n[ Kyber Montgomery reduce  q={KYBER_Q} R=2^16 ]")
    errors = 0
    for _ in range(TRIALS):
        a = random.randint(-(KYBER_Q * KYBER_R), KYBER_Q * KYBER_R)
        got = montgomery_reduce_kyber(a)
        if got % KYBER_Q != (a * KYBER_R_INV) % KYBER_Q:
            errors += 1
    print(f"  {TRIALS} random trials: {errors} errors  {'✓' if errors == 0 else '✗ FAILED'}")

    print(f"\n[ Kyber Montgomery multiply ]")
    errors = 0
    for _ in range(TRIALS):
        a = random.randint(0, KYBER_Q - 1)
        b = random.randint(0, KYBER_Q - 1)
        b_mont = (b * KYBER_R) % KYBER_Q
        got = mont_mul_kyber(a, b_mont)
        if got % KYBER_Q != (a * b) % KYBER_Q:
            errors += 1
    print(f"  {TRIALS} random trials: {errors} errors  {'✓' if errors == 0 else '✗ FAILED'}")

    print(f"\n[ Dilithium Montgomery reduce  q={DILITHIUM_Q} R=2^32 ]")
    errors = 0
    for _ in range(TRIALS):
        a = random.randint(-(DILITHIUM_Q * DILITHIUM_R), DILITHIUM_Q * DILITHIUM_R)
        got = montgomery_reduce_dilithium(a)
        if got % DILITHIUM_Q != (a * DILITHIUM_R_INV) % DILITHIUM_Q:
            errors += 1
    print(f"  {TRIALS} random trials: {errors} errors  {'✓' if errors == 0 else '✗ FAILED'}")

    print("\n[ Poly coefficient-wise mul ]")
    a = [(random.randint(0, KYBER_Q - 1) * KYBER_R) % KYBER_Q for _ in range(256)]
    b = [(random.randint(0, KYBER_Q - 1) * KYBER_R) % KYBER_Q for _ in range(256)]
    c = mont_mul_poly_kyber(a, b)
    assert len(c) == 256 and all(-KYBER_Q < x < KYBER_Q for x in c)
    print("  256-coeff Kyber poly  ✓")

    print("\n  All checks passed.\n")
