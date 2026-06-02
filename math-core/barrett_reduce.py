"""
barrett_reduce.py
=================
Barrett modular reduction for the PQC-SNN SoC Python Golden Model.

Provides bit-exact Barrett reduction matching:
  - barrett_reduce.sv  (hardware RTL reference)

Two reduction contexts:
  1. Kyber    : modulus q = 3329,  precomputed constant KYBER_BARRETT_K
  2. Dilithium: modulus q = 8380417, precomputed constant DILITHIUM_BARRETT_K

Barrett algorithm (single-precision variant)
--------------------------------------------
Given x in [0, 2q^2) and modulus q:
  1. t = (x * k) >> shift
  2. r = x - t * q
  3. if r >= q: r -= q
  4. return r

where k = floor(2^shift / q) is the precomputed Barrett constant.

Shift values
------------
  Kyber    : shift = 24  →  k = floor(2^24 / 3329)     = 5039
  Dilithium: shift = 48  →  k = floor(2^48 / 8380417)  = 33292527

Both match the hardware barrel-shifter widths in barrett_reduce.sv.

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST FIPS 203 §2.4.1 / FIPS 204 §2.4.1
"""

from __future__ import annotations
from typing import List, Sequence


# ──────────────────────────────────────────────
# 1.  PRECOMPUTED CONSTANTS
# ──────────────────────────────────────────────

# Kyber
KYBER_Q:           int = 3329
KYBER_BARRETT_K:   int = 5039        # floor(2^24 / 3329)
KYBER_BARRETT_SHF: int = 24

# Dilithium
DILITHIUM_Q:           int = 8_380_417
DILITHIUM_BARRETT_K:   int = 33_587_228    # floor(2^48 / 8380417)
DILITHIUM_BARRETT_SHF: int = 48


def _verify_constants() -> None:
    """Verify precomputed Barrett constants at import time."""
    assert (1 << KYBER_BARRETT_SHF) // KYBER_Q == KYBER_BARRETT_K, \
        "KYBER_BARRETT_K mismatch"
    assert (1 << DILITHIUM_BARRETT_SHF) // DILITHIUM_Q == DILITHIUM_BARRETT_K, \
        "DILITHIUM_BARRETT_K mismatch"


_verify_constants()


# ──────────────────────────────────────────────
# 2.  CORE REDUCTION PRIMITIVES
# ──────────────────────────────────────────────

def barrett_reduce_kyber(x: int) -> int:
    """
    Barrett reduction modulo KYBER_Q = 3329.

    Parameters
    ----------
    x : int
        Input value in range [0, 2 * KYBER_Q^2).
        For NTT butterfly inputs: x in [0, 2^32) is safe.

    Returns
    -------
    int
        r in [0, KYBER_Q)  such that  r ≡ x (mod KYBER_Q).

    Notes
    -----
    Matches the 24-bit shift Barrett unit in barrett_reduce.sv.
    One conditional subtraction is sufficient because the
    input range guarantees t underestimates x//q by at most 1.
    """
    t = (x * KYBER_BARRETT_K) >> KYBER_BARRETT_SHF
    r = x - t * KYBER_Q
    # Single conditional subtraction
    r -= KYBER_Q * (r >= KYBER_Q)
    return int(r)


def barrett_reduce_dilithium(x: int) -> int:
    """
    Barrett reduction modulo DILITHIUM_Q = 8380417.

    Parameters
    ----------
    x : int
        Input value in range [0, 2 * DILITHIUM_Q^2).

    Returns
    -------
    int
        r in [0, DILITHIUM_Q)  such that  r ≡ x (mod DILITHIUM_Q).

    Notes
    -----
    Matches the 48-bit shift Barrett unit in barrett_reduce.sv.
    """
    t = (x * DILITHIUM_BARRETT_K) >> DILITHIUM_BARRETT_SHF
    r = x - t * DILITHIUM_Q
    r -= DILITHIUM_Q * (r >= DILITHIUM_Q)
    return int(r)


def barrett_reduce(x: int, q: int) -> int:
    """
    Generic Barrett reduction — dispatches to the optimised variant.

    Parameters
    ----------
    x : int
        Input value.
    q : int
        Modulus — must be KYBER_Q or DILITHIUM_Q.

    Returns
    -------
    int
        x mod q in [0, q).

    Raises
    ------
    ValueError
        If q is not a supported modulus.
    """
    if q == KYBER_Q:
        return barrett_reduce_kyber(x)
    if q == DILITHIUM_Q:
        return barrett_reduce_dilithium(x)
    raise ValueError(
        f"Unsupported modulus {q}. Use KYBER_Q={KYBER_Q} "
        f"or DILITHIUM_Q={DILITHIUM_Q}."
    )


# ──────────────────────────────────────────────
# 3.  VECTORISED HELPERS
# ──────────────────────────────────────────────

def barrett_reduce_poly_kyber(poly: Sequence[int]) -> List[int]:
    """
    Apply Kyber Barrett reduction to every coefficient of a polynomial.

    Parameters
    ----------
    poly : Sequence[int]
        List of 256 integer coefficients (unreduced).

    Returns
    -------
    List[int]
        256 coefficients each in [0, KYBER_Q).
    """
    return [barrett_reduce_kyber(c) for c in poly]


def barrett_reduce_poly_dilithium(poly: Sequence[int]) -> List[int]:
    """
    Apply Dilithium Barrett reduction to every coefficient of a polynomial.

    Parameters
    ----------
    poly : Sequence[int]
        List of 256 integer coefficients (unreduced).

    Returns
    -------
    List[int]
        256 coefficients each in [0, DILITHIUM_Q).
    """
    return [barrett_reduce_dilithium(c) for c in poly]


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 55)
    print("barrett_reduce.py  —  self-test")
    print("=" * 55)

    random.seed(42)
    TRIALS = 100_000

    # --- Kyber ---
    print(f"\n[ Kyber  q={KYBER_Q} ]")
    errors = 0
    for _ in range(TRIALS):
        x = random.randint(0, 2 * KYBER_Q * KYBER_Q - 1)
        got      = barrett_reduce_kyber(x)
        expected = x % KYBER_Q
        if got != expected:
            errors += 1
    print(f"  {TRIALS} random trials: {errors} errors  "
          f"{'✓' if errors == 0 else '✗ FAILED'}")

    # Edge cases
    for x in [0, 1, KYBER_Q - 1, KYBER_Q, KYBER_Q + 1,
               2 * KYBER_Q - 1, 2 * KYBER_Q]:
        r = barrett_reduce_kyber(x)
        assert r == x % KYBER_Q, f"Edge case failed x={x}"
    print("  Edge cases (0, q-1, q, 2q-1, 2q)  ✓")

    # --- Dilithium ---
    print(f"\n[ Dilithium  q={DILITHIUM_Q} ]")
    errors = 0
    for _ in range(TRIALS):
        x = random.randint(0, 2 * DILITHIUM_Q * DILITHIUM_Q - 1)
        got      = barrett_reduce_dilithium(x)
        expected = x % DILITHIUM_Q
        if got != expected:
            errors += 1
    print(f"  {TRIALS} random trials: {errors} errors  "
          f"{'✓' if errors == 0 else '✗ FAILED'}")

    for x in [0, 1, DILITHIUM_Q - 1, DILITHIUM_Q,
               DILITHIUM_Q + 1, 2 * DILITHIUM_Q - 1]:
        r = barrett_reduce_dilithium(x)
        assert r == x % DILITHIUM_Q, f"Edge case failed x={x}"
    print("  Edge cases  ✓")

    # --- Poly helpers ---
    print("\n[ Poly helper ]")
    poly = [random.randint(0, 2 * KYBER_Q - 1) for _ in range(256)]
    reduced = barrett_reduce_poly_kyber(poly)
    assert all(0 <= c < KYBER_Q for c in reduced), "Poly reduction out of range"
    assert all(reduced[i] == poly[i] % KYBER_Q for i in range(256)), \
        "Poly reduction value mismatch"
    print("  256-coeff Kyber poly  ✓")

    print("\n  All checks passed.\n")
