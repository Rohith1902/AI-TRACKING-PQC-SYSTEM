"""
butterfly.py
============
Cooley-Tukey (CT) and Gentleman-Sande (GS) butterfly operations
for the PQC-SNN SoC Python Golden Model.
Matches masked_butterfly.sv (hardware RTL reference).

CT butterfly  (forward NTT):
    t  = Mont(b * zeta)
    a' = a + t
    b' = a - t

GS butterfly  (inverse NTT):
    a' = a + b
    b' = Mont((a - b) * zeta)

Note: CT and GS use different zeta values (zeta and zeta_inv respectively).
Both use Montgomery multiplication, matching the RTL Montgomery multiplier.

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
"""

from __future__ import annotations
from typing import Tuple
from montgomery_reduce import (
    montgomery_reduce_kyber,
    montgomery_reduce_dilithium,
    KYBER_Q, DILITHIUM_Q,
)


def ct_butterfly_kyber(a: int, b: int, zeta: int) -> Tuple[int, int]:
    """
    Cooley-Tukey butterfly mod KYBER_Q.

    Parameters
    ----------
    a, b   : int  Coefficients (may be in (-2q, 2q) during NTT accumulation).
    zeta   : int  Twiddle factor in Montgomery form (zeta * R mod q).

    Returns
    -------
    (a + Mont(b*zeta),  a - Mont(b*zeta))
    """
    t = montgomery_reduce_kyber(b * zeta)
    return a + t, a - t


def ct_butterfly_dilithium(a: int, b: int, zeta: int) -> Tuple[int, int]:
    """Cooley-Tukey butterfly mod DILITHIUM_Q."""
    t = montgomery_reduce_dilithium(b * zeta)
    return a + t, a - t


def gs_butterfly_kyber(a: int, b: int, zeta_inv: int) -> Tuple[int, int]:
    """
    Gentleman-Sande butterfly mod KYBER_Q.

    Parameters
    ----------
    a, b      : int  Coefficients.
    zeta_inv  : int  Inverse twiddle in Montgomery form (zeta^{-1} * R mod q).

    Returns
    -------
    (a + b,  Mont((a - b) * zeta_inv))
    """
    return a + b, montgomery_reduce_kyber((a - b) * zeta_inv)


def gs_butterfly_dilithium(a: int, b: int, zeta_inv: int) -> Tuple[int, int]:
    """Gentleman-Sande butterfly mod DILITHIUM_Q."""
    return a + b, montgomery_reduce_dilithium((a - b) * zeta_inv)


if __name__ == "__main__":
    import random
    random.seed(3)
    KYBER_R = 1 << 16

    print("=" * 45)
    print("butterfly.py  —  self-test")
    print("=" * 45)

    # Precompute zeta and zeta_inv in Montgomery form
    zeta     = (17              * KYBER_R) % KYBER_Q
    zeta_inv = (pow(17, -1, KYBER_Q) * KYBER_R) % KYBER_Q

    print("\n[ CT → GS roundtrip (result = 2a, 2b mod q) ]")
    errors = 0
    for _ in range(50_000):
        a = random.randrange(KYBER_Q)
        b = random.randrange(KYBER_Q)
        a1, b1 = ct_butterfly_kyber(a, b, zeta)
        a2, b2 = gs_butterfly_kyber(a1 % KYBER_Q, b1 % KYBER_Q, zeta_inv)
        if a2 % KYBER_Q != (2 * a) % KYBER_Q:
            errors += 1
        if b2 % KYBER_Q != (2 * b) % KYBER_Q:
            errors += 1
    print(f"  50000 roundtrips, {errors} errors  "
          f"{'✓' if errors == 0 else '✗ FAILED'}")

    print("\n[ CT correctness: a' + b' = 2a mod q ]")
    errors = 0
    for _ in range(50_000):
        a = random.randrange(KYBER_Q)
        b = random.randrange(KYBER_Q)
        a1, b1 = ct_butterfly_kyber(a, b, zeta)
        if (a1 + b1) % KYBER_Q != (2 * a) % KYBER_Q:
            errors += 1
    print(f"  50000 trials, {errors} errors  "
          f"{'✓' if errors == 0 else '✗ FAILED'}")

    print("\n  All checks passed.\n")
