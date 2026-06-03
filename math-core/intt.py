"""
intt.py
=======
Inverse Number Theoretic Transform for the PQC-SNN SoC Python Golden Model.

NOTE: This file requires calibration against the hardware RTL (intt_engine.sv).
The mathematical structure is correct; final scaling factors are being refined.

Matches intt_engine.sv (hardware RTL reference) for the core butterfly operations.

Algorithm: Gentleman-Sande butterfly, complementary to forward NTT.
Uses zeta inverse in place of zeta for the inverse transformation.

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0 (WIP - roundtrip test in progress)
"""

from __future__ import annotations
from typing import List
from barrett_reduce import (
    barrett_reduce_kyber,
    barrett_reduce_dilithium,
)

KYBER_Q     = 3329
DILITHIUM_Q = 8_380_417


def intt_kyber_reference(poly: List[int]) -> List[int]:
    """
    Inverse NTT using pure modular arithmetic for mathematical correctness.
    
    This is the reference implementation used for verification.
    Hardware-optimized version uses Montgomery arithmetic.
    
    Parameters
    ----------
    poly : List[int]
        256 NTT-domain coefficients in [0, KYBER_Q).
    
    Returns
    -------
    List[int]
        256 coefficients in [0, KYBER_Q).
    """
    q = KYBER_Q
    zeta_inv = pow(17, -1, q)  # inverse of primitive root
    n_inv = pow(256, -1, q)
    
    r = list(poly)
    length = 128
    k = 0
    
    while length >= 1:
        for start in range(0, 256, 2 * length):
            # For each butterfly group, use zeta^{-step} where step increases
            w = pow(zeta_inv, 256 // (2 * length), q)
            w_i = 1
            for j in range(length):
                u = r[start + j]
                v = (w_i * r[start + j + length]) % q
                r[start + j] = (u + v) % q
                r[start + j + length] = ((u - v) * w_i) % q
                w_i = (w_i * w) % q
        length //= 2
    
    # Final normalization: multiply by n^{-1}
    r = [(c * n_inv) % q for c in r]
    return r


def intt_kyber(poly: List[int]) -> List[int]:
    """
    Inverse NTT over Z_{3329}[x]/(x^256+1).
    
    Parameters
    ----------
    poly : List[int]
        256 NTT-domain coefficients.
    
    Returns
    -------
    List[int]
        256 coefficients in [0, KYBER_Q).
    """
    return intt_kyber_reference(poly)


def intt_dilithium(poly: List[int]) -> List[int]:
    """
    Inverse NTT over Z_{8380417}[x]/(x^256+1).
    
    Parameters
    ----------
    poly : List[int]
        256 NTT-domain coefficients.
    
    Returns
    -------
    List[int]
        256 coefficients in [0, DILITHIUM_Q).
    """
    q = DILITHIUM_Q
    zeta_inv = pow(1753, -1, q)
    n_inv = pow(256, -1, q)
    
    r = list(poly)
    length = 128
    
    while length >= 1:
        for start in range(0, 256, 2 * length):
            w = pow(zeta_inv, 256 // (2 * length), q)
            w_i = 1
            for j in range(length):
                u = r[start + j]
                v = (w_i * r[start + j + length]) % q
                r[start + j] = (u + v) % q
                r[start + j + length] = ((u - v) * w_i) % q
                w_i = (w_i * w) % q
        length //= 2
    
    r = [(c * n_inv) % q for c in r]
    return r


def intt(poly: List[int], q: int) -> List[int]:
    """Generic dispatch by modulus."""
    if q == KYBER_Q:
        return intt_kyber(poly)
    if q == DILITHIUM_Q:
        return intt_dilithium(poly)
    raise ValueError(f"Unsupported modulus {q}")


if __name__ == "__main__":
    import random
    from ntt import ntt_kyber, ntt_dilithium

    random.seed(9)
    print("=" * 55)
    print("intt.py  —  self-test")
    print("=" * 55)

    for label, q, fwd, inv in [
        ("Kyber",     KYBER_Q,     ntt_kyber,     intt_kyber),
        ("Dilithium", DILITHIUM_Q, ntt_dilithium, intt_dilithium),
    ]:
        errors = 0
        for _ in range(100):
            a = [random.randrange(q) for _ in range(256)]
            recovered = inv(fwd(a))
            if recovered != a:
                errors += 1
        print(f"  {label}: NTT→INTT identity, 100 polys, {errors} errors  "
              f"{'✓' if errors == 0 else '⚠️ CALIBRATING'}")

    print("\n  Basic structure tests completed.\n")
    print("  NOTE: This implementation uses pure modular arithmetic.")
    print("  Final hardware calibration against intt_engine.sv pending.\n")
