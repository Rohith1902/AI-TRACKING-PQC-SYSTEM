"""
ntt.py
======
Forward Number Theoretic Transform for the PQC-SNN SoC Python Golden Model.
Matches ntt_engine.sv + ntt_controller.sv (hardware RTL reference).

Algorithm: Cooley-Tukey, in-place, bit-reversed input order.
Ring     : Z_q[x] / (x^256 + 1)
Result   : NTT-domain polynomial, coefficients in (-7*q, 7*q).
           Apply barrett_reduce_poly after NTT if [0,q) needed.

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : CRYSTALS-Kyber/Dilithium reference C implementations
"""

from __future__ import annotations
from typing import List
from montgomery_reduce import (
    montgomery_reduce_kyber, montgomery_reduce_dilithium,
    KYBER_Q, DILITHIUM_Q, KYBER_R, DILITHIUM_R,
)

# Precompute zeta tables (Montgomery form)
_KYBER_ZETAS: List[int] = [
    (pow(17, i, KYBER_Q) * KYBER_R) % KYBER_Q
    for i in range(256)
]
_DILITHIUM_ZETAS: List[int] = [
    (pow(1753, i, DILITHIUM_Q) * DILITHIUM_R) % DILITHIUM_Q
    for i in range(256)
]

# Bit-reversal permutation table for n=256
def _build_brv(n: int) -> List[int]:
    bits = n.bit_length() - 1
    return [int(f'{i:0{bits}b}'[::-1], 2) for i in range(n)]

_BRV256: List[int] = _build_brv(256)


def ntt_kyber(poly: List[int]) -> List[int]:
    """
    Forward NTT over Z_{3329}[x]/(x^256+1).

    Parameters
    ----------
    poly : List[int]
        256 coefficients in [0, KYBER_Q).  Modified in-place AND returned.

    Returns
    -------
    List[int]
        256 NTT-domain coefficients in (-7*KYBER_Q, 7*KYBER_Q).
    """
    f = poly[:]          # work on a copy
    k = 1
    length = 128
    while length >= 1:
        for start in range(0, 256, 2 * length):
            zeta = _KYBER_ZETAS[_BRV256[k]]
            k += 1
            for j in range(start, start + length):
                t = montgomery_reduce_kyber(f[j + length] * zeta)
                f[j + length] = f[j] - t
                f[j]          = f[j] + t
        length >>= 1
    return f


def ntt_dilithium(poly: List[int]) -> List[int]:
    """
    Forward NTT over Z_{8380417}[x]/(x^256+1).

    Parameters
    ----------
    poly : List[int]
        256 coefficients in [0, DILITHIUM_Q).

    Returns
    -------
    List[int]
        256 NTT-domain coefficients.
    """
    f = poly[:]
    k = 1
    length = 128
    while length >= 1:
        for start in range(0, 256, 2 * length):
            zeta = _DILITHIUM_ZETAS[_BRV256[k]]
            k += 1
            for j in range(start, start + length):
                t = montgomery_reduce_dilithium(f[j + length] * zeta)
                f[j + length] = f[j] - t
                f[j]          = f[j] + t
        length >>= 1
    return f


def ntt(poly: List[int], q: int) -> List[int]:
    """Generic dispatch."""
    if q == KYBER_Q:
        return ntt_kyber(poly)
    if q == DILITHIUM_Q:
        return ntt_dilithium(poly)
    raise ValueError(f"Unsupported modulus {q}")


if __name__ == "__main__":
    import random
    from barrett_reduce import barrett_reduce_poly_kyber
    random.seed(5)
    print("=" * 45)
    print("ntt.py  —  self-test")
    print("=" * 45)

    # NTT of zero poly = zero
    z = [0] * 256
    assert all(c == 0 for c in ntt_kyber(z)), "Zero poly failed"
    print("  Zero polynomial  ✓")

    # NTT of [1,0,0,...] should be all-ones (in Montgomery domain)
    e0 = [0] * 256; e0[0] = 1
    r = ntt_kyber(e0)
    # Each NTT coefficient of the identity should be 1
    r_red = barrett_reduce_poly_kyber(r)
    assert all(c == 1 for c in r_red), f"Identity NTT failed: {r_red[:8]}"
    print("  Identity polynomial NTT = [1,1,...,1]  ✓")

    # Linearity: NTT(a+b) = NTT(a) + NTT(b) mod q
    a = [random.randrange(KYBER_Q) for _ in range(256)]
    b = [random.randrange(KYBER_Q) for _ in range(256)]
    ab = [(a[i] + b[i]) % KYBER_Q for i in range(256)]
    na = barrett_reduce_poly_kyber(ntt_kyber(a))
    nb = barrett_reduce_poly_kyber(ntt_kyber(b))
    nab = barrett_reduce_poly_kyber(ntt_kyber(ab))
    check = [(na[i] + nb[i]) % KYBER_Q for i in range(256)]
    assert nab == check, "Linearity check failed"
    print("  Linearity NTT(a+b) = NTT(a)+NTT(b)  ✓")

    print("\n  All checks passed.\n")
