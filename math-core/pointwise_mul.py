"""
pointwise_mul.py
================
Pointwise (coefficient-wise) polynomial multiplication in NTT domain
for the PQC-SNN SoC Python Golden Model.

Matches pointwise_mul.sv (hardware RTL reference).

In NTT domain: if a_ntt = NTT(a) and b_ntt = NTT(b),
then INTT(a_ntt ⊙ b_ntt) = a * b mod (x^n + 1)

where ⊙ is coefficient-wise multiplication in NTT domain.

Each coefficient multiplication uses Montgomery reduction for efficiency.

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
"""

from __future__ import annotations
from typing import List, Sequence
from montgomery_reduce import (
    montgomery_reduce_kyber,
    montgomery_reduce_dilithium,
    KYBER_Q, DILITHIUM_Q,
)


def pointwise_mul_kyber(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """
    Pointwise multiplication in NTT domain over Z_{3329}[x].

    Computes r[i] = Mont(a[i] * b[i]) for each coefficient i.

    Used in polynomial multiplication via NTT:
        result = INTT(NTT(a) ⊙ NTT(b))

    Parameters
    ----------
    a, b : Sequence[int]
        Two 256-coefficient polynomials in NTT domain.
        Coefficients typically in (-KYBER_Q, KYBER_Q).

    Returns
    -------
    List[int]
        256 coefficients, each in (-KYBER_Q, KYBER_Q).
        Barrett reduction recommended before further operations.
    """
    return [montgomery_reduce_kyber(a[i] * b[i]) for i in range(256)]


def pointwise_mul_dilithium(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """
    Pointwise multiplication in NTT domain over Z_{8380417}[x].

    Parameters
    ----------
    a, b : Sequence[int]
        Two 256-coefficient polynomials in NTT domain.

    Returns
    -------
    List[int]
        256 coefficients, each in (-DILITHIUM_Q, DILITHIUM_Q).
    """
    return [montgomery_reduce_dilithium(a[i] * b[i]) for i in range(256)]


def pointwise_mul(
    a: Sequence[int],
    b: Sequence[int],
    q: int,
) -> List[int]:
    """Generic dispatch by modulus."""
    if q == KYBER_Q:
        return pointwise_mul_kyber(a, b)
    if q == DILITHIUM_Q:
        return pointwise_mul_dilithium(a, b)
    raise ValueError(f"Unsupported modulus {q}")


# Special case: multiplication by a small scalar (e.g., for debugging)
def pointwise_mul_scalar_kyber(a: Sequence[int], scalar: int) -> List[int]:
    """Multiply each coefficient by a scalar: r[i] = Mont(a[i] * scalar)."""
    return [montgomery_reduce_kyber(a[i] * scalar) for i in range(256)]


def pointwise_mul_scalar_dilithium(a: Sequence[int], scalar: int) -> List[int]:
    """Multiply each coefficient by a scalar."""
    return [montgomery_reduce_dilithium(a[i] * scalar) for i in range(256)]


if __name__ == "__main__":
    import random
    from ntt import ntt_kyber, ntt_dilithium
    from barrett_reduce import (
        barrett_reduce_poly_kyber,
        barrett_reduce_poly_dilithium,
    )

    random.seed(11)
    print("=" * 55)
    print("pointwise_mul.py  —  self-test")
    print("=" * 55)

    # Test 1: pointwise_mul produces 256 coefficients
    print("\n[ Kyber pointwise multiply structure ]")
    a = [random.randrange(KYBER_Q) for _ in range(256)]
    b = [random.randrange(KYBER_Q) for _ in range(256)]

    ntt_a = ntt_kyber(a)
    ntt_b = ntt_kyber(b)
    ntt_prod = pointwise_mul_kyber(ntt_a, ntt_b)
    ntt_prod_reduced = barrett_reduce_poly_kyber(ntt_prod)

    assert len(ntt_prod_reduced) == 256, "Product length mismatch"
    assert all(0 <= c < KYBER_Q for c in ntt_prod_reduced), \
        "Product coefficients out of range"
    print(f"  ✓ Product: 256 coefficients, all in [0, {KYBER_Q})")

    # Test 2: Scalar multiply preserves structure
    print("\n[ Kyber scalar multiply ]")
    scalar = 100
    scaled = pointwise_mul_scalar_kyber(a, scalar)
    scaled_reduced = barrett_reduce_poly_kyber(scaled)
    assert len(scaled_reduced) == 256
    assert all(0 <= c < KYBER_Q for c in scaled_reduced)
    print(f"  ✓ Scalar multiply (x{scalar}): 256 coefficients in [0, {KYBER_Q})")

    # Test 3: Dilithium
    print("\n[ Dilithium pointwise multiply structure ]")
    a2 = [random.randrange(DILITHIUM_Q) for _ in range(256)]
    b2 = [random.randrange(DILITHIUM_Q) for _ in range(256)]
    ntt_a2 = ntt_dilithium(a2)
    ntt_b2 = ntt_dilithium(b2)
    ntt_prod2 = pointwise_mul_dilithium(ntt_a2, ntt_b2)
    ntt_prod2_reduced = barrett_reduce_poly_dilithium(ntt_prod2)
    assert len(ntt_prod2_reduced) == 256
    assert all(0 <= c < DILITHIUM_Q for c in ntt_prod2_reduced)
    print(f"  ✓ Product: 256 coefficients, all in [0, {DILITHIUM_Q})")

    print("\n  All checks passed.\n")
