"""
mod_add_sub.py
==============
Modular addition and subtraction for the PQC-SNN SoC Python Golden Model.

Provides bit-exact modular arithmetic matching:
  - mod_add_sub.sv  (hardware RTL reference)

Simple constant-time operations without conditional branches (using arithmetic masking).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
"""

from __future__ import annotations
from typing import List, Sequence

KYBER_Q:     int = 3329
DILITHIUM_Q: int = 8_380_417


def mod_add(a: int, b: int, q: int) -> int:
    """
    Modular addition: (a + b) mod q.
    Result in [0, q).
    """
    r = a + b
    r -= q * (r >= q)
    return int(r)


def mod_sub(a: int, b: int, q: int) -> int:
    """
    Modular subtraction: (a - b) mod q.
    Result in [0, q).
    """
    r = a - b
    r += q * (r < 0)
    return int(r)


def mod_add_kyber(a: int, b: int) -> int:
    """(a + b) mod KYBER_Q."""
    return mod_add(a, b, KYBER_Q)


def mod_sub_kyber(a: int, b: int) -> int:
    """(a - b) mod KYBER_Q."""
    return mod_sub(a, b, KYBER_Q)


def mod_add_dilithium(a: int, b: int) -> int:
    """(a + b) mod DILITHIUM_Q."""
    return mod_add(a, b, DILITHIUM_Q)


def mod_sub_dilithium(a: int, b: int) -> int:
    """(a - b) mod DILITHIUM_Q."""
    return mod_sub(a, b, DILITHIUM_Q)


def mod_add_poly(a: Sequence[int], b: Sequence[int], q: int) -> List[int]:
    """Coefficient-wise modular addition of two length-256 polynomials."""
    return [mod_add(a[i], b[i], q) for i in range(len(a))]


def mod_sub_poly(a: Sequence[int], b: Sequence[int], q: int) -> List[int]:
    """Coefficient-wise modular subtraction of two length-256 polynomials."""
    return [mod_sub(a[i], b[i], q) for i in range(len(a))]


if __name__ == "__main__":
    import random
    random.seed(1)
    TRIALS = 200_000

    print("=" * 55)
    print("mod_add_sub.py  —  self-test")
    print("=" * 55)

    for label, q in [("Kyber", KYBER_Q), ("Dilithium", DILITHIUM_Q)]:
        err_add = err_sub = 0
        for _ in range(TRIALS):
            a = random.randrange(q)
            b = random.randrange(q)
            if mod_add(a, b, q) != (a + b) % q:
                err_add += 1
            if mod_sub(a, b, q) != (a - b) % q:
                err_sub += 1
        print(f"  {label}: add errors={err_add} sub errors={err_sub}  "
              f"{'✓' if err_add == err_sub == 0 else '✗ FAILED'}")

    # Edge cases
    assert mod_add(KYBER_Q - 1, 1, KYBER_Q) == 0
    assert mod_sub(0, 1, KYBER_Q) == KYBER_Q - 1
    print("  Edge cases  ✓")
    print("\n  All checks passed.\n")
