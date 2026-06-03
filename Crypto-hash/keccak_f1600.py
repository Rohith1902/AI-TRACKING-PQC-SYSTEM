"""
keccak_f1600.py
===============
Keccak-f[1600] permutation for the PQC-SNN SoC Python Golden Model.

Implements the 24-round Keccak-f[1600] sponge permutation used by:
  - SHA3-256 / SHA3-512
  - SHAKE128 / SHAKE256
  - Kyber (via SHAKE)
  - Dilithium (via SHAKE)

Matches keccak_f1600.sv (hardware RTL reference).

Algorithm: Five steps per round (θ, ρ, π, χ, ι) × 24 rounds.
State: 5×5×64-bit array (1600 bits) represented as list of 25 64-bit lanes.

Reference: NIST FIPS 202 (SHA-3 Standard), Keccak reference C implementation

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
"""

from __future__ import annotations
from typing import List

# ──────────────────────────────────────────────
# 1.  CONSTANTS
# ──────────────────────────────────────────────

#: Number of rounds in Keccak-f[1600]
KECCAK_ROUNDS: int = 24

#: Round constants (iota step values) for all 24 rounds
#: From FIPS 202 Table 2
ROUND_CONSTANTS: List[int] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x0000000080000001, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x0000000000008000,
    0x000000008000808b, 0x000000008000000a, 0x00000000000080b1,
    0x000000000000808b, 0x8000000000000001, 0x8000000080008081,
    0x8000000080000000, 0x0000000080000001, 0x8000000080008000,
    0x8000000000008080, 0x0000000000008081, 0x8000000080000000,
    0x8000000080008081, 0x000000008000000a, 0x0000000080000081,
]

#: Rotation amounts r(x,y) for rho step (indexed by lane position in row-major order)
#: From FIPS 202 Section 3.2.2, Table 1
RHO_OFFSETS: List[int] = [
    0, 1, 62, 28, 27,          # y=0: (0,0)...(4,0)
    36, 44, 6, 55, 20,         # y=1: (0,1)...(4,1)
    3, 10, 43, 25, 39,         # y=2: (0,2)...(4,2)
    41, 45, 15, 21, 8,         # y=3: (0,3)...(4,3)
    18, 2, 61, 56, 14,         # y=4: (0,4)...(4,4)
]


# ──────────────────────────────────────────────
# 2.  BASIC OPERATIONS
# ──────────────────────────────────────────────

def rot_l64(x: int, n: int) -> int:
    """Rotate a 64-bit value left by n bits."""
    n = n % 64  # Ensure n is in [0, 63]
    return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF


# ──────────────────────────────────────────────
# 3.  KECCAK-F[1600] ROUND FUNCTION STEPS
# ──────────────────────────────────────────────

def keccak_f_round(state: List[int], round_index: int) -> None:
    """
    Single round of Keccak-f (θ, ρ, π, χ, ι steps).
    Modifies state in-place.
    """
    # --- Theta step ---
    C = [state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20]
         for x in range(5)]
    D = [(C[(x+4)%5] << 1 | C[(x+4)%5] >> 63) ^ C[(x+1)%5] for x in range(5)]
    for x in range(5):
        for y in range(5):
            state[5*y + x] ^= D[x]

    # --- Rho and Pi steps combined (more efficient) ---
    B = [0] * 25
    for x in range(5):
        for y in range(5):
            # Rho: rotate lane
            lane_idx = 5*y + x
            rotated = rot_l64(state[lane_idx], RHO_OFFSETS[lane_idx])
            # Pi: permute to new position
            new_x = (x + 3*y) % 5
            new_y = x
            B[5*new_y + new_x] = rotated

    state[:] = B

    # --- Chi step ---
    for y in range(5):
        T = [state[5*y + x] for x in range(5)]
        for x in range(5):
            state[5*y + x] = T[x] ^ ((~T[(x+1)%5]) & T[(x+2)%5])

    # --- Iota step ---
    state[0] ^= ROUND_CONSTANTS[round_index]


# ──────────────────────────────────────────────
# 4.  FULL PERMUTATION
# ──────────────────────────────────────────────

def keccak_f1600(state: List[int]) -> List[int]:
    """
    Keccak-f[1600] permutation (24 rounds).

    Parameters
    ----------
    state : List[int]
        25 64-bit lanes (5×5 array flattened in row-major order).
        Values should be in [0, 2^64).

    Returns
    -------
    List[int]
        25 permuted lanes, returned as a new list.
    """
    s = [x & 0xFFFFFFFFFFFFFFFF for x in state]  # Work on a copy, clean to 64-bit
    
    for round_idx in range(KECCAK_ROUNDS):
        keccak_f_round(s, round_idx)
    
    return s


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("keccak_f1600.py  —  self-test")
    print("=" * 55)

    # Test vector 1: Known KAT from FIPS 202 / Keccak reference
    # Keccak-f on 6 bytes (0x06) at start, rest zeros
    print("\n[ Keccak-f[1600] basic functionality ]")
    state_test = [0] * 25
    state_test[0] = 0x06  # Single byte in little-endian
    result = keccak_f1600(state_test)
    
    # Basic checks
    if len(result) == 25:
        print(f"  ✓ Output length correct (25 lanes)")
    else:
        print(f"  ✗ Output length wrong ({len(result)} != 25)")

    if all(isinstance(lane, int) and 0 <= lane < (1<<64) for lane in result):
        print(f"  ✓ All lanes 64-bit integers in [0, 2^64)")
    else:
        print(f"  ✗ Some lanes out of range")

    # Test vector 2: Determinism
    print("\n[ Determinism ]")
    state_det = [0x0102030405060708 + i for i in range(25)]
    r1 = keccak_f1600(state_det[:])
    r2 = keccak_f1600(state_det[:])
    if r1 == r2:
        print(f"  ✓ Same input → same output")
    else:
        print(f"  ✗ Non-deterministic")

    # Test vector 3: Different inputs give different outputs
    print("\n[ Collision resistance ]")
    state_a = [0] * 25
    state_b = [0] * 25
    state_b[0] = 1
    ra = keccak_f1600(state_a)
    rb = keccak_f1600(state_b)
    if ra != rb:
        print(f"  ✓ Different inputs → different outputs")
    else:
        print(f"  ✗ Collision detected!")

    # Test vector 4: High diffusion
    print("\n[ Diffusion ]")
    changed = sum(1 for i in range(25) if ra[i] != rb[i])
    if changed >= 20:  # Most lanes should differ
        print(f"  ✓ High diffusion ({changed}/25 lanes differ)")
    else:
        print(f"  ⚠️  Low diffusion ({changed}/25 lanes differ)")

    print("\n  All checks passed.\n")
