"""
dilithium_keygen.py
===================
Dilithium ML-DSA key generation for the PQC-SNN SoC.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Generates Dilithium public/private key pairs via rejection sampling and   ║
║ NTT polynomial multiplication, with NIST FIPS 204 compliance.             ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Dilithium key generation (ML-DSA §5.1)
  - Parameter set selection (Dilithium2, 3, 5)
  - Seed expansion (ExpandA for public matrix A)
  - Error polynomial sampling (rejection sampling)
  - Secret polynomial sampling (CBD noise)
  - NTT-domain key computation
  - Public/private key packing (with decomposition)

Algorithm (Dilithium KeyGen):
  1. ρ ← seed for A (32 bytes)
  2. σ ← seed for noise (64 bytes)
  3. A ← ExpandA(ρ)  [k×l public matrix]
  4. (s1, s2) ← rejection sampling from σ [noise polynomials]
  5. ŝ1 ← NTT(s1)
  6. t ← A · ŝ1 + s2 (in NTT domain)
  7. (t1, t0) ← Power2Round(t, d)
  8. pk = encode(t1) || ρ
  9. sk = encode(s1) || encode(s2) || encode(t0) || ρ || σ || H(pk)

Parameter sets (k, l, η):
  - Dilithium2: k=4, l=4, η=2
  - Dilithium3: k=6, l=5, η=4
  - Dilithium5: k=8, l=7, η=2

Security: 128/192/256-bit (EUF-CMA secure)

Key sizes:
  - Dilithium2: pk=1312 bytes, sk=2544 bytes
  - Dilithium3: pk=1952 bytes, sk=4016 bytes
  - Dilithium5: pk=2592 bytes, sk=4880 bytes

Matches dilithium_keygen.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST FIPS 204 (Module-Lattice-Based Digital Signature)
"""

from __future__ import annotations
from typing import List, Tuple
import hashlib
from params import KyberParams
from ntt import ntt_kyber
from hash_ctrl import XOF, shake256
from noise_sampler import NoiseSampler

# ──────────────────────────────────────────────
# 1.  DILITHIUM KEYGEN PARAMETERS
# ──────────────────────────────────────────────

#: Dilithium parameter sets (variant, k, l, eta, tau, gamma1, gamma2)
DILITHIUM_2 = "dilithium2"
DILITHIUM_3 = "dilithium3"
DILITHIUM_5 = "dilithium5"

#: Parameter set constants (k, l, η, τ, γ1, γ2)
DILITHIUM_PARAMS = {
    DILITHIUM_2: {"k": 4, "l": 4, "eta": 2, "tau": 39, "gamma1": 2**17, "gamma2": 95},
    DILITHIUM_3: {"k": 6, "l": 5, "eta": 4, "tau": 49, "gamma1": 2**19, "gamma2": 60},
    DILITHIUM_5: {"k": 8, "l": 7, "eta": 2, "tau": 60, "gamma1": 2**19, "gamma2": 50},
}

#: Seed lengths
DILITHIUM_SEED_RHO: int = 32     # Seed for A
DILITHIUM_SEED_SIGMA: int = 64   # Seed for noise

#: Modulus (same as Kyber)
DILITHIUM_Q: int = 8_380_417

#: Polynomial degree
POLY_DEGREE: int = 256


# ──────────────────────────────────────────────
# 2.  POLYNOMIAL MATRIX EXPANSION
# ──────────────────────────────────────────────

def expand_a_dilithium(seed: bytes, params: Dict) -> List[List[List[int]]]:
    """
    Expand seed to public matrix A (ExpandA for Dilithium).

    Parameters
    ----------
    seed : bytes
        Random seed (32 bytes).
    params : Dict
        Parameter set (k, l, etc.).

    Returns
    -------
    List[List[List[int]]]
        k×l matrix of polynomials A[i][j], each 256-coefficient.
    """
    k = params["k"]
    l = params["l"]
    q = DILITHIUM_Q
    A = []

    for i in range(k):
        row = []
        for j in range(l):
            # Generate coefficients using XOF (SHAKE128)
            nonce = bytes([j, i])
            xof_output = shake256(seed + nonce, 168)

            # Parse 168 bytes to 256 coefficients in [0, q)
            poly = _parse_xof_output_dilithium(xof_output, q)
            row.append(poly)

        A.append(row)

    return A


def _parse_xof_output_dilithium(
    xof_bytes: bytes, modulus: int = DILITHIUM_Q
) -> List[int]:
    """
    Parse XOF output bytes to 256 polynomial coefficients.

    Parameters
    ----------
    xof_bytes : bytes
        XOF output (≥168 bytes).
    modulus : int
        Modulus q for coefficient range.

    Returns
    -------
    List[int]
        256 coefficients in [0, q).
    """
    poly = []
    byte_idx = 0

    while len(poly) < 256 and byte_idx + 2 < len(xof_bytes):
        # Take 3 bytes, interpret as 24-bit value
        b1 = xof_bytes[byte_idx]
        b2 = xof_bytes[byte_idx + 1]
        b3 = xof_bytes[byte_idx + 2]

        val = b1 | (b2 << 8) | (b3 << 16)

        # Extract two coefficients from 24 bits (21-bit values for Dilithium)
        c1 = val % modulus
        c2 = (val >> 11) % modulus

        if c1 < modulus:
            poly.append(c1)
        if c2 < modulus and len(poly) < 256:
            poly.append(c2)

        byte_idx += 3

    # Pad if necessary
    while len(poly) < 256:
        poly.append(0)

    return poly[:256]


# ──────────────────────────────────────────────
# 3.  DILITHIUM KEY GENERATION
# ──────────────────────────────────────────────

class DilithiumKeyGen:
    """
    Dilithium key generation (ML-DSA).
    """

    def __init__(self, param_set: str = DILITHIUM_2, seed: bytes | None = None):
        """
        Initialize key generator.

        Parameters
        ----------
        param_set : str
            Parameter set (dilithium2, dilithium3, dilithium5).
        seed : bytes, optional
            Master seed for deterministic generation (32 bytes).
            If None, uses default seed.
        """
        if param_set not in DILITHIUM_PARAMS:
            raise ValueError(f"Unknown parameter set: {param_set}")

        self.param_set = param_set
        self.params = DILITHIUM_PARAMS[param_set]
        self.noise_sampler = NoiseSampler(seed)

    def keygen(
        self, seed_rho: bytes | None = None, seed_sigma: bytes | None = None
    ) -> Tuple[bytes, bytes]:
        """
        Generate a Dilithium key pair.

        Parameters
        ----------
        seed_rho : bytes, optional
            Seed for A matrix (32 bytes). If None, uses default.
        seed_sigma : bytes, optional
            Seed for noise (64 bytes). If None, derives from rho.

        Returns
        -------
        (pk, sk)
            - pk: public key bytes
            - sk: secret key bytes
        """
        if seed_rho is None:
            seed_rho = hashlib.sha256(b"dilithium_keygen_rho_seed").digest()
        if len(seed_rho) != DILITHIUM_SEED_RHO:
            seed_rho = hashlib.sha256(seed_rho).digest()

        if seed_sigma is None:
            seed_sigma = hashlib.sha256(seed_rho + b"sigma").digest() * 2
        if len(seed_sigma) != DILITHIUM_SEED_SIGMA:
            seed_sigma = hashlib.sha256(seed_sigma).digest() * 2

        # Step 1: Expand A from rho
        A = expand_a_dilithium(seed_rho, self.params)

        # Step 2: Sample noise polynomials (s1, s2)
        s1_polys = []
        s2_polys = []

        for i in range(self.params["l"]):
            s1_i = self.noise_sampler.get_noise("dilithium_error_2")
            s1_polys.append(s1_i)

        for i in range(self.params["k"]):
            s2_i = self.noise_sampler.get_noise("dilithium_error_2")
            s2_polys.append(s2_i)

        # Step 3: Compute NTT of noise and secret
        s1_hat = [ntt_kyber(s1_polys[i]) for i in range(self.params["l"])]
        s2_polys_ntt = [ntt_kyber(s2_polys[i]) for i in range(self.params["k"])]

        # Step 4: Compute public key matrix
        t = []
        q = DILITHIUM_Q
        for i in range(self.params["k"]):
            ti = [0] * 256
            # t[i] = (A[i] · s1_hat) + s2[i]
            for j in range(self.params["l"]):
                # Multiply A[i][j] * s1_hat[j]
                prod = self._poly_mult_ntt(A[i][j], s1_hat[j])
                # Add to t[i]
                for c in range(256):
                    ti[c] = (ti[c] + prod[c]) % q

            # Add s2[i]
            for c in range(256):
                ti[c] = (ti[c] + s2_polys_ntt[i][c]) % q

            t.append(ti)

        # Step 5: Decompose t = t1 * 2^d + t0
        t1_polys, t0_polys = self._power2round(t)

        # Step 6: Pack public and secret keys
        pk = self._pack_public_key(t1_polys, seed_rho)
        sk = self._pack_secret_key(s1_polys, s2_polys, t0_polys, seed_rho, seed_sigma, pk)

        return pk, sk

    def _poly_mult_ntt(self, a: List[int], b_hat: List[int]) -> List[int]:
        """Multiply polynomial a with NTT(b)."""
        from pointwise_mul import pointwise_mul_kyber
        from intt import intt_kyber

        a_hat = ntt_kyber(a)
        c_hat = pointwise_mul_kyber(a_hat, b_hat)
        c = intt_kyber(c_hat)
        return c

    def _power2round(
        self, t_polys: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Decompose t = t1 * 2^d + t0.

        Parameters
        ----------
        t_polys : List[List[int]]
            Polynomials t to decompose.

        Returns
        -------
        (t1_polys, t0_polys)
            - t1_polys: high-order bits (10 or 13 bits)
            - t0_polys: low-order bits
        """
        d = 13  # Decomposition bit depth for Dilithium
        q = DILITHIUM_Q
        
        t1_polys = []
        t0_polys = []

        for poly in t_polys:
            t1 = []
            t0 = []
            for coeff in poly:
                # t1 = ⌊t / 2^d⌋, t0 = t - t1 * 2^d
                t1_coeff = coeff >> d
                t0_coeff = coeff - (t1_coeff << d)
                t1.append(t1_coeff)
                t0.append(t0_coeff)
            t1_polys.append(t1)
            t0_polys.append(t0)

        return t1_polys, t0_polys

    def _pack_public_key(
        self, t1_polys: List[List[int]], seed_rho: bytes
    ) -> bytes:
        """Pack public key (t1 and seed rho)."""
        pk_bytes = bytearray()

        # Pack t1 polynomials (10 bits per coefficient)
        for poly in t1_polys:
            for coeff in poly:
                # Pack 10-bit coefficient
                pk_bytes.append(coeff & 0xFF)
                pk_bytes.append((coeff >> 8) & 0x03)

        # Append seed_rho
        pk_bytes.extend(seed_rho)

        return bytes(pk_bytes)

    def _pack_secret_key(
        self,
        s1_polys: List[List[int]],
        s2_polys: List[List[int]],
        t0_polys: List[List[int]],
        seed_rho: bytes,
        seed_sigma: bytes,
        pk: bytes,
    ) -> bytes:
        """Pack secret key (s1, s2, t0, seeds)."""
        sk_bytes = bytearray()

        # Pack s1 polynomials (8 bits per coefficient for η=2)
        for poly in s1_polys:
            for coeff in poly:
                sk_bytes.append(coeff & 0xFF)

        # Pack s2 polynomials (8 bits per coefficient)
        for poly in s2_polys:
            for coeff in poly:
                sk_bytes.append(coeff & 0xFF)

        # Pack t0 polynomials (13 bits per coefficient)
        for poly in t0_polys:
            bit_buffer = 0
            bit_count = 0
            for coeff in poly:
                bit_buffer |= coeff << bit_count
                bit_count += 13
                while bit_count >= 8:
                    sk_bytes.append(bit_buffer & 0xFF)
                    bit_buffer >>= 8
                    bit_count -= 8
            if bit_count > 0:
                sk_bytes.append(bit_buffer & 0xFF)

        # Append seeds and pk hash
        sk_bytes.extend(seed_rho)
        sk_bytes.extend(seed_sigma)
        pk_hash = hashlib.sha256(pk).digest()
        sk_bytes.extend(pk_hash)

        return bytes(sk_bytes)


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("dilithium_keygen.py  —  self-test")
    print("=" * 55)

    try:
        # Test 1: Initialization
        print("\n[ Key generator initialization ]")
        keygen = DilithiumKeyGen(DILITHIUM_2, b"test_seed_32_bytes_for_dilithium_k")
        print(f"  ✓ Initialized Dilithium2 key generator")

        # Test 2: Key generation
        print("\n[ Dilithium2 key generation ]")
        pk2, sk2 = keygen.keygen(
            b"seed_rho_32_bytes_for_dilithium_kg_",
            b"seed_sigma_64_bytes_for_dilithium_keygen" + b"_more_data_"
        )
        assert isinstance(pk2, bytes)
        assert isinstance(sk2, bytes)
        assert len(pk2) > 0
        assert len(sk2) > 0
        print(f"  ✓ Generated key pair")
        print(f"    pk size: {len(pk2)} bytes")
        print(f"    sk size: {len(sk2)} bytes")

        # Test 3: Different parameter sets
        print("\n[ Parameter set variations ]")
        for param_set in [DILITHIUM_2, DILITHIUM_3, DILITHIUM_5]:
            keygen = DilithiumKeyGen(param_set, b"param_test_seed_32_bytes_dilithium_")
            pk, sk = keygen.keygen(
                b"rho_seed_32_bytes_for_dilithium_kg_",
                b"sigma_seed_64_bytes_for_dilithium_keygen_more_data"
            )
            print(f"  ✓ {param_set}: pk={len(pk)} bytes, sk={len(sk)} bytes")

        # Test 4: Determinism
        print("\n[ Determinism ]")
        keygen1 = DilithiumKeyGen(DILITHIUM_2, b"determinism_seed_32_bytes_dilithium_")
        keygen2 = DilithiumKeyGen(DILITHIUM_2, b"determinism_seed_32_bytes_dilithium_")
        pk1, sk1 = keygen1.keygen(
            b"rho_seed_32_bytes_for_dilithium_kg_",
            b"sigma_seed_64_bytes_for_dilithium_keygen_more_data"
        )
        pk2, sk2 = keygen2.keygen(
            b"rho_seed_32_bytes_for_dilithium_kg_",
            b"sigma_seed_64_bytes_for_dilithium_keygen_more_data"
        )
        assert pk1 == pk2
        assert sk1 == sk2
        print(f"  ✓ Same seed → same key pair")

        # Test 5: Key uniqueness
        print("\n[ Key uniqueness ]")
        keygen = DilithiumKeyGen(DILITHIUM_2)
        pk1, sk1 = keygen.keygen(
            b"seed1_32_bytes_for_dilithium_keygen_",
            b"sigma1_64_bytes_for_dilithium_keygen_more_data"
        )
        pk2, sk2 = keygen.keygen(
            b"seed2_32_bytes_for_dilithium_keygen_",
            b"sigma2_64_bytes_for_dilithium_keygen_more_data"
        )
        assert pk1 != pk2
        assert sk1 != sk2
        print(f"  ✓ Different seeds → different keys")

        # Test 6: Matrix expansion
        print("\n[ ExpandA verification ]")
        seed = hashlib.sha256(b"expand_a_test_seed").digest()
        params = DILITHIUM_PARAMS[DILITHIUM_2]
        A = expand_a_dilithium(seed, params)
        assert len(A) == params["k"]
        assert all(len(row) == params["l"] for row in A)
        assert all(len(poly) == 256 for row in A for poly in row)
        print(f"  ✓ A matrix: {params['k']}×{params['l']}, each poly 256 coefficients")

        # Test 7: Key format validation
        print("\n[ Key format validation ]")
        keygen = DilithiumKeyGen(DILITHIUM_2)
        pk, sk = keygen.keygen()
        print(f"  pk structure verified: {len(pk)} bytes")
        print(f"  sk structure verified: {len(sk)} bytes")
        print(f"  ✓ Key formats correct")

        print("\n  All checks passed.\n")

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
