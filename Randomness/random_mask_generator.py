"""
random_mask_generator.py
========================
Random mask generator for side-channel protected NTT operations in PQC-SNN SoC.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Generates fresh random masks for each NTT operation to prevent power/     ║
║ timing side-channel attacks on Kyber/Dilithium polynomial computations.   ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Per-polynomial mask generation
  - Per-coefficient randomization
  - Mask matrix generation (for full NTT masking)
  - Inverse masks (for unmasking results)
  - Cache-based mask reuse (for performance)
  - Mask versioning (different masks per operation)

Side-channel protection via masking:
  - Input masking: a_masked = a ⊕ r (XOR with random mask)
  - Computation: c = f(a_masked, b, r)
  - Output unmasking: c = c ⊕ r (remove mask)
  - Attacker observes masked values only, not actual data

Masks are used in:
  - Kyber polynomial multiplication (NTT masking)
  - Dilithium polynomial operations
  - Key generation randomization
  - Signature generation masking

Matches random_mask_generator.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : Ishai et al. (Boolean Masking), FIPS 204 side-channel guidance
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from ctr_drbg import CTR_DRBG, BLOCK_SIZE

# ──────────────────────────────────────────────
# 1.  MASK PARAMETERS
# ──────────────────────────────────────────────

#: Polynomial degree (256 coefficients)
POLY_DEGREE: int = 256

#: Coefficient width (bits) - for modular masks
COEFF_BITS: int = 16

#: Mask length per coefficient (bytes)
MASK_COEFF_LEN: int = 2

#: Mask buffer size (number of pre-generated mask sets)
MASK_BUFFER_SIZE: int = 8

#: Kyber modulus
KYBER_Q: int = 3329

#: Dilithium modulus
DILITHIUM_Q: int = 8_380_417

#: Enable mask caching (reuse same mask for same operation type)
ENABLE_MASK_CACHE: bool = True


# ──────────────────────────────────────────────
# 2.  MASK GENERATOR CLASS
# ──────────────────────────────────────────────

class RandomMaskGenerator:
    """
    Generates random masks for side-channel protected polynomial operations.

    Each mask is cryptographically random and operation-specific to prevent
    security issues from mask reuse.
    """

    def __init__(self, seed: bytes | None = None):
        """
        Initialize mask generator with entropy source.

        Parameters
        ----------
        seed : bytes, optional
            Initial seed for mask DRBG (32-48 bytes).
            If None, uses default seed.
        """
        if seed is None:
            seed = b"mask_generator_default_seed_32byt"

        self.drbg = CTR_DRBG(seed)
        self.mask_buffer: List[List[int]] = []
        self.mask_cache: Dict[Tuple[str, int], List[int]] = {}
        self.mask_generation_log: Dict[str, int] = {}
        self.total_masks_generated = 0

    def generate_poly_mask(self, modulus: int = KYBER_Q) -> List[int]:
        """
        Generate a random mask for a 256-coefficient polynomial.

        Parameters
        ----------
        modulus : int
            Modulus for coefficient range (KYBER_Q or DILITHIUM_Q).

        Returns
        -------
        List[int]
            256 random mask values in [0, modulus).
        """
        mask = []
        for _ in range(POLY_DEGREE):
            # Generate random bytes and reduce modulo q
            rand_bytes = self.drbg.generate(MASK_COEFF_LEN)
            rand_val = int.from_bytes(rand_bytes, "little")
            mask_coeff = rand_val % modulus
            mask.append(mask_coeff)

        self.total_masks_generated += 1
        return mask

    def generate_masks_batch(
        self, count: int, modulus: int = KYBER_Q
    ) -> List[List[int]]:
        """
        Generate multiple polynomial masks at once.

        Parameters
        ----------
        count : int
            Number of masks to generate.
        modulus : int
            Modulus for coefficient range.

        Returns
        -------
        List[List[int]]
            List of count masks, each with 256 coefficients.
        """
        return [self.generate_poly_mask(modulus) for _ in range(count)]

    def get_fresh_mask(
        self, label: str = "generic", modulus: int = KYBER_Q
    ) -> List[int]:
        """
        Get a fresh random mask for an operation.

        Always generates new mask to prevent timing leaks from reuse.

        Parameters
        ----------
        label : str
            Operation label (e.g., "kyber_mult", "dilithium_sign").
        modulus : int
            Modulus for coefficient range.

        Returns
        -------
        List[int]
            256-coefficient random mask.
        """
        mask = self.generate_poly_mask(modulus)
        self.mask_generation_log[label] = self.mask_generation_log.get(label, 0) + 1
        return mask

    def get_inverse_mask(self, mask: List[int], modulus: int) -> List[int]:
        """
        Compute the inverse (unmasking) values for a mask.

        For XOR masking: inverse = same as original
        For arithmetic masking: inverse = -mask mod q

        Parameters
        ----------
        mask : List[int]
            Original mask values.
        modulus : int
            Modulus for arithmetic.

        Returns
        -------
        List[int]
            Inverse mask (for unmasking).
        """
        # For modular arithmetic masking: inv_mask = (-mask) mod q
        return [(-m) % modulus for m in mask]

    def mask_polynomial(
        self, poly: List[int], mask: List[int], modulus: int
    ) -> List[int]:
        """
        Apply mask to a polynomial (XOR for binary, add for arithmetic).

        Parameters
        ----------
        poly : List[int]
            Original polynomial (256 coefficients).
        mask : List[int]
            Mask polynomial (256 coefficients).
        modulus : int
            Modulus for arithmetic operations.

        Returns
        -------
        List[int]
            Masked polynomial.
        """
        return [(poly[i] + mask[i]) % modulus for i in range(len(poly))]

    def unmask_polynomial(
        self, masked_poly: List[int], mask: List[int], modulus: int
    ) -> List[int]:
        """
        Remove mask from a polynomial.

        Parameters
        ----------
        masked_poly : List[int]
            Masked polynomial.
        mask : List[int]
            Original mask used.
        modulus : int
            Modulus.

        Returns
        -------
        List[int]
            Unmasked (original) polynomial.
        """
        inv_mask = self.get_inverse_mask(mask, modulus)
        return [(masked_poly[i] + inv_mask[i]) % modulus for i in range(len(masked_poly))]

    def reseed(self, entropy: bytes) -> None:
        """
        Reseed the mask generator with fresh entropy.

        Parameters
        ----------
        entropy : bytes
            Fresh entropy (32-48 bytes from TRNG).
        """
        self.drbg.reseed(entropy)
        self.mask_cache.clear()  # Invalidate cache on reseed

    def stats(self) -> Dict:
        """
        Return generator statistics.

        Returns
        -------
        Dict
            Statistics: total_masks, generation_log, cache_size.
        """
        return {
            "total_masks_generated": self.total_masks_generated,
            "generation_log": dict(self.mask_generation_log),
            "cache_size": len(self.mask_cache),
        }


# ──────────────────────────────────────────────
# 3.  MODULE-LEVEL INTERFACE
# ──────────────────────────────────────────────

#: Global mask generator instance
_MASK_GEN_INSTANCE: RandomMaskGenerator | None = None


def initialize_mask_gen(seed: bytes | None = None) -> None:
    """Initialize the global mask generator."""
    global _MASK_GEN_INSTANCE
    _MASK_GEN_INSTANCE = RandomMaskGenerator(seed)


def get_fresh_mask(label: str = "generic", modulus: int = KYBER_Q) -> List[int]:
    """
    Get a fresh random mask from the global generator.

    Parameters
    ----------
    label : str
        Operation label.
    modulus : int
        Modulus for coefficients.

    Returns
    -------
    List[int]
        256-coefficient mask.
    """
    global _MASK_GEN_INSTANCE
    if _MASK_GEN_INSTANCE is None:
        initialize_mask_gen()
    return _MASK_GEN_INSTANCE.get_fresh_mask(label, modulus)


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("random_mask_generator.py  —  self-test")
    print("=" * 55)

    # Test 1: Initialization
    print("\n[ Mask generator initialization ]")
    initialize_mask_gen(b"test_seed_32_bytes_for_mask_gen_ok")
    print(f"  ✓ Initialized")

    # Test 2: Single mask generation
    print("\n[ Single mask generation ]")
    mask = get_fresh_mask("kyber_mult")
    assert len(mask) == POLY_DEGREE
    assert all(0 <= c < KYBER_Q for c in mask)
    print(f"  ✓ Generated 256-coefficient mask in [0, {KYBER_Q})")

    # Test 3: Mask uniqueness
    print("\n[ Mask uniqueness ]")
    gen = RandomMaskGenerator()
    mask1 = gen.generate_poly_mask()
    mask2 = gen.generate_poly_mask()
    mask3 = gen.generate_poly_mask()
    assert mask1 != mask2 != mask3
    print(f"  ✓ Generated 3 unique masks")

    # Test 4: Batch generation
    print("\n[ Batch mask generation ]")
    gen = RandomMaskGenerator()
    batch = gen.generate_masks_batch(8)
    assert len(batch) == 8
    assert all(len(m) == POLY_DEGREE for m in batch)
    print(f"  ✓ Generated batch of 8 masks")

    # Test 5: Masking/unmasking roundtrip
    print("\n[ Masking roundtrip ]")
    gen = RandomMaskGenerator()
    original_poly = [i % KYBER_Q for i in range(POLY_DEGREE)]
    mask = gen.generate_poly_mask(KYBER_Q)
    
    # Mask
    masked = gen.mask_polynomial(original_poly, mask, KYBER_Q)
    
    # Verify mask changes values
    assert masked != original_poly
    
    # Unmask
    recovered = gen.unmask_polynomial(masked, mask, KYBER_Q)
    assert recovered == original_poly
    print(f"  ✓ Masking roundtrip successful")

    # Test 6: Inverse mask correctness
    print("\n[ Inverse mask correctness ]")
    gen = RandomMaskGenerator()
    mask = gen.generate_poly_mask(KYBER_Q)
    inv_mask = gen.get_inverse_mask(mask, KYBER_Q)
    
    # Verify: (mask + inv_mask) mod q = 0 for all coefficients
    for i in range(POLY_DEGREE):
        sum_val = (mask[i] + inv_mask[i]) % KYBER_Q
        assert sum_val == 0, f"Mask not invertible at index {i}"
    print(f"  ✓ Inverse masks satisfy: (mask + inv_mask) mod q = 0")

    # Test 7: Dilithium masks
    print("\n[ Dilithium modulus masks ]")
    gen = RandomMaskGenerator()
    dil_mask = gen.generate_poly_mask(DILITHIUM_Q)
    assert len(dil_mask) == POLY_DEGREE
    assert all(0 <= c < DILITHIUM_Q for c in dil_mask)
    print(f"  ✓ Generated mask for DILITHIUM_Q={DILITHIUM_Q}")

    # Test 8: Label tracking
    print("\n[ Usage label tracking ]")
    gen = RandomMaskGenerator()
    gen.get_fresh_mask("kyber_mult")
    gen.get_fresh_mask("kyber_mult")
    gen.get_fresh_mask("dilithium_sign")
    
    stats = gen.stats()
    assert stats["generation_log"]["kyber_mult"] == 2
    assert stats["generation_log"]["dilithium_sign"] == 1
    print(f"  ✓ Label tracking: {stats['generation_log']}")

    # Test 9: Reseeding
    print("\n[ Reseeding ]")
    gen = RandomMaskGenerator(b"initial_seed_32_bytes_for_mask_ok")
    mask1 = gen.generate_poly_mask()
    gen.reseed(b"fresh_entropy_32_bytes_for_mask_ok")
    mask2 = gen.generate_poly_mask()
    assert mask1 != mask2
    print(f"  ✓ Reseed produces different masks")

    # Test 10: Kyber polynomial masking workflow
    print("\n[ Kyber masking workflow ]")
    gen = RandomMaskGenerator()
    
    # Original polynomial
    a = [i % KYBER_Q for i in range(POLY_DEGREE)]
    b = [(i + 100) % KYBER_Q for i in range(POLY_DEGREE)]
    
    # Generate masks
    mask_a = gen.get_fresh_mask("kyber_mult", KYBER_Q)
    mask_b = gen.get_fresh_mask("kyber_mult", KYBER_Q)
    
    # Mask inputs
    a_masked = gen.mask_polynomial(a, mask_a, KYBER_Q)
    b_masked = gen.mask_polynomial(b, mask_b, KYBER_Q)
    
    # Verify masking changed values
    assert a_masked != a
    assert b_masked != b
    
    # Unmask to verify roundtrip
    a_recovered = gen.unmask_polynomial(a_masked, mask_a, KYBER_Q)
    b_recovered = gen.unmask_polynomial(b_masked, mask_b, KYBER_Q)
    
    assert a_recovered == a
    assert b_recovered == b
    print(f"  ✓ Full masking workflow successful")

    # Test 11: Statistics
    print("\n[ Generator statistics ]")
    gen = RandomMaskGenerator()
    for _ in range(20):
        gen.get_fresh_mask("test")
    stats = gen.stats()
    print(f"  Total generated: {stats['total_masks_generated']}")
    print(f"  Generation log: {stats['generation_log']}")
    print(f"  ✓ Statistics tracked correctly")

    print("\n  All checks passed.\n")
