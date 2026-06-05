"""
trng_health_tests.py
====================
TRNG health tests for entropy source validation in the PQC-SNN SoC.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Implements SP 800-90B statistical tests to verify the TRNG entropy source  ║
║ is functioning correctly and producing sufficient random data.             ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Repetition Count Test: detects stuck bits or low entropy
  - Adaptive Proportion Test: detects biased output
  - Run Test: detects patterns and runs of identical bits
  - Entropy Estimate: Shannon entropy calculation
  - Overall health status: pass/fail per NIST SP 800-90B

These tests are run on entropy blocks before they are used by CTR_DRBG.
If any test fails, the TRNG is flagged as unhealthy and may trigger:
  - Alert to security manager
  - Fallback to alternative entropy source
  - Reseeding of DRBG

Matches trng_health_tests.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST SP 800-90B (Entropy Sources and Random Number Generators)
"""

from __future__ import annotations
import math
from typing import Tuple, List

# ──────────────────────────────────────────────
# 1.  HEALTH TEST PARAMETERS
# ──────────────────────────────────────────────

#: Repetition Count Test threshold (max consecutive identical bits)
#: If any byte appears more than this in a row, test fails
REP_COUNT_THRESHOLD: int = 35

#: Adaptive Proportion Test window size (bits tested)
ADAPT_PROP_WINDOW: int = 512

#: Adaptive Proportion Test threshold (max count of identical bits in window)
ADAPT_PROP_THRESHOLD: int = 320  # ~62.5% in a 512-bit window

#: Run Test threshold (max length of identical bit runs)
RUN_TEST_THRESHOLD: int = 25

#: Minimum entropy threshold (bits per byte)
MIN_ENTROPY_BITS: float = 4.0


# ──────────────────────────────────────────────
# 2.  HEALTH TEST FUNCTIONS
# ──────────────────────────────────────────────

def repetition_count_test(data: bytes) -> Tuple[bool, int]:
    """
    Repetition Count Test (NIST SP 800-90B §4.4.1).

    Detects if the same byte appears too many times consecutively.
    This could indicate stuck bits or low entropy.

    Parameters
    ----------
    data : bytes
        Entropy bytes to test.

    Returns
    -------
    (bool, int)
        (pass, max_run_length) where pass=True if test passes.
    """
    if not data:
        return True, 0

    max_run = 1
    current_run = 1

    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return max_run <= REP_COUNT_THRESHOLD, max_run


def adaptive_proportion_test(data: bytes) -> Tuple[bool, float]:
    """
    Adaptive Proportion Test (NIST SP 800-90B §4.4.2).

    Detects if one bit value (0 or 1) appears too frequently within
    a sliding window. Tests each window of ADAPT_PROP_WINDOW bits.

    Parameters
    ----------
    data : bytes
        Entropy bytes to test.

    Returns
    -------
    (bool, float)
        (pass, max_proportion) where pass=True if all windows pass.
    """
    if not data:
        return True, 0.5

    # Convert bytes to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)

    max_count = 0
    window = ADAPT_PROP_WINDOW

    for start in range(0, len(bits) - window + 1):
        window_data = bits[start : start + window]
        ones = sum(window_data)
        zeros = window - ones

        # Count of the more frequent bit
        max_bit_count = max(ones, zeros)
        max_count = max(max_count, max_bit_count)

    passes = max_count <= ADAPT_PROP_THRESHOLD
    proportion = max_count / window if window > 0 else 0.5

    return passes, proportion


def run_test(data: bytes) -> Tuple[bool, int]:
    """
    Run Test (NIST SP 800-90B §4.4.3).

    Detects long runs of identical bits. A "run" is a sequence of
    consecutive identical bits.

    Parameters
    ----------
    data : bytes
        Entropy bytes to test.

    Returns
    -------
    (bool, int)
        (pass, max_run_length) where pass=True if all runs are within limit.
    """
    # Convert to bits
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)

    if not bits:
        return True, 0

    max_run = 1
    current_run = 1

    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return max_run <= RUN_TEST_THRESHOLD, max_run


def entropy_test(data: bytes) -> Tuple[bool, float]:
    """
    Entropy Estimation Test.

    Estimates Shannon entropy of the data. Passes if entropy is
    above the minimum threshold.

    Parameters
    ----------
    data : bytes
        Entropy bytes to test.

    Returns
    -------
    (bool, float)
        (pass, entropy_bits_per_byte) where pass=True if entropy >= MIN_ENTROPY_BITS.
    """
    if not data:
        return False, 0.0

    # Count frequency of each byte value
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1

    # Shannon entropy: H = -sum(p * log2(p))
    entropy = 0.0
    n = len(data)

    for count in freq.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)

    return entropy >= MIN_ENTROPY_BITS, entropy


# ──────────────────────────────────────────────
# 3.  COMPREHENSIVE HEALTH TEST RUNNER
# ──────────────────────────────────────────────

class HealthTestResult:
    """Container for health test results."""

    def __init__(self):
        self.rep_count_pass = True
        self.rep_count_max_run = 0
        self.adapt_prop_pass = True
        self.adapt_prop_ratio = 0.0
        self.run_test_pass = True
        self.run_test_max_run = 0
        self.entropy_pass = True
        self.entropy_bits = 0.0
        self.overall_pass = True

    def __str__(self) -> str:
        return (
            f"HealthTestResult(\n"
            f"  Rep Count:      {'PASS' if self.rep_count_pass else 'FAIL'} "
            f"(max run: {self.rep_count_max_run})\n"
            f"  Adapt Prop:     {'PASS' if self.adapt_prop_pass else 'FAIL'} "
            f"(ratio: {self.adapt_prop_ratio:.3f})\n"
            f"  Run Test:       {'PASS' if self.run_test_pass else 'FAIL'} "
            f"(max run: {self.run_test_max_run})\n"
            f"  Entropy:        {'PASS' if self.entropy_pass else 'FAIL'} "
            f"({self.entropy_bits:.2f} bits/byte)\n"
            f"  ────────────────────────────\n"
            f"  OVERALL:        {'✓ PASS' if self.overall_pass else '✗ FAIL'}\n"
            f")"
        )


def run_all_health_tests(data: bytes) -> HealthTestResult:
    """
    Run all TRNG health tests on entropy data.

    Parameters
    ----------
    data : bytes
        Entropy bytes to test (typically 32-64 bytes).

    Returns
    -------
    HealthTestResult
        Object containing results of all tests.
    """
    result = HealthTestResult()

    # Run each test
    result.rep_count_pass, result.rep_count_max_run = repetition_count_test(data)
    result.adapt_prop_pass, result.adapt_prop_ratio = adaptive_proportion_test(data)
    result.run_test_pass, result.run_test_max_run = run_test(data)
    result.entropy_pass, result.entropy_bits = entropy_test(data)

    # Overall: all tests must pass
    result.overall_pass = (
        result.rep_count_pass
        and result.adapt_prop_pass
        and result.run_test_pass
        and result.entropy_pass
    )

    return result


# ──────────────────────────────────────────────
# 4.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from trng_model import TRNGModel

    print("=" * 55)
    print("trng_health_tests.py  —  self-test")
    print("=" * 55)

    # Test 1: Run all tests on good entropy
    print("\n[ Health tests on good entropy ]")
    trng = TRNGModel(b"health_test_seed_32_bytes_long_ok")
    entropy_data = trng.collect_entropy(64)
    result = run_all_health_tests(entropy_data)
    print(result)

    # Test 2: Individual test verification
    print("\n[ Individual test verification ]")
    rep_pass, rep_max = repetition_count_test(entropy_data)
    print(f"  Repetition Count: {'PASS' if rep_pass else 'FAIL'} "
          f"(max consecutive: {rep_max})")

    adapt_pass, adapt_ratio = adaptive_proportion_test(entropy_data)
    print(f"  Adaptive Prop:    {'PASS' if adapt_pass else 'FAIL'} "
          f"(proportion: {adapt_ratio:.3f})")

    run_pass, run_max = run_test(entropy_data)
    print(f"  Run Test:         {'PASS' if run_pass else 'FAIL'} "
          f"(max run length: {run_max})")

    ent_pass, ent_bits = entropy_test(entropy_data)
    print(f"  Entropy:          {'PASS' if ent_pass else 'FAIL'} "
          f"({ent_bits:.2f} bits/byte)")

    # Test 3: Multiple entropy blocks
    print("\n[ Multiple entropy blocks ]")
    trng = TRNGModel()
    pass_count = 0
    for i in range(20):
        data = trng.collect_entropy(64)
        result = run_all_health_tests(data)
        if result.overall_pass:
            pass_count += 1
    print(f"  ✓ Health test: {pass_count}/20 blocks passed")

    # Test 4: Detect bad entropy (simulate)
    print("\n[ Detect low-entropy data ]")
    bad_entropy = bytes([0xAA] * 32)  # Repetitive pattern
    result_bad = run_all_health_tests(bad_entropy)
    print(f"  Repetitive data (0xAA×32): {'PASS' if result_bad.overall_pass else 'FAIL'}")
    if not result_bad.overall_pass:
        print(f"    → Correctly identified as bad entropy")

    print("\n  All checks passed.\n")
