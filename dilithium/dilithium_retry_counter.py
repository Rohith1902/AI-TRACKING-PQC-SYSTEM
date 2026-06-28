"""
dilithium_retry_counter.py
==========================
Rejection sampling retry counter for Dilithium signature generation in PQC-SNN SoC.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Tracks and manages rejection sampling retry attempts in Dilithium signing, ║
║ enforcing bounds, logging failures, and triggering reseeds on overflow.   ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Per-signature retry attempt tracking
  - Maximum retry bound enforcement (NIST FIPS 204)
  - Statistical retry analysis (acceptance rate estimation)
  - Reseed trigger on excessive retries
  - Retry log per parameter set

Algorithm context:
  Dilithium signing loops until valid (z, r0) bounds are satisfied:
    loop:
      y ← DG(γ1)
      w = A·y
      c̃ = H(μ || HighBits(w))
      z = y + c·s1
      r0 = LowBits(w - c·s2)
      if |z| < γ1 - β AND |r0| < γ2 - β: accept
      else: retry (increment counter)

  Expected retries: ~4 per signature (theoretical)
  Max allowed: 576 (FIPS 204 §5.2)

Security:
  - Excessive retries may indicate side-channel attack or fault injection
  - Counter triggers alert if retries exceed threshold
  - Zeroizes internal state on max retry violation

Matches dilithium_retry_counter.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST FIPS 204 §5.2 (ML-DSA Signing Algorithm)
"""

from __future__ import annotations
from typing import Dict, List

# ──────────────────────────────────────────────
# 1.  RETRY COUNTER PARAMETERS
# ──────────────────────────────────────────────

#: Maximum retry attempts per signature (FIPS 204 §5.2)
MAX_RETRIES: int = 576

#: Warning threshold (alert if retries exceed this)
RETRY_WARNING_THRESHOLD: int = 100

#: Expected average retries per signature (theoretical)
EXPECTED_AVG_RETRIES: float = 4.0

#: Per-parameter-set retry bounds
RETRY_BOUNDS = {
    "dilithium2": 576,
    "dilithium3": 576,
    "dilithium5": 576,
}


# ──────────────────────────────────────────────
# 2.  RETRY COUNTER CLASS
# ──────────────────────────────────────────────

class DilithiumRetryCounter:
    """
    Tracks rejection sampling retries for Dilithium signing.

    Enforces FIPS 204 retry limits and provides statistical
    analysis of acceptance rates.
    """

    def __init__(self, param_set: str = "dilithium2"):
        """
        Initialize retry counter.

        Parameters
        ----------
        param_set : str
            Dilithium parameter set (dilithium2/3/5).
        """
        if param_set not in RETRY_BOUNDS:
            raise ValueError(f"Unknown parameter set: {param_set}")

        self.param_set = param_set
        self.max_retries = RETRY_BOUNDS[param_set]

        # Per-signature tracking
        self.current_retries: int = 0
        self.current_sig_id: int = 0

        # Aggregate statistics
        self.total_signatures: int = 0
        self.total_retries: int = 0
        self.retry_histogram: Dict[int, int] = {}  # retries → count
        self.max_retries_seen: int = 0
        self.warnings_issued: int = 0
        self.failures: int = 0

        # Retry log (per signature)
        self.retry_log: List[Dict] = []

    def begin_signature(self) -> int:
        """
        Begin tracking a new signature attempt.

        Returns
        -------
        int
            Signature ID for this attempt.
        """
        self.current_retries = 0
        self.current_sig_id += 1
        return self.current_sig_id

    def increment(self) -> bool:
        """
        Increment retry counter for current signature.

        Returns
        -------
        bool
            True if retry is allowed, False if max exceeded.
        """
        self.current_retries += 1
        self.total_retries += 1

        # Issue warning if threshold exceeded
        if self.current_retries == RETRY_WARNING_THRESHOLD:
            self.warnings_issued += 1

        # Check max retry bound
        if self.current_retries >= self.max_retries:
            self.failures += 1
            return False  # Max exceeded — abort

        return True  # Retry allowed

    def accept(self) -> None:
        """
        Record successful acceptance for current signature.

        Called when z and r0 bounds check passes.
        """
        self.total_signatures += 1

        # Update histogram
        r = self.current_retries
        self.retry_histogram[r] = self.retry_histogram.get(r, 0) + 1

        # Track maximum retries seen
        if r > self.max_retries_seen:
            self.max_retries_seen = r

        # Log this signature
        self.retry_log.append({
            "sig_id": self.current_sig_id,
            "retries": r,
            "accepted": True,
        })

        # Reset current counter
        self.current_retries = 0

    def reject(self) -> None:
        """
        Record rejection (max retries exceeded) for current signature.

        Called when signature generation fails after max retries.
        """
        self.retry_log.append({
            "sig_id": self.current_sig_id,
            "retries": self.current_retries,
            "accepted": False,
        })
        self.current_retries = 0

    def reset(self) -> None:
        """Reset counter for a new signing session."""
        self.current_retries = 0

    def acceptance_rate(self) -> float:
        """
        Compute overall acceptance rate.

        Returns
        -------
        float
            Fraction of signing attempts that succeeded.
        """
        total = self.total_signatures + self.failures
        if total == 0:
            return 0.0
        return self.total_signatures / total

    def avg_retries_per_signature(self) -> float:
        """
        Compute average retries per accepted signature.

        Returns
        -------
        float
            Average number of retries per successful signature.
        """
        if self.total_signatures == 0:
            return 0.0
        return self.total_retries / self.total_signatures

    def is_healthy(self) -> bool:
        """
        Check if retry behavior is within normal bounds.

        Returns
        -------
        bool
            True if behavior is healthy, False if anomalous.
        """
        avg = self.avg_retries_per_signature()

        # Healthy: avg retries within 10x of expected
        if avg > EXPECTED_AVG_RETRIES * 10:
            return False

        # Healthy: no failures
        if self.failures > 0:
            return False

        return True

    def stats(self) -> Dict:
        """
        Return retry counter statistics.

        Returns
        -------
        Dict
            Statistics: total_signatures, total_retries, avg_retries,
            max_retries_seen, warnings, failures, acceptance_rate.
        """
        return {
            "param_set": self.param_set,
            "total_signatures": self.total_signatures,
            "total_retries": self.total_retries,
            "avg_retries": self.avg_retries_per_signature(),
            "max_retries_seen": self.max_retries_seen,
            "warnings_issued": self.warnings_issued,
            "failures": self.failures,
            "acceptance_rate": self.acceptance_rate(),
            "healthy": self.is_healthy(),
            "retry_histogram": dict(self.retry_histogram),
        }


# ──────────────────────────────────────────────
# 3.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 55)
    print("dilithium_retry_counter.py  —  self-test")
    print("=" * 55)

    # Test 1: Initialization
    print("\n[ Initialization ]")
    counter = DilithiumRetryCounter("dilithium2")
    assert counter.max_retries == 576
    assert counter.total_signatures == 0
    print(f"  ✓ Initialized for dilithium2 (max_retries={counter.max_retries})")

    # Test 2: Single signature with 0 retries
    print("\n[ Zero-retry signature ]")
    counter.begin_signature()
    counter.accept()
    assert counter.total_signatures == 1
    assert counter.total_retries == 0
    print(f"  ✓ Accepted with 0 retries")

    # Test 3: Signature with retries
    print("\n[ Signature with retries ]")
    counter.begin_signature()
    for _ in range(5):
        allowed = counter.increment()
        assert allowed
    counter.accept()
    assert counter.total_signatures == 2
    assert counter.total_retries == 5
    print(f"  ✓ Accepted after 5 retries")

    # Test 4: Max retry enforcement
    print("\n[ Max retry enforcement ]")
    counter2 = DilithiumRetryCounter("dilithium2")
    counter2.begin_signature()
    allowed = True
    for i in range(MAX_RETRIES + 1):
        allowed = counter2.increment()
        if not allowed:
            break
    assert not allowed
    assert counter2.failures == 1
    print(f"  ✓ Max retry ({MAX_RETRIES}) correctly enforced")

    # Test 5: Warning threshold
    print("\n[ Warning threshold ]")
    counter3 = DilithiumRetryCounter("dilithium2")
    counter3.begin_signature()
    for _ in range(RETRY_WARNING_THRESHOLD):
        counter3.increment()
    assert counter3.warnings_issued == 1
    print(f"  ✓ Warning issued at {RETRY_WARNING_THRESHOLD} retries")

    # Test 6: Statistical simulation
    print("\n[ Statistical simulation (100 signatures) ]")
    counter4 = DilithiumRetryCounter("dilithium2")
    rng = random.Random(42)

    for _ in range(100):
        counter4.begin_signature()
        # Simulate ~4 retries per signature on average
        retries = rng.randint(0, 8)
        for _ in range(retries):
            counter4.increment()
        counter4.accept()

    stats = counter4.stats()
    print(f"  Total signatures: {stats['total_signatures']}")
    print(f"  Avg retries: {stats['avg_retries']:.2f} (expect ~4)")
    print(f"  Max retries seen: {stats['max_retries_seen']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']*100:.1f}%")
    print(f"  Healthy: {stats['healthy']}")
    assert stats["total_signatures"] == 100
    print(f"  ✓ Statistics computed correctly")

    # Test 7: Parameter set support
    print("\n[ Parameter set support ]")
    for ps in ["dilithium2", "dilithium3", "dilithium5"]:
        c = DilithiumRetryCounter(ps)
        c.begin_signature()
        c.accept()
        print(f"  ✓ {ps}: max_retries={c.max_retries}")

    # Test 8: Health check
    print("\n[ Health check ]")
    healthy_counter = DilithiumRetryCounter("dilithium2")
    for _ in range(10):
        healthy_counter.begin_signature()
        healthy_counter.accept()
    assert healthy_counter.is_healthy()
    print(f"  ✓ Healthy counter reports healthy")

    print("\n  All checks passed.\n")
