"""
kdf_sp800_108.py
================
NIST SP 800-108 Key Derivation Function (Counter Mode) for PQC-SNN SoC.

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Implements NIST SP 800-108 counter-mode KDF using HMAC-SHA256 to derive    ║
║ multiple independent keys from a single master key for FIPS compliance.   ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - KDF in Counter Mode (SP 800-108 §5.1)
  - KDF in Feedback Mode (SP 800-108 §5.2)
  - KDF in Double-Pipeline Mode (SP 800-108 §5.3)
  - Fixed input data construction (label || context || length)
  - PRF selection (HMAC-SHA256)
  - Multi-key derivation with counter-based domain separation

Algorithm (Counter Mode, SP 800-108 §5.1):
  For i = 1 to ⌈L/h⌉:
    K(i) = PRF(KI, [i]_2 || Label || 0x00 || Context || [L]_2)
  KO = K(1) || K(2) || ... || K(⌈L/h⌉)  truncated to L bits

  where:
    KI      : key-derivation key (master key)
    [i]_2   : binary representation of counter i (4 bytes, big-endian)
    Label   : identifies purpose of derived key
    Context : additional context (e.g. nonce, party IDs)
    [L]_2   : binary representation of output length in bits (4 bytes)
    h       : PRF output length in bits (256 for HMAC-SHA256)

Algorithm (Feedback Mode, SP 800-108 §5.2):
  K(0) = IV (optional, defaults to empty)
  K(i) = PRF(KI, K(i-1) || [i]_2 || Label || 0x00 || Context || [L]_2)

Algorithm (Double-Pipeline Mode, SP 800-108 §5.3):
  A(0) = Label || 0x00 || Context || [L]_2
  A(i) = PRF(KI, A(i-1))
  K(i) = PRF(KI, A(i) || [i]_2 || Label || 0x00 || Context || [L]_2)

Usage in AEGIS-NEURO:
  - Derive AES session keys from master key for each crypto operation
  - Derive per-channel keys for AXI bus encryption
  - Separate keys for Kyber/Dilithium subsystems from one root key

Matches kdf_sp800_108.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : NIST SP 800-108r1 (Key Derivation Using Pseudorandom Functions)
"""

from __future__ import annotations
import hmac
import hashlib
from typing import Dict, List

# ──────────────────────────────────────────────
# 1.  KDF PARAMETERS
# ──────────────────────────────────────────────

#: PRF output length (bits) — HMAC-SHA256
PRF_OUTPUT_BITS: int = 256
PRF_OUTPUT_BYTES: int = PRF_OUTPUT_BITS // 8

#: Counter field length (bytes, big-endian per SP 800-108)
COUNTER_LEN: int = 4

#: Length field length (bytes, big-endian, in bits)
LENGTH_FIELD_LEN: int = 4

#: Separator byte between Label and Context
SEPARATOR: bytes = b"\x00"


# ──────────────────────────────────────────────
# 2.  PRF PRIMITIVE
# ──────────────────────────────────────────────

def prf(key: bytes, data: bytes) -> bytes:
    """
    Pseudorandom function: HMAC-SHA256(key, data).

    Parameters
    ----------
    key : bytes
        Key-derivation key (KI).
    data : bytes
        Input data.

    Returns
    -------
    bytes
        32-byte PRF output.
    """
    return hmac.new(key, data, hashlib.sha256).digest()


def _int_to_bytes(value: int, length: int) -> bytes:
    """Encode integer as big-endian bytes of given length."""
    return value.to_bytes(length, byteorder="big")


# ──────────────────────────────────────────────
# 3.  COUNTER MODE KDF (SP 800-108 §5.1)
# ──────────────────────────────────────────────

def kdf_counter_mode(
    ki: bytes, label: bytes, context: bytes, length_bytes: int
) -> bytes:
    """
    KDF in Counter Mode per NIST SP 800-108 §5.1.

    Parameters
    ----------
    ki : bytes
        Key-derivation key (master key).
    label : bytes
        Purpose identifier for the derived key.
    context : bytes
        Additional context (nonce, party IDs, etc.).
    length_bytes : int
        Desired output length in bytes.

    Returns
    -------
    bytes
        Derived key material of `length_bytes`.
    """
    length_bits = length_bytes * 8
    fixed_suffix = label + SEPARATOR + context + _int_to_bytes(length_bits, LENGTH_FIELD_LEN)

    n_blocks = (length_bytes + PRF_OUTPUT_BYTES - 1) // PRF_OUTPUT_BYTES
    output = b""

    for i in range(1, n_blocks + 1):
        counter = _int_to_bytes(i, COUNTER_LEN)
        block = prf(ki, counter + fixed_suffix)
        output += block

    return output[:length_bytes]


# ──────────────────────────────────────────────
# 4.  FEEDBACK MODE KDF (SP 800-108 §5.2)
# ──────────────────────────────────────────────

def kdf_feedback_mode(
    ki: bytes,
    label: bytes,
    context: bytes,
    length_bytes: int,
    iv: bytes = b"",
) -> bytes:
    """
    KDF in Feedback Mode per NIST SP 800-108 §5.2.

    Each block feeds the previous PRF output back into the next.

    Parameters
    ----------
    ki : bytes
        Key-derivation key.
    label : bytes
        Purpose identifier.
    context : bytes
        Additional context.
    length_bytes : int
        Desired output length in bytes.
    iv : bytes
        Optional initialization vector (K(0)).

    Returns
    -------
    bytes
        Derived key material of `length_bytes`.
    """
    length_bits = length_bytes * 8
    fixed_suffix = label + SEPARATOR + context + _int_to_bytes(length_bits, LENGTH_FIELD_LEN)

    n_blocks = (length_bytes + PRF_OUTPUT_BYTES - 1) // PRF_OUTPUT_BYTES
    output = b""
    k_prev = iv

    for i in range(1, n_blocks + 1):
        counter = _int_to_bytes(i, COUNTER_LEN)
        k_prev = prf(ki, k_prev + counter + fixed_suffix)
        output += k_prev

    return output[:length_bytes]


# ──────────────────────────────────────────────
# 5.  DOUBLE-PIPELINE MODE KDF (SP 800-108 §5.3)
# ──────────────────────────────────────────────

def kdf_double_pipeline_mode(
    ki: bytes, label: bytes, context: bytes, length_bytes: int
) -> bytes:
    """
    KDF in Double-Pipeline Iteration Mode per NIST SP 800-108 §5.3.

    Uses two PRF invocations per block for additional diffusion.

    Parameters
    ----------
    ki : bytes
        Key-derivation key.
    label : bytes
        Purpose identifier.
    context : bytes
        Additional context.
    length_bytes : int
        Desired output length in bytes.

    Returns
    -------
    bytes
        Derived key material of `length_bytes`.
    """
    length_bits = length_bytes * 8
    fixed_suffix = label + SEPARATOR + context + _int_to_bytes(length_bits, LENGTH_FIELD_LEN)

    n_blocks = (length_bytes + PRF_OUTPUT_BYTES - 1) // PRF_OUTPUT_BYTES
    output = b""

    # A(0) = fixed input data
    a_prev = fixed_suffix

    for i in range(1, n_blocks + 1):
        a_i = prf(ki, a_prev)
        counter = _int_to_bytes(i, COUNTER_LEN)
        k_i = prf(ki, a_i + counter + fixed_suffix)
        output += k_i
        a_prev = a_i

    return output[:length_bytes]


# ──────────────────────────────────────────────
# 6.  MULTI-KEY DERIVATION HELPER
# ──────────────────────────────────────────────

class SP800108Engine:
    """
    Stateful SP 800-108 KDF engine for deriving multiple keys
    from a single master key, with usage tracking.
    """

    def __init__(self, master_key: bytes, mode: str = "counter"):
        """
        Initialize engine.

        Parameters
        ----------
        master_key : bytes
            Key-derivation key (KI).
        mode : str
            KDF mode: "counter", "feedback", or "double_pipeline".
        """
        if mode not in ("counter", "feedback", "double_pipeline"):
            raise ValueError(f"Unknown mode: {mode}")

        self.master_key = master_key
        self.mode = mode
        self.derivation_log: List[Dict] = []

    def derive(
        self, label: str, context: bytes = b"", length_bytes: int = 32
    ) -> bytes:
        """
        Derive a key for the given label and context.

        Parameters
        ----------
        label : str
            Purpose identifier.
        context : bytes
            Additional context bytes.
        length_bytes : int
            Desired output length.

        Returns
        -------
        bytes
            Derived key.
        """
        label_bytes = label.encode("utf-8")

        if self.mode == "counter":
            key = kdf_counter_mode(self.master_key, label_bytes, context, length_bytes)
        elif self.mode == "feedback":
            key = kdf_feedback_mode(self.master_key, label_bytes, context, length_bytes)
        else:
            key = kdf_double_pipeline_mode(self.master_key, label_bytes, context, length_bytes)

        self.derivation_log.append(
            {"label": label, "context_len": len(context), "length": length_bytes}
        )
        return key

    def stats(self) -> Dict:
        """Return engine statistics."""
        return {
            "mode": self.mode,
            "total_derivations": len(self.derivation_log),
            "labels": [entry["label"] for entry in self.derivation_log],
        }


# ──────────────────────────────────────────────
# 7.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("kdf_sp800_108.py  —  self-test")
    print("=" * 55)

    ki = b"master_key_for_sp800_108_kdf_test"
    label = b"aes_session_key"
    context = b"axi_channel_0"

    # Test 1: Counter mode basic
    print("\n[ Counter mode KDF ]")
    k1 = kdf_counter_mode(ki, label, context, 32)
    assert len(k1) == 32
    print(f"  ✓ 32-byte key: {k1.hex()[:32]}...")

    # Test 2: Counter mode multi-block (longer than PRF output)
    print("\n[ Counter mode — multi-block output ]")
    k_long = kdf_counter_mode(ki, label, context, 96)
    assert len(k_long) == 96
    print(f"  ✓ 96-byte key (3 PRF blocks)")

    # Test 3: Determinism
    print("\n[ Determinism ]")
    k1b = kdf_counter_mode(ki, label, context, 32)
    assert k1 == k1b
    print(f"  ✓ Same inputs → same output")

    # Test 4: Different labels → different keys
    print("\n[ Label-based domain separation ]")
    k_a = kdf_counter_mode(ki, b"key_a", context, 32)
    k_b = kdf_counter_mode(ki, b"key_b", context, 32)
    assert k_a != k_b
    print(f"  ✓ Different labels produce different keys")

    # Test 5: Different context → different keys
    print("\n[ Context-based domain separation ]")
    k_ctx1 = kdf_counter_mode(ki, label, b"channel_0", 32)
    k_ctx2 = kdf_counter_mode(ki, label, b"channel_1", 32)
    assert k_ctx1 != k_ctx2
    print(f"  ✓ Different contexts produce different keys")

    # Test 6: Feedback mode
    print("\n[ Feedback mode KDF ]")
    # Note: with empty IV, feedback mode's first block equals counter mode's
    # first block (K(0)="" feeds identically into the first PRF call). Use a
    # non-empty IV to get an independent output stream, as recommended by
    # SP 800-108 §5.2 for practical deployments.
    fb_iv = b"\x42" * 32
    k_fb = kdf_feedback_mode(ki, label, context, 32, iv=fb_iv)
    assert len(k_fb) == 32
    assert k_fb != k1  # distinct IV → different output than counter mode
    print(f"  ✓ Feedback mode (with IV): 32-byte key")

    # Test 7: Feedback mode with IV
    print("\n[ Feedback mode with custom IV ]")
    iv = b"\x01" * 32
    k_fb_iv = kdf_feedback_mode(ki, label, context, 32, iv=iv)
    k_fb_no_iv = kdf_feedback_mode(ki, label, context, 32)
    assert k_fb_iv != k_fb_no_iv
    print(f"  ✓ IV changes output as expected")

    # Test 8: Double-pipeline mode
    print("\n[ Double-pipeline mode KDF ]")
    k_dp = kdf_double_pipeline_mode(ki, label, context, 32)
    assert len(k_dp) == 32
    assert k_dp != k1
    assert k_dp != k_fb
    print(f"  ✓ Double-pipeline mode: 32-byte key, distinct from other modes")

    # Test 9: SP800108Engine — multi-key derivation
    print("\n[ SP800108Engine multi-key derivation ]")
    engine = SP800108Engine(ki, mode="counter")
    keys = {
        "axi_channel_0": engine.derive("axi_enc", b"ch0", 32),
        "axi_channel_1": engine.derive("axi_enc", b"ch1", 32),
        "kyber_session": engine.derive("kyber_sess", b"", 32),
        "dilithium_op":  engine.derive("dilithium_op", b"", 32),
    }
    assert len(set(keys.values())) == 4  # all unique
    print(f"  ✓ Derived 4 unique keys from one master key")

    # Test 10: Mode comparison via engine
    print("\n[ Mode comparison ]")
    for mode in ["counter", "feedback", "double_pipeline"]:
        eng = SP800108Engine(ki, mode=mode)
        k = eng.derive("test_label", b"test_ctx", 32)
        print(f"  ✓ {mode}: {k.hex()[:24]}...")

    # Test 11: Statistics
    print("\n[ Engine statistics ]")
    stats = engine.stats()
    assert stats["total_derivations"] == 4
    print(f"  Mode: {stats['mode']}, Total: {stats['total_derivations']}")
    print(f"  Labels: {stats['labels']}")
    print(f"  ✓ Statistics tracked correctly")

    # Test 12: AXI per-channel key derivation (use case)
    print("\n[ AXI per-channel encryption keys ]")
    axi_master_key = hashlib.sha256(b"axi_root_key").digest()
    engine_axi = SP800108Engine(axi_master_key, mode="counter")
    channel_keys = [engine_axi.derive("axi_aes", f"ch{i}".encode(), 32) for i in range(4)]
    assert len(set(channel_keys)) == 4
    print(f"  ✓ Derived 4 independent AXI channel keys")

    print("\n  All checks passed.\n")
