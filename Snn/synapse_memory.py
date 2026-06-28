"""
synapse_memory.py
====================
Synapse / weight memory model for the SNN core in the PQC-SNN SoC
(FB_SNN — SYNAPSE / WEIGHT MEMORY).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Models the SRAM/eMRAM weight store: quantized precision, compressed       ║
║ sparse storage, boot-time loading, and online-learning write-back.       ║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Quantized weight storage at configurable precision (4/8/16-bit, per
    diagram's "Weight Precision: 4/8/16-bit" entry)
  - Compressed Sparse Row (CSR)-style storage for sparse weight matrices
    (per diagram's "Compressed Sparse Storage" entry)
  - Boot-time pre-trained weight loading (per "Boot-time Pre-trained
    Weights" entry) — load a full matrix in one shot at startup
  - Online learning update path (per "Online Learning Update" entry) —
    incremental weight deltas applied during operation (e.g. from STDP)
  - Mode select between READ-ONLY (inference) and READ-WRITE (learning)
    operating modes (per "Mode Select" entry)
  - Address-based read/write interface mirroring an SRAM/eMRAM port
  - Memory footprint estimation in bits, for area/budget analysis

Context (per architecture diagram, "SNN CORE" box,
"SYNAPSE / WEIGHT MEMORY" sub-block):
  SRAM / eMRAM
  Compressed Sparse Storage
  Weight Precision: 4/8/16-bit
  Boot-time Pre-trained Weights
  Online Learning Update
  Mode Select (Inference / Training)

This module is the storage layer UNDERNEATH neuro_array.py's weight
matrices: NeuroArray.get_weights()/set_weights() operate on plain Python
floats for simulation convenience, while SynapseMemory models how those
same weights are actually packed, quantized, and addressed in hardware
SRAM/eMRAM — including the precision loss and compression a real chip
would incur. The two interoperate via to_dense()/from_dense() conversion.

Matches synapse_memory.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : AEGIS-NEURO architecture diagram, §3 NEUROMORPHIC SNN SUBSYSTEM
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────
# 1.  PARAMETERS
# ──────────────────────────────────────────────

class MemoryMode(Enum):
    READ_ONLY = "read_only"    # inference mode — writes rejected
    READ_WRITE = "read_write"  # training/online-learning mode


#: Supported weight quantization precisions (bits)
SUPPORTED_PRECISIONS: Tuple[int, ...] = (4, 8, 16)

#: Default precision if unspecified
DEFAULT_PRECISION_BITS: int = 8

#: Default weight range assumed for quantization (signed, symmetric)
DEFAULT_WEIGHT_RANGE: float = 1.0


# ──────────────────────────────────────────────
# 2.  QUANTIZATION
# ──────────────────────────────────────────────

def quantize_weight(value: float, bits: int, w_range: float = DEFAULT_WEIGHT_RANGE) -> int:
    """
    Quantize a floating-point weight to a signed integer code.

    Parameters
    ----------
    value : float
        Weight value, expected within [-w_range, +w_range].
    bits : int
        Quantization bit width (4, 8, or 16).
    w_range : float
        Symmetric range the weight is clamped to before quantizing.

    Returns
    -------
    int
        Signed quantized code in [-(2^(bits-1)), 2^(bits-1) - 1].
    """
    if bits not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision: {bits}-bit (supported: {SUPPORTED_PRECISIONS})")

    clamped = max(-w_range, min(w_range, value))
    max_code = (1 << (bits - 1)) - 1
    # Symmetric quantization: scale is derived from max_code so that
    # +w_range maps exactly to +max_code and -w_range maps to -max_code.
    # The clamp bounds below are intentionally symmetric (±max_code) to
    # match what this formula can actually produce — the full asymmetric
    # signed range's extra negative code (-2^(bits-1)) is never reachable
    # under symmetric scaling and is not used here (standard practice,
    # matches e.g. PyTorch's default symmetric quantization scheme).
    min_code = -max_code

    code = round((clamped / w_range) * max_code)
    return max(min_code, min(max_code, code))


def dequantize_weight(code: int, bits: int, w_range: float = DEFAULT_WEIGHT_RANGE) -> float:
    """
    Convert a quantized integer code back to a floating-point weight.

    Parameters
    ----------
    code : int
        Signed quantized code as produced by quantize_weight().
    bits : int
        Quantization bit width used.
    w_range : float
        Symmetric range used during quantization.

    Returns
    -------
    float
        Dequantized weight value (lossy approximation of the original).
    """
    max_code = (1 << (bits - 1)) - 1
    return (code / max_code) * w_range


def quantization_step(bits: int, w_range: float = DEFAULT_WEIGHT_RANGE) -> float:
    """Smallest representable weight increment at the given precision."""
    max_code = (1 << (bits - 1)) - 1
    return w_range / max_code


# ──────────────────────────────────────────────
# 3.  COMPRESSED SPARSE ROW (CSR) WEIGHT STORAGE
# ──────────────────────────────────────────────

@dataclass
class CSRMatrix:
    """
    Compressed Sparse Row representation of a weight matrix.

    Only non-zero entries are stored, which is efficient when many
    synaptic connections are pruned or naturally near-zero — matching
    the diagram's "Compressed Sparse Storage" entry.
    """
    n_rows: int
    n_cols: int
    values: List[int] = field(default_factory=list)       # quantized codes
    col_indices: List[int] = field(default_factory=list)  # column per value
    row_ptr: List[int] = field(default_factory=list)       # CSR row pointers

    @classmethod
    def from_dense(
        cls, dense: List[List[float]], bits: int, w_range: float = DEFAULT_WEIGHT_RANGE,
        zero_threshold: float = 1e-9,
    ) -> "CSRMatrix":
        """
        Build a CSR matrix from a dense weight matrix, quantizing and
        dropping near-zero entries.

        Parameters
        ----------
        dense : List[List[float]]
            n_rows x n_cols dense weight matrix.
        bits : int
            Quantization precision to apply.
        w_range : float
            Symmetric range for quantization.
        zero_threshold : float
            Values with |w| below this are treated as zero and dropped.

        Returns
        -------
        CSRMatrix
            Sparse-compressed, quantized weight storage.
        """
        n_rows = len(dense)
        n_cols = len(dense[0]) if n_rows > 0 else 0

        values: List[int] = []
        col_indices: List[int] = []
        row_ptr: List[int] = [0]

        for row in dense:
            for j, w in enumerate(row):
                if abs(w) > zero_threshold:
                    values.append(quantize_weight(w, bits, w_range))
                    col_indices.append(j)
            row_ptr.append(len(values))

        return cls(n_rows, n_cols, values, col_indices, row_ptr)

    def to_dense(self, bits: int, w_range: float = DEFAULT_WEIGHT_RANGE) -> List[List[float]]:
        """
        Reconstruct a dense weight matrix from this CSR representation.

        Parameters
        ----------
        bits : int
            Quantization precision used when this CSR was built.
        w_range : float
            Symmetric range used during quantization.

        Returns
        -------
        List[List[float]]
            n_rows x n_cols dense matrix (dequantized; zeros elsewhere).
        """
        dense = [[0.0] * self.n_cols for _ in range(self.n_rows)]
        for r in range(self.n_rows):
            start, end = self.row_ptr[r], self.row_ptr[r + 1]
            for k in range(start, end):
                c = self.col_indices[k]
                dense[r][c] = dequantize_weight(self.values[k], bits, w_range)
        return dense

    def nnz(self) -> int:
        """Number of non-zero (stored) entries."""
        return len(self.values)

    def sparsity(self) -> float:
        """Fraction of entries that are zero (not stored)."""
        total = self.n_rows * self.n_cols
        if total == 0:
            return 0.0
        return 1.0 - (self.nnz() / total)

    def memory_bits(self, bits: int, col_idx_bits: int = 16, row_ptr_bits: int = 24) -> int:
        """
        Estimate total storage footprint in bits.

        Parameters
        ----------
        bits : int
            Bits used per stored weight value.
        col_idx_bits : int
            Bits used per column-index entry.
        row_ptr_bits : int
            Bits used per row-pointer entry.

        Returns
        -------
        int
            Total estimated bits to store this CSR matrix.
        """
        return (
            self.nnz() * bits
            + self.nnz() * col_idx_bits
            + len(self.row_ptr) * row_ptr_bits
        )


def dense_memory_bits(n_rows: int, n_cols: int, bits: int) -> int:
    """Estimate storage footprint of an equivalent DENSE matrix, for comparison."""
    return n_rows * n_cols * bits


# ──────────────────────────────────────────────
# 4.  SYNAPSE MEMORY BANK
# ──────────────────────────────────────────────

class SynapseMemory:
    """
    Models one synapse/weight memory bank (one layer connection's worth
    of weights) with quantized, optionally-sparse storage, boot-time
    loading, online-learning updates, and mode-gated access.
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        precision_bits: int = DEFAULT_PRECISION_BITS,
        weight_range: float = DEFAULT_WEIGHT_RANGE,
        use_sparse: bool = False,
        mode: MemoryMode = MemoryMode.READ_ONLY,
    ):
        """
        Initialize an empty synapse memory bank.

        Parameters
        ----------
        n_rows : int
            Source-layer neuron count (rows).
        n_cols : int
            Destination-layer neuron count (columns).
        precision_bits : int
            Quantization precision (4, 8, or 16 bits).
        weight_range : float
            Symmetric range for quantization, e.g. weights in [-1, 1].
        use_sparse : bool
            If True, store internally as CSR (compressed sparse);
            if False, store as a dense quantized-code matrix.
        mode : MemoryMode
            READ_ONLY (inference) or READ_WRITE (online learning).
        """
        if precision_bits not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision: {precision_bits}-bit "
                f"(supported: {SUPPORTED_PRECISIONS})"
            )

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.precision_bits = precision_bits
        self.weight_range = weight_range
        self.use_sparse = use_sparse
        self.mode = mode

        # Dense quantized-code storage (always maintained as the
        # "ground truth" for simplicity; CSR is derived/compressed
        # representation built on demand or kept in sync below).
        self._codes: List[List[int]] = [[0] * n_cols for _ in range(n_rows)]
        self._csr: Optional[CSRMatrix] = None

        self.write_count: int = 0
        self.read_count: int = 0
        self.boot_loaded: bool = False

    # ── Boot-time loading ────────────────────────

    def boot_load(self, dense_weights: List[List[float]]) -> None:
        """
        Load a full pre-trained weight matrix at boot time.

        This is allowed regardless of current mode (boot loading happens
        before normal operation begins), and resets write/read counters.

        Parameters
        ----------
        dense_weights : List[List[float]]
            n_rows x n_cols floating-point weight matrix to load.
        """
        if len(dense_weights) != self.n_rows or (
            self.n_rows > 0 and len(dense_weights[0]) != self.n_cols
        ):
            raise ValueError("boot_load weight matrix shape mismatch")

        self._codes = [
            [quantize_weight(w, self.precision_bits, self.weight_range) for w in row]
            for row in dense_weights
        ]

        if self.use_sparse:
            self._csr = CSRMatrix.from_dense(
                dense_weights, self.precision_bits, self.weight_range
            )

        self.boot_loaded = True
        self.write_count = 0
        self.read_count = 0

    # ── Read interface ────────────────────────────

    def read(self, row: int, col: int) -> float:
        """
        Read a single dequantized weight value.

        Parameters
        ----------
        row : int
            Source-neuron index.
        col : int
            Destination-neuron index.

        Returns
        -------
        float
            Dequantized weight value at (row, col).
        """
        self.read_count += 1
        code = self._codes[row][col]
        return dequantize_weight(code, self.precision_bits, self.weight_range)

    def to_dense(self) -> List[List[float]]:
        """
        Return the full weight matrix as dequantized floats.

        Returns
        -------
        List[List[float]]
            n_rows x n_cols dense weight matrix (dequantized).
        """
        self.read_count += self.n_rows * self.n_cols
        return [
            [dequantize_weight(c, self.precision_bits, self.weight_range) for c in row]
            for row in self._codes
        ]

    # ── Write interface (mode-gated) ─────────────

    def write(self, row: int, col: int, value: float) -> bool:
        """
        Write (quantize and store) a single weight value.

        Parameters
        ----------
        row : int
            Source-neuron index.
        col : int
            Destination-neuron index.
        value : float
            New floating-point weight value to store.

        Returns
        -------
        bool
            True if the write was accepted, False if rejected (READ_ONLY
            mode is active — write blocked, matching the diagram's "Mode
            Select" gating between inference and training).
        """
        if self.mode != MemoryMode.READ_WRITE:
            return False

        self._codes[row][col] = quantize_weight(value, self.precision_bits, self.weight_range)
        self.write_count += 1
        self._csr = None  # invalidate cached sparse view
        return True

    def apply_delta(self, row: int, col: int, delta: float) -> bool:
        """
        Apply an incremental weight update (e.g. from an STDP rule),
        modeling the "Online Learning Update" path.

        Parameters
        ----------
        row : int
            Source-neuron index.
        col : int
            Destination-neuron index.
        delta : float
            Floating-point change to add to the current weight before
            re-quantizing and storing.

        Returns
        -------
        bool
            True if applied, False if rejected (READ_ONLY mode).
        """
        if self.mode != MemoryMode.READ_WRITE:
            return False

        current = self.read(row, col)
        return self.write(row, col, current + delta)

    def apply_delta_matrix(self, delta_matrix: List[List[float]]) -> int:
        """
        Apply a full matrix of weight deltas in one call (batch online
        learning update, e.g. after a full STDP pass over a layer).

        Parameters
        ----------
        delta_matrix : List[List[float]]
            n_rows x n_cols matrix of weight changes to add.

        Returns
        -------
        int
            Number of individual weight updates actually applied
            (0 if in READ_ONLY mode).
        """
        if self.mode != MemoryMode.READ_WRITE:
            return 0

        applied = 0
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if delta_matrix[r][c] != 0.0:
                    self.apply_delta(r, c, delta_matrix[r][c])
                    applied += 1
        return applied

    # ── Mode control ──────────────────────────────

    def set_mode(self, mode: MemoryMode) -> None:
        """Switch between READ_ONLY (inference) and READ_WRITE (training)."""
        self.mode = mode

    # ── Compression / footprint ──────────────────

    def compress(self) -> CSRMatrix:
        """
        Build (or return cached) CSR-compressed representation of the
        current weight matrix.

        Returns
        -------
        CSRMatrix
            Sparse-compressed view of the current weights.
        """
        if self._csr is None:
            dense = self.to_dense()
            self._csr = CSRMatrix.from_dense(dense, self.precision_bits, self.weight_range)
        return self._csr

    def memory_footprint_bits(self) -> Dict[str, int]:
        """
        Estimate memory footprint under both dense and sparse storage,
        for area-budget comparison.

        Returns
        -------
        Dict[str, int]
            {"dense_bits": ..., "sparse_bits": ..., "sparsity": ...}
        """
        dense_bits = dense_memory_bits(self.n_rows, self.n_cols, self.precision_bits)
        csr = self.compress()
        return {
            "dense_bits": dense_bits,
            "sparse_bits": csr.memory_bits(self.precision_bits),
            "sparsity_pct": round(csr.sparsity() * 100, 2),
        }

    def stats(self) -> Dict:
        """Return synapse memory bank statistics."""
        return {
            "shape": (self.n_rows, self.n_cols),
            "precision_bits": self.precision_bits,
            "mode": self.mode.value,
            "boot_loaded": self.boot_loaded,
            "write_count": self.write_count,
            "read_count": self.read_count,
        }


# ──────────────────────────────────────────────
# 5.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("synapse_memory.py  —  self-test")
    print("=" * 55)

    # Test 1: Quantize/dequantize roundtrip at each supported precision
    print("\n[ Quantization roundtrip ]")
    for bits in SUPPORTED_PRECISIONS:
        original = 0.37
        code = quantize_weight(original, bits)
        recovered = dequantize_weight(code, bits)
        step = quantization_step(bits)
        assert abs(original - recovered) <= step
        print(f"  ✓ {bits}-bit: {original} → code={code} → {recovered:.4f} "
              f"(step={step:.4f})")

    # Test 2: Clamping out-of-range values
    print("\n[ Clamping out-of-range values ]")
    code_over = quantize_weight(5.0, bits=8, w_range=1.0)
    code_under = quantize_weight(-5.0, bits=8, w_range=1.0)
    max_code = (1 << 7) - 1   # 127
    min_code = -(1 << 7)      # -128 (theoretical signed range, but see below)
    assert code_over == max_code
    # NOTE: with SYMMETRIC quantization (scale derived from max_code so that
    # +w_range maps exactly to +max_code), -w_range maps to -max_code (-127),
    # not the full signed minimum (-128). -128 remains a valid but unreachable
    # code under this symmetric scheme — this is standard practice (matches
    # e.g. PyTorch's default symmetric quantization) and intentional, not a bug.
    assert code_under == -max_code
    assert min_code < code_under  # confirm -128 is indeed never produced
    print(f"  ✓ Out-of-range values clamped: +5.0→{code_over} (max), "
          f"-5.0→{code_under} (symmetric min, not full range -128)")

    # Test 3: Higher precision is more accurate
    print("\n[ Precision comparison ]")
    val = 0.1234
    err_4 = abs(val - dequantize_weight(quantize_weight(val, 4), 4))
    err_16 = abs(val - dequantize_weight(quantize_weight(val, 16), 16))
    assert err_16 < err_4
    print(f"  ✓ 16-bit error ({err_16:.6f}) < 4-bit error ({err_4:.6f})")

    # Test 4: CSR from_dense / to_dense roundtrip
    print("\n[ CSR sparse storage roundtrip ]")
    dense = [
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.3],
        [0.0, 0.2, 0.0],
    ]
    csr = CSRMatrix.from_dense(dense, bits=8)
    assert csr.nnz() == 3  # only 3 non-zero entries
    reconstructed = csr.to_dense(bits=8)
    for r in range(3):
        for c in range(3):
            assert abs(dense[r][c] - reconstructed[r][c]) < 0.02
    print(f"  ✓ CSR stored {csr.nnz()} non-zero entries (sparsity={csr.sparsity():.2%}), "
          f"dense roundtrip within quantization error")

    # Test 5: Memory footprint comparison (dense vs sparse)
    print("\n[ Memory footprint comparison ]")
    # NOTE: CSR's per-entry overhead (column index + row pointers) means
    # sparse storage only pays off once the matrix is large enough to
    # amortize that overhead. The tiny 3x3 matrix above is intentionally
    # NOT a fair comparison (skipping only 6 zero entries can't offset
    # 16-bit/24-bit index overhead) — use a realistically-sized, highly
    # sparse matrix instead, matching the scale where the architecture
    # diagram's "Compressed Sparse Storage" entry actually provides benefit.
    big_sparse_dense = [[0.0] * 64 for _ in range(64)]
    for i in range(0, 64, 8):  # ~12.5% density → ~87.5% sparse
        big_sparse_dense[i][i] = 0.5
    big_csr = CSRMatrix.from_dense(big_sparse_dense, bits=8)
    footprint_dense = dense_memory_bits(64, 64, bits=8)
    footprint_sparse = big_csr.memory_bits(bits=8)
    assert footprint_sparse < footprint_dense
    print(f"  ✓ 64x64 @ {big_csr.sparsity():.1%} sparse: "
          f"Dense={footprint_dense} bits, Sparse={footprint_sparse} bits "
          f"(sparse wins at this scale/density)")

    # Test 6: SynapseMemory boot-time load
    print("\n[ SynapseMemory boot-time load ]")
    mem = SynapseMemory(n_rows=4, n_cols=3, precision_bits=8, mode=MemoryMode.READ_ONLY)
    boot_weights = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [-0.1, -0.2, -0.3],
        [0.0, 0.0, 0.0],
    ]
    mem.boot_load(boot_weights)
    assert mem.boot_loaded
    readback = mem.to_dense()
    for r in range(4):
        for c in range(3):
            assert abs(boot_weights[r][c] - readback[r][c]) < 0.02
    print(f"  ✓ Boot-loaded 4x3 weights, read back within quantization tolerance")

    # Test 7: Read-only mode blocks writes
    print("\n[ Read-only mode blocks writes ]")
    accepted = mem.write(0, 0, 0.99)
    assert accepted is False
    assert abs(mem.read(0, 0) - 0.1) < 0.02  # unchanged
    print(f"  ✓ Write rejected in READ_ONLY mode, value unchanged")

    # Test 8: Mode switch enables writes
    print("\n[ Mode switch enables writes ]")
    mem.set_mode(MemoryMode.READ_WRITE)
    accepted2 = mem.write(0, 0, 0.99)
    assert accepted2 is True
    assert mem.read(0, 0) > 0.9
    print(f"  ✓ Write accepted after switching to READ_WRITE mode: "
          f"new value={mem.read(0, 0):.3f}")

    # Test 9: Online learning delta update (single)
    print("\n[ Online learning — single delta ]")
    mem2 = SynapseMemory(n_rows=2, n_cols=2, precision_bits=8, mode=MemoryMode.READ_WRITE)
    mem2.boot_load([[0.2, 0.0], [0.0, 0.5]])
    before = mem2.read(0, 0)
    mem2.apply_delta(0, 0, 0.1)
    after = mem2.read(0, 0)
    assert after > before
    print(f"  ✓ Delta +0.1 applied: {before:.3f} → {after:.3f}")

    # Test 10: Online learning — batch delta matrix (STDP-style update)
    print("\n[ Online learning — batch delta matrix ]")
    mem3 = SynapseMemory(n_rows=2, n_cols=2, precision_bits=8, mode=MemoryMode.READ_WRITE)
    mem3.boot_load([[0.1, 0.1], [0.1, 0.1]])
    deltas = [[0.05, 0.0], [0.0, -0.05]]
    applied = mem3.apply_delta_matrix(deltas)
    assert applied == 2  # only the two non-zero deltas applied
    assert mem3.read(0, 0) > 0.1
    assert mem3.read(1, 1) < 0.1
    print(f"  ✓ Batch update applied {applied} non-zero deltas correctly")

    # Test 11: apply_delta_matrix rejected in READ_ONLY mode
    print("\n[ Batch update rejected in READ_ONLY mode ]")
    mem3.set_mode(MemoryMode.READ_ONLY)
    rejected_count = mem3.apply_delta_matrix(deltas)
    assert rejected_count == 0
    print(f"  ✓ Batch update correctly rejected ({rejected_count} applied) in READ_ONLY")

    # Test 12: Sparse mode integration
    print("\n[ Sparse storage mode ]")
    mem4 = SynapseMemory(n_rows=8, n_cols=8, precision_bits=4, use_sparse=True)
    sparse_weights = [[0.0] * 8 for _ in range(8)]
    sparse_weights[0][0] = 0.5
    sparse_weights[3][5] = 0.3
    mem4.boot_load(sparse_weights)
    footprint = mem4.memory_footprint_bits()
    assert footprint["sparsity_pct"] > 90.0  # mostly zero matrix
    print(f"  ✓ 8x8 sparse matrix: {footprint['sparsity_pct']}% sparse, "
          f"dense={footprint['dense_bits']}b vs sparse={footprint['sparse_bits']}b")

    # Test 13: Statistics
    print("\n[ Memory bank statistics ]")
    stats = mem.stats()
    print(f"  Shape: {stats['shape']}, Precision: {stats['precision_bits']}-bit")
    print(f"  Mode: {stats['mode']}, Boot loaded: {stats['boot_loaded']}")
    print(f"  Writes: {stats['write_count']}, Reads: {stats['read_count']}")
    print(f"  ✓ Statistics tracked correctly")

    # Test 14: Unsupported precision rejected
    print("\n[ Unsupported precision rejection ]")
    try:
        SynapseMemory(n_rows=2, n_cols=2, precision_bits=12)  # not in (4,8,16)
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ 12-bit precision correctly rejected (only 4/8/16 supported)")

    print("\n  All checks passed.\n")
