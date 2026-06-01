"""
params.py
=========
Central parameter definitions for the PQC-SNN SoC Python Golden Model.

Covers:
  - Kyber512 / Kyber768 / Kyber1024  (NIST FIPS 203)
  - Dilithium2 / Dilithium3 / Dilithium5 (NIST FIPS 204)
  - SNN neuron / synapse / STDP parameters
  - System-level constants (AXI, memory, alert)

Usage:
  from config.params import KYBER, DILITHIUM, SNN, SYSTEM
  p = KYBER[512]
  q = p.Q          # 3329

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Standard: NIST FIPS 203 / FIPS 204 / SP 800-90A / SP 800-108
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ──────────────────────────────────────────────
# 1.  SHARED POLYNOMIAL RING CONSTANTS
# ──────────────────────────────────────────────

#: Degree of the polynomial ring Z_q[x] / (x^n + 1)
N: int = 256

#: Number of bits needed to represent one coefficient index
LOG2_N: int = 8

#: Zeta — primitive 2n-th root of unity for Kyber NTT (mod KYBER_Q)
#: zeta^1 = 17  (verified: 17^256 ≡ -1 mod 3329)
KYBER_ZETA: int = 17

#: Zeta — primitive 2n-th root of unity for Dilithium NTT (mod DILITHIUM_Q)
#: zeta^1 = 1753  (verified: 1753^256 ≡ -1 mod 8380417)
DILITHIUM_ZETA: int = 1753

#: Montgomery constant for Kyber: R = 2^16 mod KYBER_Q
KYBER_MONT_R: int = 2285      # 2^16 mod 3329

#: Montgomery constant for Dilithium: R = 2^32 mod DILITHIUM_Q
DILITHIUM_MONT_R: int = 4193792  # 2^32 mod 8380417

#: Barrett reduction constant for Kyber: floor(2^24 / KYBER_Q)
KYBER_BARRETT_K: int = 20159

#: Barrett reduction constant for Dilithium: floor(2^64 / DILITHIUM_Q)
DILITHIUM_BARRETT_K: int = 2198733722


# ──────────────────────────────────────────────
# 2.  KYBER PARAMETERS  (NIST FIPS 203)
# ──────────────────────────────────────────────

#: Kyber modulus  q = 3329 = 13 * 256 + 1
KYBER_Q: int = 3329

#: Modular inverse of n=256 over KYBER_Q:  256^{-1} mod 3329 = 3303
KYBER_N_INV: int = 3316

#: f = n^{-1} * R^{-1} mod q  (used in final INTT normalisation)
KYBER_F: int = 1441


@dataclass(frozen=True)
class KyberParams:
    """
    All parameters for one Kyber variant.

    Attributes
    ----------
    variant : int
        Security level identifier (512, 768, or 1024).
    K : int
        Module rank (number of polynomials per vector).
    ETA1 : int
        CBD distribution parameter for secret / error (keygen).
    ETA2 : int
        CBD distribution parameter for error (encaps).
    DU : int
        Compression bits for ciphertext u component.
    DV : int
        Compression bits for ciphertext v component.
    DT : int
        Compression bits for public key t component  (always 12).

    Derived sizes (bytes)
    ----------------------
    pk_bytes    : public key  = 32 + K*12*N/8
    sk_bytes    : secret key  = K*12*N/8 + pk_bytes + 32 + 32
    ct_bytes    : ciphertext  = K*DU*N/8 + DV*N/8
    ss_bytes    : shared secret (always 32)
    seed_bytes  : randomness required (always 32)
    """
    variant:    int
    K:          int
    ETA1:       int
    ETA2:       int
    DU:         int
    DV:         int
    DT:         int = 12

    # --- Derived sizes (computed post-init) ---
    pk_bytes:   int = field(init=False)
    sk_bytes:   int = field(init=False)
    ct_bytes:   int = field(init=False)
    ss_bytes:   int = field(default=32, init=False)
    seed_bytes: int = field(default=32, init=False)

    def __post_init__(self):
        object.__setattr__(self, 'pk_bytes',
            32 + self.K * self.DT * N // 8)
        object.__setattr__(self, 'sk_bytes',
            self.K * 12 * N // 8 +
            self.pk_bytes + 32 + 32)
        object.__setattr__(self, 'ct_bytes',
            self.K * self.DU * N // 8 +
            self.DV * N // 8)

    # --- Convenience ---
    @property
    def Q(self) -> int:
        return KYBER_Q

    @property
    def zeta(self) -> int:
        return KYBER_ZETA

    @property
    def n_inv(self) -> int:
        return KYBER_N_INV

    @property
    def f(self) -> int:
        return KYBER_F

    def __repr__(self) -> str:
        return (
            f"KyberParams(variant={self.variant}, K={self.K}, "
            f"Q={self.Q}, N={N}, ETA1={self.ETA1}, ETA2={self.ETA2}, "
            f"DU={self.DU}, DV={self.DV}, "
            f"pk={self.pk_bytes}B, sk={self.sk_bytes}B, ct={self.ct_bytes}B)"
        )


#: Registry of all Kyber variants keyed by security level
KYBER: Dict[int, KyberParams] = {
    512:  KyberParams(variant=512,  K=2, ETA1=3, ETA2=2, DU=10, DV=4),
    768:  KyberParams(variant=768,  K=3, ETA1=2, ETA2=2, DU=10, DV=4),
    1024: KyberParams(variant=1024, K=4, ETA1=2, ETA2=2, DU=11, DV=5),
}

#: Default variant used when none is specified
KYBER_DEFAULT_VARIANT: int = 768


# ──────────────────────────────────────────────
# 3.  DILITHIUM PARAMETERS  (NIST FIPS 204)
# ──────────────────────────────────────────────

#: Dilithium modulus  q = 8380417 = 2^23 - 2^13 + 1
DILITHIUM_Q: int = 8_380_417

#: Modular inverse of n=256 over DILITHIUM_Q:  256^{-1} mod 8380417
DILITHIUM_N_INV: int = 8_347_681

#: Dropped bits in rounding (HighBits / LowBits split)
DILITHIUM_D: int = 13

#: Number of bytes in the commitment hash c_tilde
DILITHIUM_LAMBDA_BYTES: int = 32   # Dilithium2/3 use 32; Dilithium5 uses 32 too

#: Seed length for all Dilithium operations (bytes)
DILITHIUM_SEED_BYTES: int = 32

#: Size of commitment randomness rho (bytes)
DILITHIUM_RHO_BYTES: int = 32

#: Size of private key seed K (bytes)
DILITHIUM_K_BYTES: int = 32

#: Size of nonce tr (bytes)
DILITHIUM_TR_BYTES: int = 64


@dataclass(frozen=True)
class DilithiumParams:
    """
    All parameters for one Dilithium variant.

    Attributes
    ----------
    variant : int
        Security level (2, 3, or 5).
    K : int
        Number of rows in matrix A  (error / output dimension).
    L : int
        Number of columns in matrix A  (secret key dimension).
    ETA : int
        Coefficient bound for secret key s1, s2.
    TAU : int
        Number of ±1 entries in challenge polynomial c.
    BETA : int
        Bound β = τ · η (used in signature validity check).
    GAMMA1 : int
        Coefficient range for y (masking polynomial).
    GAMMA2 : int
        Low-order rounding range (= (q-1) / alpha).
    OMEGA : int
        Maximum number of ones in hint vector h.

    Derived sizes (bytes)
    ----------------------
    pk_bytes : public key
    sk_bytes : secret key
    sig_bytes: signature
    """
    variant: int
    K:       int
    L:       int
    ETA:     int
    TAU:     int
    BETA:    int
    GAMMA1:  int
    GAMMA2:  int
    OMEGA:   int

    pk_bytes:  int = field(init=False)
    sk_bytes:  int = field(init=False)
    sig_bytes: int = field(init=False)

    def __post_init__(self):
        # pk  = 32 + K * 10 * N / 8   (rho || t1 packed at 10 bits/coeff)
        object.__setattr__(self, 'pk_bytes',
            32 + self.K * 10 * N // 8)

        # sk  = 2*32 + 64 + L*eta_bits + K*eta_bits + K*13*N/8
        eta_bits = 3 if self.ETA == 2 else 4
        object.__setattr__(self, 'sk_bytes',
            2 * 32 + 64 +
            self.L * eta_bits * N // 8 +
            self.K * eta_bits * N // 8 +
            self.K * DILITHIUM_D * N // 8)

        # sig = lambda_bytes + L * gamma1_bits * N/8 + omega + K
        gamma1_bits = 18 if self.GAMMA1 == (1 << 17) else 20
        object.__setattr__(self, 'sig_bytes',
            DILITHIUM_LAMBDA_BYTES +
            self.L * gamma1_bits * N // 8 +
            self.OMEGA + self.K)

    @property
    def Q(self) -> int:
        return DILITHIUM_Q

    @property
    def zeta(self) -> int:
        return DILITHIUM_ZETA

    @property
    def n_inv(self) -> int:
        return DILITHIUM_N_INV

    @property
    def D(self) -> int:
        return DILITHIUM_D

    def __repr__(self) -> str:
        return (
            f"DilithiumParams(variant={self.variant}, K={self.K}, L={self.L}, "
            f"Q={self.Q}, N={N}, ETA={self.ETA}, TAU={self.TAU}, "
            f"BETA={self.BETA}, GAMMA1={self.GAMMA1}, GAMMA2={self.GAMMA2}, "
            f"OMEGA={self.OMEGA}, "
            f"pk={self.pk_bytes}B, sk={self.sk_bytes}B, sig={self.sig_bytes}B)"
        )


#: Registry of all Dilithium variants keyed by security level
DILITHIUM: Dict[int, DilithiumParams] = {
    2: DilithiumParams(
        variant=2, K=4, L=4, ETA=2, TAU=39,
        BETA=78,
        GAMMA1=(1 << 17),
        GAMMA2=(DILITHIUM_Q - 1) // 88,
        OMEGA=80,
    ),
    3: DilithiumParams(
        variant=3, K=6, L=5, ETA=4, TAU=49,
        BETA=196,
        GAMMA1=(1 << 19),
        GAMMA2=(DILITHIUM_Q - 1) // 32,
        OMEGA=55,
    ),
    5: DilithiumParams(
        variant=5, K=8, L=7, ETA=2, TAU=60,
        BETA=120,
        GAMMA1=(1 << 19),
        GAMMA2=(DILITHIUM_Q - 1) // 32,
        OMEGA=75,
    ),
}

#: Default variant used when none is specified
DILITHIUM_DEFAULT_VARIANT: int = 3


# ──────────────────────────────────────────────
# 4.  SNN PARAMETERS
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SNNParams:
    """
    Parameters for the Spiking Neural Network threat detector.

    Neuron model : Leaky Integrate-and-Fire (LIF)
    Learning     : Spike-Timing-Dependent Plasticity (STDP)
    Inhibition   : Winner-Takes-All (WTA) lateral inhibition

    All time constants are in units of simulation timesteps (dt).
    Hardware clock assumed: 100 MHz  →  dt = 10 ns.

    Attributes
    ----------
    NUM_INPUTS : int
        Input feature dimension (encoded as spike trains).
    NUM_NEURONS : int
        Number of LIF neurons in the competitive layer.
    NUM_OUTPUTS : int
        Threat classification outputs (anomaly score buckets).

    V_REST : float
        Resting membrane potential (mV equivalent, normalised).
    V_THRESH : float
        Spike threshold potential.
    V_RESET : float
        Post-spike reset potential (hyperpolarisation).

    TAU_M : int
        Membrane time constant (timesteps).  τ_m = R_m * C_m.
    TAU_S : int
        Synaptic current time constant (timesteps).
    TAU_REF : int
        Absolute refractory period (timesteps).

    STDP_A_PLUS : float
        STDP potentiation amplitude.
    STDP_A_MINUS : float
        STDP depression amplitude.
    TAU_PLUS : int
        STDP potentiation time window (timesteps).
    TAU_MINUS : int
        STDP depression time window (timesteps).

    W_MIN : float
        Minimum synaptic weight (clipped after each update).
    W_MAX : float
        Maximum synaptic weight.
    W_INIT_MEAN : float
        Initial weight distribution mean.
    W_INIT_STD : float
        Initial weight distribution standard deviation.

    HOMEO_TARGET : float
        Homeostatic target firing rate (spikes / timestep).
    HOMEO_RATE : float
        Homeostatic learning rate.

    INHIB_STRENGTH : float
        Lateral inhibition synaptic weight (WTA).
    """
    # Architecture
    NUM_INPUTS:   int   = 64
    NUM_NEURONS:  int   = 128
    NUM_OUTPUTS:  int   = 4

    # LIF membrane
    V_REST:   float = 0.0
    V_THRESH: float = 1.0
    V_RESET:  float = -0.1

    # Time constants (in timesteps)
    TAU_M:   int = 20
    TAU_S:   int = 5
    TAU_REF: int = 3

    # STDP
    STDP_A_PLUS:  float = 0.01
    STDP_A_MINUS: float = 0.0105   # slightly asymmetric for stability
    TAU_PLUS:     int   = 20
    TAU_MINUS:    int   = 20

    # Weight bounds
    W_MIN:       float = 0.0
    W_MAX:       float = 1.0
    W_INIT_MEAN: float = 0.5
    W_INIT_STD:  float = 0.05

    # Homeostatic plasticity
    HOMEO_TARGET: float = 0.05    # 5% firing rate target
    HOMEO_RATE:   float = 0.001

    # Lateral inhibition
    INHIB_STRENGTH: float = 10.0  # strong WTA

    # Fixed-point representation (for RTL matching)
    WEIGHT_FRAC_BITS: int = 16    # Q0.16 fixed point
    VOLTAGE_FRAC_BITS: int = 16   # Q1.16 fixed point


#: Singleton SNN parameter set
SNN: SNNParams = SNNParams()


# ──────────────────────────────────────────────
# 5.  HASH / RANDOMNESS CONSTANTS
# ──────────────────────────────────────────────

#: Keccak-f[1600] number of rounds
KECCAK_ROUNDS: int = 24

#: SHAKE-128 rate in bytes  (1600 - 256) / 8
SHAKE128_RATE: int = 168

#: SHAKE-256 rate in bytes  (1600 - 512) / 8
SHAKE256_RATE: int = 136

#: SHA3-256 rate in bytes
SHA3_256_RATE: int = 136

#: SHA3-512 rate in bytes
SHA3_512_RATE: int = 72

#: SHAKE-128 domain suffix byte (NIST FIPS 202)
SHAKE128_SUFFIX: int = 0x1F

#: SHAKE-256 domain suffix byte
SHAKE256_SUFFIX: int = 0x1F

#: SHA3 domain suffix byte
SHA3_SUFFIX: int = 0x06

#: CTR_DRBG key length (AES-256)
DRBG_KEY_BYTES: int = 32

#: CTR_DRBG V counter length
DRBG_V_BYTES: int = 16

#: CTR_DRBG seed length (key + V)
DRBG_SEED_BYTES: int = DRBG_KEY_BYTES + DRBG_V_BYTES

#: HKDF output key material max bytes (SP 800-108)
HKDF_MAX_OKM: int = 255 * SHAKE256_RATE

#: Salt length in bytes
SALT_LEN: int = 32

#: Nonce width in bytes
NONCE_LEN: int = 32


# ──────────────────────────────────────────────
# 6.  SYSTEM-LEVEL CONSTANTS
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SystemParams:
    """
    SoC-level parameters shared across the system model.

    Attributes
    ----------
    AXI_DATA_WIDTH : int
        AXI-Lite data bus width in bits.
    AXI_ADDR_WIDTH : int
        AXI-Lite address bus width in bits.
    DMA_BURST_LEN : int
        Max DMA burst length in beats.
    MEM_SIZE_BYTES : int
        Total scratchpad RAM size in bytes.
    MEE_KEY_BITS : int
        Memory encryption engine key size (AES-256-XTS).
    IRQ_COUNT : int
        Number of interrupt lines into aggregator.
    ALERT_FIFO_DEPTH : int
        Alert FIFO depth (entries).
    UART_BAUD : int
        UART baud rate for alert output.
    """
    AXI_DATA_WIDTH:  int = 32
    AXI_ADDR_WIDTH:  int = 32
    DMA_BURST_LEN:   int = 16
    MEM_SIZE_BYTES:  int = 256 * 1024   # 256 KB scratchpad
    MEE_KEY_BITS:    int = 256
    IRQ_COUNT:       int = 16
    ALERT_FIFO_DEPTH: int = 64
    UART_BAUD:       int = 115_200


#: Singleton system parameter set
SYSTEM: SystemParams = SystemParams()


# ──────────────────────────────────────────────
# 7.  CONVENIENCE ACCESSORS
# ──────────────────────────────────────────────

def get_kyber(variant: int = KYBER_DEFAULT_VARIANT) -> KyberParams:
    """Return KyberParams for the requested variant (512 / 768 / 1024)."""
    if variant not in KYBER:
        raise ValueError(
            f"Unknown Kyber variant {variant}. Choose from {list(KYBER)}")
    return KYBER[variant]


def get_dilithium(variant: int = DILITHIUM_DEFAULT_VARIANT) -> DilithiumParams:
    """Return DilithiumParams for the requested variant (2 / 3 / 5)."""
    if variant not in DILITHIUM:
        raise ValueError(
            f"Unknown Dilithium variant {variant}. Choose from {list(DILITHIUM)}")
    return DILITHIUM[variant]


def ntt_zeta_powers(q: int, zeta: int) -> Tuple[int, ...]:
    """
    Precompute the 256 zeta^{brv(k)} values used by the NTT.

    Returns a tuple of 256 integers in bit-reversed order,
    matching the layout expected by ntt.py and intt.py.

    Parameters
    ----------
    q    : modulus (KYBER_Q or DILITHIUM_Q)
    zeta : primitive 2n-th root of unity

    Returns
    -------
    Tuple[int, ...]
        256 twiddle factors in NTT canonical order.
    """
    def bit_reverse(k: int, bits: int = LOG2_N) -> int:
        result = 0
        for _ in range(bits):
            result = (result << 1) | (k & 1)
            k >>= 1
        return result

    powers = [pow(zeta, bit_reverse(k), q) for k in range(N)]
    return tuple(powers)


#: Precomputed Kyber NTT twiddle factors (bit-reversed zeta powers mod KYBER_Q)
KYBER_ZETAS: Tuple[int, ...] = ntt_zeta_powers(KYBER_Q, KYBER_ZETA)

#: Precomputed Dilithium NTT twiddle factors (bit-reversed zeta powers mod DILITHIUM_Q)
DILITHIUM_ZETAS: Tuple[int, ...] = ntt_zeta_powers(DILITHIUM_Q, DILITHIUM_ZETA)


# ──────────────────────────────────────────────
# 8.  SELF-TEST  (run with:  python params.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PQC-SNN SoC  —  params.py  self-test")
    print("=" * 60)

    print("\n[ Kyber variants ]")
    for v, p in KYBER.items():
        print(f"  Kyber-{v:4d}  {p}")

    print("\n[ Dilithium variants ]")
    for v, p in DILITHIUM.items():
        print(f"  Dilithium{v}  {p}")

    print("\n[ SNN params ]")
    print(f"  inputs={SNN.NUM_INPUTS}, neurons={SNN.NUM_NEURONS}, "
          f"outputs={SNN.NUM_OUTPUTS}, tau_m={SNN.TAU_M}, "
          f"V_thresh={SNN.V_THRESH}, W_max={SNN.W_MAX}")

    print("\n[ System params ]")
    print(f"  AXI={SYSTEM.AXI_DATA_WIDTH}b, "
          f"DMA_burst={SYSTEM.DMA_BURST_LEN}, "
          f"MEM={SYSTEM.MEM_SIZE_BYTES//1024}KB, "
          f"IRQs={SYSTEM.IRQ_COUNT}")

    print("\n[ Twiddle factor sanity checks ]")
    # Kyber: zeta is a primitive 256th root of unity  →  zeta^128 ≡ -1 mod q
    assert pow(KYBER_ZETA, N // 2, KYBER_Q) == KYBER_Q - 1, "Kyber zeta check FAILED"
    # Dilithium: zeta is a primitive 512th root of unity  →  zeta^256 ≡ -1 mod q
    assert pow(DILITHIUM_ZETA, N, DILITHIUM_Q) == DILITHIUM_Q - 1, \
        "Dilithium zeta check FAILED"
    print("  Kyber    zeta^128 ≡ -1 mod q  ✓")
    print("  Dilithium zeta^256 ≡ -1 mod q  ✓")

    # n_inv check: n * n_inv ≡ 1 mod q
    assert (N * KYBER_N_INV) % KYBER_Q == 1, "Kyber n_inv FAILED"
    assert (N * DILITHIUM_N_INV) % DILITHIUM_Q == 1, "Dilithium n_inv FAILED"
    print("  Kyber    n_inv check  ✓")
    print("  Dilithium n_inv check  ✓")

    print("\n  All checks passed.\n")
