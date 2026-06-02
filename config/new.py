

from dataclasses import dataclass, field
from typing import Dict, Tuple







N: int = 256


LOG2_N: int = 8



KYBER_ZETA: int = 17



DILITHIUM_ZETA: int = 1753


KYBER_MONT_R: int = 2285      


DILITHIUM_MONT_R: int = 4193792  


KYBER_BARRETT_K: int = 20159


DILITHIUM_BARRETT_K: int = 2198733722







KYBER_Q: int = 3329


KYBER_N_INV: int = 3316


KYBER_F: int = 1441


@dataclass(frozen=True)
class KyberParams:
    
    variant:    int
    K:          int
    ETA1:       int
    ETA2:       int
    DU:         int
    DV:         int
    DT:         int = 12

    
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



KYBER: Dict[int, KyberParams] = {
    512:  KyberParams(variant=512,  K=2, ETA1=3, ETA2=2, DU=10, DV=4),
    768:  KyberParams(variant=768,  K=3, ETA1=2, ETA2=2, DU=10, DV=4),
    1024: KyberParams(variant=1024, K=4, ETA1=2, ETA2=2, DU=11, DV=5),
}


KYBER_DEFAULT_VARIANT: int = 768







DILITHIUM_Q: int = 8_380_417


DILITHIUM_N_INV: int = 8_347_681


DILITHIUM_D: int = 13


DILITHIUM_LAMBDA_BYTES: int = 32   


DILITHIUM_SEED_BYTES: int = 32


DILITHIUM_RHO_BYTES: int = 32


DILITHIUM_K_BYTES: int = 32


DILITHIUM_TR_BYTES: int = 64


@dataclass(frozen=True)
class DilithiumParams:
    
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
        
        object.__setattr__(self, 'pk_bytes',
            32 + self.K * 10 * N // 8)

        
        eta_bits = 3 if self.ETA == 2 else 4
        object.__setattr__(self, 'sk_bytes',
            2 * 32 + 64 +
            self.L * eta_bits * N // 8 +
            self.K * eta_bits * N // 8 +
            self.K * DILITHIUM_D * N // 8)

        
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


DILITHIUM_DEFAULT_VARIANT: int = 3






@dataclass(frozen=True)
class SNNParams:
    
    
    NUM_INPUTS:   int   = 64
    NUM_NEURONS:  int   = 128
    NUM_OUTPUTS:  int   = 4

    
    V_REST:   float = 0.0
    V_THRESH: float = 1.0
    V_RESET:  float = -0.1

    
    TAU_M:   int = 20
    TAU_S:   int = 5
    TAU_REF: int = 3

    
    STDP_A_PLUS:  float = 0.01
    STDP_A_MINUS: float = 0.0105   
    TAU_PLUS:     int   = 20
    TAU_MINUS:    int   = 20

    
    W_MIN:       float = 0.0
    W_MAX:       float = 1.0
    W_INIT_MEAN: float = 0.5
    W_INIT_STD:  float = 0.05

    
    HOMEO_TARGET: float = 0.05    
    HOMEO_RATE:   float = 0.001

    
    INHIB_STRENGTH: float = 10.0  

    
    WEIGHT_FRAC_BITS: int = 16    
    VOLTAGE_FRAC_BITS: int = 16   



SNN: SNNParams = SNNParams()







KECCAK_ROUNDS: int = 24


SHAKE128_RATE: int = 168


SHAKE256_RATE: int = 136


SHA3_256_RATE: int = 136


SHA3_512_RATE: int = 72


SHAKE128_SUFFIX: int = 0x1F


SHAKE256_SUFFIX: int = 0x1F


SHA3_SUFFIX: int = 0x06


DRBG_KEY_BYTES: int = 32


DRBG_V_BYTES: int = 16


DRBG_SEED_BYTES: int = DRBG_KEY_BYTES + DRBG_V_BYTES


HKDF_MAX_OKM: int = 255 * SHAKE256_RATE


SALT_LEN: int = 32


NONCE_LEN: int = 32






@dataclass(frozen=True)
class SystemParams:
    
    AXI_DATA_WIDTH:  int = 32
    AXI_ADDR_WIDTH:  int = 32
    DMA_BURST_LEN:   int = 16
    MEM_SIZE_BYTES:  int = 256 * 1024   
    MEE_KEY_BITS:    int = 256
    IRQ_COUNT:       int = 16
    ALERT_FIFO_DEPTH: int = 64
    UART_BAUD:       int = 115_200



SYSTEM: SystemParams = SystemParams()






def get_kyber(variant: int = KYBER_DEFAULT_VARIANT) -> KyberParams:
    
    if variant not in KYBER:
        raise ValueError(
            f"Unknown Kyber variant {variant}. Choose from {list(KYBER)}")
    return KYBER[variant]


def get_dilithium(variant: int = DILITHIUM_DEFAULT_VARIANT) -> DilithiumParams:
    
    if variant not in DILITHIUM:
        raise ValueError(
            f"Unknown Dilithium variant {variant}. Choose from {list(DILITHIUM)}")
    return DILITHIUM[variant]


def ntt_zeta_powers(q: int, zeta: int) -> Tuple[int, ...]:
    
    def bit_reverse(k: int, bits: int = LOG2_N) -> int:
        result = 0
        for _ in range(bits):
            result = (result << 1) | (k & 1)
            k >>= 1
        return result

    powers = [pow(zeta, bit_reverse(k), q) for k in range(N)]
    return tuple(powers)



KYBER_ZETAS: Tuple[int, ...] = ntt_zeta_powers(KYBER_Q, KYBER_ZETA)


DILITHIUM_ZETAS: Tuple[int, ...] = ntt_zeta_powers(DILITHIUM_Q, DILITHIUM_ZETA)






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
    
    assert pow(KYBER_ZETA, N // 2, KYBER_Q) == KYBER_Q - 1, "Kyber zeta check FAILED"
    
    assert pow(DILITHIUM_ZETA, N, DILITHIUM_Q) == DILITHIUM_Q - 1,        "Dilithium zeta check FAILED"
    print("  Kyber    zeta^128 ≡ -1 mod q  ✓")
    print("  Dilithium zeta^256 ≡ -1 mod q  ✓")

    
    assert (N * KYBER_N_INV) % KYBER_Q == 1, "Kyber n_inv FAILED"
    assert (N * DILITHIUM_N_INV) % DILITHIUM_Q == 1, "Dilithium n_inv FAILED"
    print("  Kyber    n_inv check  ✓")
    print("  Dilithium n_inv check  ✓")

    print("\n  All checks passed.\n")
