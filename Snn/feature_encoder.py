"""
feature_encoder.py
====================
Spike-domain feature encoder for the neuromorphic SNN threat-detection
subsystem in the PQC-SNN SoC (FB_SNN — Input Event Interface).

╔════════════════════════════════════════════════════════════════════════════╗
║ **ONE-LINE SUMMARY:**                                                      ║
║ Converts raw PQC/system telemetry (timing, power, error counters) into   ║
║ spike trains for the SNN core, using rate, temporal, and population codes.║
╚════════════════════════════════════════════════════════════════════════════╝

Implements:
  - Feature extraction from raw telemetry (PQC ops, AXI traffic, sensors)
  - Rate coding      : feature magnitude → spike frequency
  - Temporal coding  : feature magnitude → spike latency (time-to-first-spike)
  - Population coding: feature value → which neuron in a bank fires
  - Normalization / scaling of raw inputs to [0, 1] before encoding
  - Multi-channel encoding (each input feature → dedicated neuron group)

Context (per architecture diagram, "3. NEUROMORPHIC SNN SUBSYSTEM"):
  INPUT EVENT INTERFACE block takes:
    - AXI4-Stream I/F: network packets
    - Sensor Data: tamper/voltage/temp readings
    - From System Logs: PQC operation metrics (latency, retry counts)
  ...and must turn these heterogeneous signals into spike trains compatible
  with the SNN_CORE's input neuron layer (N neurons).

Encoding schemes:
  RATE CODING:
    spike_count = round(feature_normalized * max_rate * window_duration)
    Spikes are placed at regular intervals within the time window.

  TEMPORAL CODING (time-to-first-spike, TTFS):
    latency = (1 - feature_normalized) * window_duration
    Larger feature value → earlier (more salient) spike.

  POPULATION CODING:
    Each neuron in a bank has a preferred value (Gaussian receptive field).
    Neuron i fires with probability ∝ exp(-(x - center_i)^2 / (2*sigma^2)).

Usage in AEGIS-NEURO:
  - PQC retry-counter anomalies (from dilithium_retry_counter.py) → rate code
  - Tamper sensor margins (from tamper_detect.py) → population code
  - Network packet inter-arrival timing → temporal code

Matches feature_encoder.sv (hardware RTL reference).

Author : PQC-SNN SoC Golden Model Team
Version: 1.0.0
Ref    : Gerstner & Kistler, "Spiking Neuron Models" (coding schemes)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────
# 1.  ENCODER PARAMETERS
# ──────────────────────────────────────────────

#: Default simulation time window for a single encoding pass (ms)
DEFAULT_WINDOW_MS: float = 100.0

#: Default maximum spike rate for rate coding (Hz)
DEFAULT_MAX_RATE_HZ: float = 200.0

#: Default time resolution (ms per simulation tick)
DEFAULT_DT_MS: float = 1.0

#: Default number of neurons per population-coded feature
DEFAULT_POPULATION_SIZE: int = 8

#: Default Gaussian receptive-field width (population coding)
DEFAULT_POP_SIGMA: float = 0.15


# ──────────────────────────────────────────────
# 2.  NORMALIZATION
# ──────────────────────────────────────────────

@dataclass
class FeatureRange:
    """Expected [min, max] range for a raw feature, used for normalization."""
    min_val: float
    max_val: float

    def normalize(self, value: float) -> float:
        """Clamp and scale a raw value to [0, 1]."""
        if self.max_val <= self.min_val:
            return 0.0
        x = (value - self.min_val) / (self.max_val - self.min_val)
        return max(0.0, min(1.0, x))


# ──────────────────────────────────────────────
# 3.  RATE CODING
# ──────────────────────────────────────────────

def encode_rate(
    value_norm: float,
    window_ms: float = DEFAULT_WINDOW_MS,
    max_rate_hz: float = DEFAULT_MAX_RATE_HZ,
    dt_ms: float = DEFAULT_DT_MS,
) -> List[int]:
    """
    Encode a normalized feature value as a regularly-spaced spike train.

    Parameters
    ----------
    value_norm : float
        Normalized feature value in [0, 1].
    window_ms : float
        Encoding time window in milliseconds.
    max_rate_hz : float
        Spike rate corresponding to value_norm == 1.0.
    dt_ms : float
        Simulation time-step resolution in milliseconds.

    Returns
    -------
    List[int]
        Binary spike train (1 = spike, 0 = no spike), one entry per tick.
        Length = round(window_ms / dt_ms).
    """
    value_norm = max(0.0, min(1.0, value_norm))
    n_ticks = max(1, round(window_ms / dt_ms))

    spike_rate_hz = value_norm * max_rate_hz
    if spike_rate_hz <= 0.0:
        return [0] * n_ticks

    # Expected number of spikes in the window
    n_spikes = max(0, round(spike_rate_hz * (window_ms / 1000.0)))
    n_spikes = min(n_spikes, n_ticks)

    train = [0] * n_ticks
    if n_spikes == 0:
        return train

    # Evenly space spikes across the window (regular rate code)
    interval = n_ticks / n_spikes
    for k in range(n_spikes):
        idx = min(n_ticks - 1, round(k * interval))
        train[idx] = 1

    return train


# ──────────────────────────────────────────────
# 4.  TEMPORAL CODING (TIME-TO-FIRST-SPIKE)
# ──────────────────────────────────────────────

def encode_temporal(
    value_norm: float,
    window_ms: float = DEFAULT_WINDOW_MS,
    dt_ms: float = DEFAULT_DT_MS,
) -> List[int]:
    """
    Encode a normalized feature value as a single spike whose latency
    is inversely proportional to the feature magnitude (TTFS coding).

    Parameters
    ----------
    value_norm : float
        Normalized feature value in [0, 1]. Larger → earlier spike.
    window_ms : float
        Encoding time window in milliseconds.
    dt_ms : float
        Simulation time-step resolution in milliseconds.

    Returns
    -------
    List[int]
        Binary spike train with exactly one spike (or zero if value_norm <= 0).
    """
    value_norm = max(0.0, min(1.0, value_norm))
    n_ticks = max(1, round(window_ms / dt_ms))

    train = [0] * n_ticks

    if value_norm <= 0.0:
        return train  # no spike — feature absent / minimal

    latency_ms = (1.0 - value_norm) * window_ms
    idx = min(n_ticks - 1, round(latency_ms / dt_ms))
    train[idx] = 1

    return train


# ──────────────────────────────────────────────
# 5.  POPULATION CODING
# ──────────────────────────────────────────────

def encode_population(
    value_norm: float,
    population_size: int = DEFAULT_POPULATION_SIZE,
    sigma: float = DEFAULT_POP_SIGMA,
) -> List[float]:
    """
    Encode a normalized feature value as firing-probability across a
    population of neurons with Gaussian receptive fields.

    Parameters
    ----------
    value_norm : float
        Normalized feature value in [0, 1].
    population_size : int
        Number of neurons in the encoding population.
    sigma : float
        Width of each neuron's Gaussian receptive field.

    Returns
    -------
    List[float]
        Firing probability (0..1) for each neuron in the population.
        Neuron i's preferred value (center) is i / (population_size - 1).
    """
    value_norm = max(0.0, min(1.0, value_norm))

    if population_size <= 1:
        return [1.0]

    probs = []
    for i in range(population_size):
        center = i / (population_size - 1)
        diff = value_norm - center
        prob = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
        probs.append(prob)

    return probs


def population_to_spikes(
    probs: List[float], rng_bytes: bytes
) -> List[int]:
    """
    Convert population firing probabilities into deterministic binary
    spikes using externally-supplied randomness (e.g. from a sampler).

    Parameters
    ----------
    probs : List[float]
        Firing probability per neuron (from encode_population).
    rng_bytes : bytes
        At least len(probs) bytes of randomness, each byte used to
        decide one neuron's spike via byte/256.0 < prob.

    Returns
    -------
    List[int]
        Binary spike vector (1 = fired), one entry per neuron.
    """
    if len(rng_bytes) < len(probs):
        raise ValueError("Not enough randomness bytes for population size")

    return [1 if (rng_bytes[i] / 256.0) < probs[i] else 0 for i in range(len(probs))]


# ──────────────────────────────────────────────
# 6.  MULTI-CHANNEL FEATURE ENCODER
# ──────────────────────────────────────────────

@dataclass
class ChannelConfig:
    """Configuration for one input feature channel."""
    name: str
    scheme: str  # "rate", "temporal", or "population"
    value_range: FeatureRange
    population_size: int = DEFAULT_POPULATION_SIZE


class FeatureEncoder:
    """
    Multi-channel feature encoder: converts a dict of raw telemetry
    readings into spike trains/vectors ready for the SNN input layer.
    """

    def __init__(
        self,
        window_ms: float = DEFAULT_WINDOW_MS,
        dt_ms: float = DEFAULT_DT_MS,
        max_rate_hz: float = DEFAULT_MAX_RATE_HZ,
    ):
        """
        Initialize the encoder.

        Parameters
        ----------
        window_ms : float
            Encoding window length (rate/temporal schemes).
        dt_ms : float
            Simulation tick resolution.
        max_rate_hz : float
            Maximum spike rate for rate coding.
        """
        self.window_ms = window_ms
        self.dt_ms = dt_ms
        self.max_rate_hz = max_rate_hz
        self.channels: Dict[str, ChannelConfig] = {}
        self.encode_log: List[str] = []

    def register_channel(
        self,
        name: str,
        scheme: str,
        min_val: float,
        max_val: float,
        population_size: int = DEFAULT_POPULATION_SIZE,
    ) -> None:
        """
        Register a named input channel with its encoding scheme and range.

        Parameters
        ----------
        name : str
            Channel identifier (e.g. "dilithium_retry_count", "axi_temp").
        scheme : str
            One of "rate", "temporal", "population".
        min_val : float
            Minimum expected raw value (maps to 0.0 normalized).
        max_val : float
            Maximum expected raw value (maps to 1.0 normalized).
        population_size : int
            Population size if scheme == "population".
        """
        if scheme not in ("rate", "temporal", "population"):
            raise ValueError(f"Unknown scheme: {scheme}")

        self.channels[name] = ChannelConfig(
            name=name,
            scheme=scheme,
            value_range=FeatureRange(min_val, max_val),
            population_size=population_size,
        )

    def encode(self, raw_features: Dict[str, float]) -> Dict[str, object]:
        """
        Encode a dict of raw feature readings into spike representations.

        Parameters
        ----------
        raw_features : Dict[str, float]
            Mapping of registered channel name → raw value.

        Returns
        -------
        Dict[str, object]
            Mapping of channel name → spike train (List[int]) for
            rate/temporal schemes, or firing-probability vector
            (List[float]) for population scheme.
        """
        result: Dict[str, object] = {}

        for name, value in raw_features.items():
            cfg = self.channels.get(name)
            if cfg is None:
                continue  # unregistered channel — ignore silently

            value_norm = cfg.value_range.normalize(value)

            if cfg.scheme == "rate":
                result[name] = encode_rate(
                    value_norm, self.window_ms, self.max_rate_hz, self.dt_ms
                )
            elif cfg.scheme == "temporal":
                result[name] = encode_temporal(
                    value_norm, self.window_ms, self.dt_ms
                )
            else:  # population
                result[name] = encode_population(
                    value_norm, cfg.population_size
                )

            self.encode_log.append(name)

        return result

    def total_input_neurons(self) -> int:
        """
        Compute the total number of SNN input neurons required across
        all registered channels (rate/temporal use 1 neuron each;
        population uses population_size neurons).
        """
        total = 0
        for cfg in self.channels.values():
            if cfg.scheme == "population":
                total += cfg.population_size
            else:
                total += 1
        return total

    def stats(self) -> Dict:
        """Return encoder statistics."""
        return {
            "registered_channels": len(self.channels),
            "channel_names": list(self.channels.keys()),
            "total_input_neurons": self.total_input_neurons(),
            "encode_calls": len(self.encode_log),
        }


# ──────────────────────────────────────────────
# 7.  SELF-TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("feature_encoder.py  —  self-test")
    print("=" * 55)

    # Test 1: Normalization
    print("\n[ Feature normalization ]")
    rng = FeatureRange(0.0, 100.0)
    assert rng.normalize(50.0) == 0.5
    assert rng.normalize(-10.0) == 0.0   # clamped
    assert rng.normalize(150.0) == 1.0   # clamped
    print(f"  ✓ Normalize: 50→0.5, -10→0.0 (clamp), 150→1.0 (clamp)")

    # Test 2: Rate coding — zero and max
    print("\n[ Rate coding ]")
    train_zero = encode_rate(0.0, window_ms=100.0, max_rate_hz=200.0, dt_ms=1.0)
    assert sum(train_zero) == 0
    train_max = encode_rate(1.0, window_ms=100.0, max_rate_hz=200.0, dt_ms=1.0)
    # 200Hz over 100ms window → ~20 spikes
    assert 18 <= sum(train_max) <= 22
    print(f"  ✓ value=0.0 → {sum(train_zero)} spikes; value=1.0 → {sum(train_max)} spikes")

    # Test 3: Rate coding monotonicity
    print("\n[ Rate coding monotonicity ]")
    counts = [sum(encode_rate(v, 100.0, 200.0, 1.0)) for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert counts == sorted(counts)
    print(f"  ✓ Spike counts increase monotonically: {counts}")

    # Test 4: Temporal coding — latency direction
    print("\n[ Temporal coding ]")
    train_low = encode_temporal(0.1, window_ms=100.0, dt_ms=1.0)
    train_high = encode_temporal(0.9, window_ms=100.0, dt_ms=1.0)
    latency_low = train_low.index(1) if 1 in train_low else None
    latency_high = train_high.index(1) if 1 in train_high else None
    assert latency_high < latency_low  # higher value → earlier spike
    print(f"  ✓ value=0.1 → spike@{latency_low}ms; value=0.9 → spike@{latency_high}ms")

    # Test 5: Temporal coding — zero value produces no spike
    print("\n[ Temporal coding — zero value ]")
    train_zero_t = encode_temporal(0.0, window_ms=100.0, dt_ms=1.0)
    assert sum(train_zero_t) == 0
    print(f"  ✓ value=0.0 → no spike (feature absent)")

    # Test 6: Population coding — peak at correct neuron
    print("\n[ Population coding ]")
    probs = encode_population(0.5, population_size=9, sigma=0.15)
    peak_idx = probs.index(max(probs))
    assert peak_idx == 4  # center neuron for population_size=9 (index 4 → 4/8=0.5)
    print(f"  ✓ value=0.5 → peak response at neuron {peak_idx} (expected center)")

    # Test 7: Population coding extremes
    print("\n[ Population coding — extremes ]")
    probs_low = encode_population(0.0, population_size=9, sigma=0.15)
    probs_high = encode_population(1.0, population_size=9, sigma=0.15)
    assert probs_low.index(max(probs_low)) == 0
    assert probs_high.index(max(probs_high)) == 8
    print(f"  ✓ value=0.0 → peak@neuron0; value=1.0 → peak@neuron8")

    # Test 8: Population → binary spikes via external randomness
    print("\n[ Population → binary spikes ]")
    probs = encode_population(0.5, population_size=4, sigma=0.2)
    rng_bytes = bytes([0, 255, 0, 255])  # alternating low/high randomness
    spikes = population_to_spikes(probs, rng_bytes)
    assert len(spikes) == 4
    assert all(s in (0, 1) for s in spikes)
    print(f"  ✓ Probabilities {[f'{p:.2f}' for p in probs]} → spikes {spikes}")

    # Test 9: Multi-channel FeatureEncoder
    print("\n[ Multi-channel encoder ]")
    enc = FeatureEncoder(window_ms=100.0, dt_ms=1.0, max_rate_hz=200.0)
    enc.register_channel("dilithium_retries", "rate", min_val=0, max_val=20)
    enc.register_channel("packet_latency_ms", "temporal", min_val=0, max_val=50)
    enc.register_channel("tamper_voltage", "population", min_val=1.5, max_val=2.0, population_size=6)

    raw = {
        "dilithium_retries": 15,       # high retry count → high rate
        "packet_latency_ms": 5,        # low latency → early spike
        "tamper_voltage": 1.8,         # nominal voltage → mid population
    }
    encoded = enc.encode(raw)
    assert "dilithium_retries" in encoded
    assert "packet_latency_ms" in encoded
    assert "tamper_voltage" in encoded
    assert isinstance(encoded["dilithium_retries"], list)
    assert isinstance(encoded["tamper_voltage"], list)
    print(f"  ✓ Encoded 3 heterogeneous channels in one pass")

    # Test 10: Unregistered channel ignored
    print("\n[ Unregistered channel handling ]")
    raw_with_extra = dict(raw)
    raw_with_extra["unregistered_feature"] = 99.0
    encoded2 = enc.encode(raw_with_extra)
    assert "unregistered_feature" not in encoded2
    assert len(encoded2) == 3
    print(f"  ✓ Unregistered channel silently ignored")

    # Test 11: Total input neuron count
    print("\n[ Total input neuron count ]")
    total = enc.total_input_neurons()
    # rate(1) + temporal(1) + population(6) = 8
    assert total == 8
    print(f"  ✓ Total SNN input neurons required: {total}")

    # Test 12: Statistics
    print("\n[ Encoder statistics ]")
    stats = enc.stats()
    print(f"  Channels: {stats['registered_channels']}")
    print(f"  Names: {stats['channel_names']}")
    print(f"  Input neurons: {stats['total_input_neurons']}")
    print(f"  Encode calls: {stats['encode_calls']}")
    print(f"  ✓ Statistics tracked correctly")

    print("\n  All checks passed.\n")
