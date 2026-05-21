"""Low-latency rule-based DSP for the audio enhancer prototype."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log10, sqrt

from .audio_types import AudioFrame, AudioFormat, Sample


@dataclass(frozen=True)
class EnhancerConfig:
    """Controls conservative music enhancement before any neural model runs."""

    target_loudness_db: float = -16.0
    max_gain_db: float = 6.0
    bass_gain_db: float = 1.5
    presence_gain_db: float = 1.0
    stereo_width: float = 1.04
    limiter_ceiling: float = 0.98

    def merged(self, **overrides: float) -> "EnhancerConfig":
        return replace(self, **overrides)


@dataclass(frozen=True)
class FrameFeatures:
    """Small feature vector that can be fed into an NPU model."""

    loudness_db: float
    peak: float
    crest_factor_db: float
    stereo_correlation: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float


class AudioEnhancer:
    """Applies deterministic enhancement suitable for 10-20 ms audio frames."""

    def __init__(
        self, config: EnhancerConfig | None = None, fmt: AudioFormat | None = None
    ) -> None:
        self.config = config or EnhancerConfig()
        self.fmt = fmt or AudioFormat()
        self.fmt.validate()
        self._bass_state = _OnePoleState()
        self._presence_state = _OnePoleState()

    def analyze(self, frame: AudioFrame) -> FrameFeatures:
        self._ensure_format(frame)
        if not frame.samples:
            return FrameFeatures(-120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        left_energy = 0.0
        right_energy = 0.0
        cross_energy = 0.0
        mono: list[float] = []
        for left, right in frame.samples:
            left_energy += left * left
            right_energy += right * right
            cross_energy += left * right
            mono.append((left + right) * 0.5)

        sample_count = len(frame.samples)
        rms = sqrt((left_energy + right_energy) / (sample_count * 2))
        loudness_db = _linear_to_db(rms)
        peak = frame.peak()
        crest_factor_db = _linear_to_db(peak / rms) if rms > 0.0 else 0.0
        denominator = sqrt(left_energy * right_energy)
        stereo_correlation = cross_energy / denominator if denominator > 0.0 else 0.0
        low, mid, high = _split_band_energy(mono)
        return FrameFeatures(
            loudness_db=loudness_db,
            peak=peak,
            crest_factor_db=crest_factor_db,
            stereo_correlation=stereo_correlation,
            low_band_energy=low,
            mid_band_energy=mid,
            high_band_energy=high,
        )

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Enhance a frame while preserving the input duration and format."""

        self._ensure_format(frame)
        if not frame.samples:
            return frame

        features = self.analyze(frame)
        loudness_gain = _db_to_linear(
            min(self.config.max_gain_db, self.config.target_loudness_db - features.loudness_db)
        )
        bass_gain = _db_to_linear(self.config.bass_gain_db)
        presence_gain = _db_to_linear(self.config.presence_gain_db)

        processed: list[Sample] = []
        for left, right in frame.samples:
            mono = (left + right) * 0.5
            side = (left - right) * 0.5
            low = self._bass_state.low_pass(mono, coefficient=0.08)
            high = mono - self._presence_state.low_pass(mono, coefficient=0.28)

            tone = mono
            tone += (bass_gain - 1.0) * low
            tone += (presence_gain - 1.0) * high

            adjusted_side = side * self._safe_stereo_width(features.stereo_correlation)
            out_left = (tone + adjusted_side) * loudness_gain
            out_right = (tone - adjusted_side) * loudness_gain
            processed.append((out_left, out_right))

        limited = _true_peak_limit(processed, self.config.limiter_ceiling)
        return AudioFrame(tuple(limited), frame.fmt)

    def _safe_stereo_width(self, correlation: float) -> float:
        if correlation < 0.15:
            return min(self.config.stereo_width, 1.0)
        if correlation > 0.95:
            return min(self.config.stereo_width, 1.08)
        return self.config.stereo_width

    def _ensure_format(self, frame: AudioFrame) -> None:
        if frame.fmt != self.fmt:
            raise ValueError(f"expected format {self.fmt}, got {frame.fmt}")


@dataclass
class _OnePoleState:
    value: float = 0.0

    def low_pass(self, sample: float, coefficient: float) -> float:
        self.value += coefficient * (sample - self.value)
        return self.value


def _split_band_energy(samples: list[float]) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0

    low_state = 0.0
    mid_state = 0.0
    low_energy = 0.0
    mid_energy = 0.0
    high_energy = 0.0
    for sample in samples:
        low_state += 0.05 * (sample - low_state)
        mid_state += 0.22 * (sample - mid_state)
        low = low_state
        mid = mid_state - low_state
        high = sample - mid_state
        low_energy += low * low
        mid_energy += mid * mid
        high_energy += high * high

    total = low_energy + mid_energy + high_energy
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    return low_energy / total, mid_energy / total, high_energy / total


def _true_peak_limit(samples: list[Sample], ceiling: float) -> list[Sample]:
    peak = max((max(abs(left), abs(right)) for left, right in samples), default=0.0)
    if peak <= ceiling or peak <= 0.0:
        return [(left, right) for left, right in samples]
    gain = ceiling / peak
    return [(left * gain, right * gain) for left, right in samples]


def _linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * log10(value)


def _db_to_linear(value: float) -> float:
    return 10.0 ** (value / 20.0)
