from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnhancerConfig, db_to_linear


Array = np.ndarray


@dataclass(frozen=True)
class AudioMetrics:
    peak: float
    rms: float
    loudness_dbfs: float
    clipping_ratio: float


def ensure_audio_frame(audio: Array) -> Array:
    frame = np.asarray(audio, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError("audio must be stereo and shaped as (samples, 2)")
    if frame.shape[1] != 2:
        raise ValueError("audio must be stereo and shaped as (samples, 2)")
    if not np.all(np.isfinite(frame)):
        raise ValueError("audio contains NaN or infinite values")
    return frame


def analyze_audio(audio: Array) -> AudioMetrics:
    frame = ensure_audio_frame(audio)
    abs_frame = np.abs(frame)
    peak = float(abs_frame.max(initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64))) if frame.size else 0.0
    loudness_dbfs = 20.0 * np.log10(max(rms, 1e-12))
    clipping_ratio = float(np.mean(abs_frame >= 0.999)) if frame.size else 0.0
    return AudioMetrics(
        peak=peak,
        rms=rms,
        loudness_dbfs=float(loudness_dbfs),
        clipping_ratio=clipping_ratio,
    )


measure_metrics = analyze_audio


def balance_channels(audio: Array) -> Array:
    frame = ensure_audio_frame(audio)
    if frame.shape[1] == 1:
        return frame.copy()

    channel_rms = np.sqrt(np.mean(np.square(frame), axis=0, dtype=np.float64))
    if np.any(channel_rms < 1e-7):
        return frame.copy()

    target = float(np.mean(channel_rms))
    gains = np.clip(target / channel_rms, 0.5, 2.0).astype(np.float32)
    return frame * gains


def loudness_normalize(audio: Array, config: EnhancerConfig, previous_gain: float) -> tuple[Array, float]:
    frame = ensure_audio_frame(audio)
    metrics = analyze_audio(frame)
    if metrics.rms <= 1e-9:
        return frame.copy(), previous_gain
    desired_gain = config.target_rms_linear / metrics.rms
    desired_gain = float(np.clip(desired_gain, 1.0 / config.max_gain_linear, config.max_gain_linear))
    smoothed_gain = 0.9 * previous_gain + 0.1 * desired_gain
    return frame * np.float32(smoothed_gain), smoothed_gain


def dynamic_eq(audio: Array, config: EnhancerConfig, eq_curve: tuple[float, float, float] | None = None) -> Array:
    frame = ensure_audio_frame(audio)
    if frame.shape[0] < 3:
        return frame.copy()

    low = np.empty_like(frame)
    low[0] = frame[0]
    alpha = np.float32(0.08)
    for index in range(1, frame.shape[0]):
        low[index] = low[index - 1] + alpha * (frame[index] - low[index - 1])

    high = frame - low
    low_db, presence_db, air_db = eq_curve or config.headphone_eq_db
    low_gain = db_to_linear(low_db + config.low_volume_bass_lift_db)
    presence_gain = db_to_linear(presence_db)
    air_gain = db_to_linear(air_db)
    enhanced = frame + np.float32(low_gain - 1.0) * low
    enhanced += np.float32(((presence_gain + air_gain) * 0.5) - 1.0) * high
    return enhanced.astype(np.float32, copy=False)


def soft_knee_limiter(audio: Array, config: EnhancerConfig) -> Array:
    frame = ensure_audio_frame(audio)
    threshold = np.float32(config.true_peak_linear * 0.92)
    abs_frame = np.abs(frame)
    sign = np.sign(frame)
    limited = np.where(
        abs_frame <= threshold,
        frame,
        sign * (threshold + (1.0 - threshold) * np.tanh((abs_frame - threshold) / (1.0 - threshold))),
    )
    return np.clip(limited, -config.true_peak_linear, config.true_peak_linear).astype(np.float32)


class LoudnessNormalizer:
    def __init__(
        self,
        config: EnhancerConfig | None = None,
        *,
        target_lufs: float | None = None,
        max_gain_db: float | None = None,
    ) -> None:
        self.config = config or EnhancerConfig(
            target_lufs=-18.0 if target_lufs is None else target_lufs,
            max_gain_db=6.0 if max_gain_db is None else max_gain_db,
        )
        self._previous_gain = 1.0

    def process(self, audio: Array, metrics: AudioMetrics | None = None) -> Array:
        del metrics
        processed, self._previous_gain = loudness_normalize(
            audio,
            self.config,
            self._previous_gain,
        )
        return np.clip(processed, -1.0, 1.0).astype(np.float32)


class DynamicEq:
    def __init__(
        self,
        config: EnhancerConfig | None = None,
        *,
        sample_rate: int = 48_000,
        low_shelf_db: float = 0.8,
        presence_db: float = 0.0,
        air_db: float = 0.6,
    ) -> None:
        self.config = config or EnhancerConfig(
            sample_rate=sample_rate,
            headphone_eq_db=(low_shelf_db, presence_db, air_db),
        )

    def process(self, audio: Array, eq_curve: tuple[float, float, float] | None = None) -> Array:
        return dynamic_eq(audio, self.config, eq_curve=eq_curve)


class TruePeakLimiter:
    def __init__(
        self,
        config: EnhancerConfig | None = None,
        *,
        ceiling_dbfs: float = -1.0,
    ) -> None:
        self.config = config or EnhancerConfig(limiter_ceiling_dbfs=ceiling_dbfs)

    def process(self, audio: Array) -> Array:
        return soft_knee_limiter(audio, self.config)


FrameLimiter = TruePeakLimiter


def enhance_frame(audio: Array, config: EnhancerConfig | None = None) -> tuple[Array, AudioMetrics]:
    config = config or EnhancerConfig()
    balanced = balance_channels(audio)
    normalized = LoudnessNormalizer(config).process(balanced)
    equalized = DynamicEq(config).process(normalized)
    limited = TruePeakLimiter(config).process(equalized)
    return limited, analyze_audio(limited)
