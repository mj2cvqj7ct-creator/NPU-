from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import EnhancementConfig


Array = np.ndarray


@dataclass(frozen=True)
class AudioMetrics:
    peak: float
    rms: float
    loudness_dbfs: float
    clipping_ratio: float


@dataclass(frozen=True)
class EnhancementState:
    previous_loudness_gain: float = 1.0


def ensure_audio_frame(audio: Array) -> Array:
    frame = np.asarray(audio, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError("audio must be shaped as (samples, channels)")
    if frame.shape[1] not in (1, 2):
        raise ValueError("only mono or stereo audio is supported")
    if not np.all(np.isfinite(frame)):
        raise ValueError("audio contains NaN or infinite values")
    return frame


def measure_metrics(audio: Array) -> AudioMetrics:
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


def loudness_normalize(audio: Array, config: EnhancementConfig, previous_gain: float) -> tuple[Array, float]:
    frame = ensure_audio_frame(audio)
    metrics = measure_metrics(frame)
    desired_gain_db = config.target_loudness_dbfs - metrics.loudness_dbfs
    desired_gain = float(np.clip(10.0 ** (desired_gain_db / 20.0), config.min_gain, config.max_gain))
    smoothed_gain = (
        config.loudness_smoothing * previous_gain
        + (1.0 - config.loudness_smoothing) * desired_gain
    )
    return frame * np.float32(smoothed_gain), smoothed_gain


def dynamic_eq(audio: Array, config: EnhancementConfig) -> Array:
    frame = ensure_audio_frame(audio)
    if frame.shape[0] < 3:
        return frame.copy()

    low = np.empty_like(frame)
    low[0] = frame[0]
    alpha = np.float32(config.lowpass_alpha)
    for index in range(1, frame.shape[0]):
        low[index] = low[index - 1] + alpha * (frame[index] - low[index - 1])

    high = frame - low
    enhanced = frame + np.float32(config.low_shelf_gain - 1.0) * low
    enhanced += np.float32(config.presence_gain - 1.0) * high
    return enhanced.astype(np.float32, copy=False)


def soft_knee_limiter(audio: Array, config: EnhancementConfig) -> Array:
    frame = ensure_audio_frame(audio)
    threshold = np.float32(config.limiter_threshold)
    abs_frame = np.abs(frame)
    sign = np.sign(frame)
    limited = np.where(
        abs_frame <= threshold,
        frame,
        sign * (threshold + (1.0 - threshold) * np.tanh((abs_frame - threshold) / (1.0 - threshold))),
    )
    return np.clip(limited, -config.true_peak_ceiling, config.true_peak_ceiling).astype(np.float32)


def enhance_frame(
    audio: Array,
    config: EnhancementConfig | None = None,
    state: EnhancementState | None = None,
) -> tuple[Array, EnhancementState, AudioMetrics]:
    config = config or EnhancementConfig()
    state = state or EnhancementState()
    frame = ensure_audio_frame(audio)

    balanced = balance_channels(frame)
    normalized, gain = loudness_normalize(balanced, config, state.previous_loudness_gain)
    equalized = dynamic_eq(normalized, config)
    limited = soft_knee_limiter(equalized, config)
    return limited, replace(state, previous_loudness_gain=gain), measure_metrics(limited)
