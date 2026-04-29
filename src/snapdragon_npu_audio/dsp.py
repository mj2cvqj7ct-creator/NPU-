from __future__ import annotations

import math

from .models import AudioFrame


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)


def estimate_loudness_lufs(frame: AudioFrame) -> float:
    """Estimate short-window loudness with RMS, suitable for real-time control."""
    if not frame.samples:
        return -120.0
    return _linear_to_db(frame.rms)


def normalize_loudness(
    frame: AudioFrame,
    *,
    target_lufs: float,
    max_gain_db: float = 12.0,
) -> AudioFrame:
    normalized, _, _ = normalize_loudness_with_metrics(
        frame,
        target_lufs=target_lufs,
        max_gain_db=max_gain_db,
    )
    return normalized


def normalize_loudness_with_metrics(
    frame: AudioFrame,
    *,
    target_lufs: float,
    max_gain_db: float = 12.0,
) -> tuple[AudioFrame, float, float]:
    loudness_db = estimate_loudness_lufs(frame)
    desired_gain_db = target_lufs - loudness_db
    safe_gain_db = _clamp(desired_gain_db, -max_gain_db, max_gain_db)
    return apply_gain(frame, safe_gain_db), loudness_db, safe_gain_db


def apply_gain(
    frame: AudioFrame,
    gain_db: float,
) -> AudioFrame:
    gain = _db_to_linear(gain_db)
    return frame.with_samples((left * gain, right * gain) for left, right in frame.samples)


def apply_channel_balance(
    frame: AudioFrame,
    balance: float,
) -> AudioFrame:
    balance = _clamp(balance, -1.0, 1.0)
    left_gain = 1.0 - max(0.0, balance)
    right_gain = 1.0 + min(0.0, balance)
    return frame.with_samples(
        (left * left_gain, right * right_gain) for left, right in frame.samples
    )


def apply_dynamic_tilt_eq(
    frame: AudioFrame,
    *,
    bass_db: float,
    presence_db: float,
    stereo_width: float,
) -> AudioFrame:
    """Apply a conservative one-pole tilt EQ appropriate for 10-20 ms frames."""
    if not frame.samples:
        return frame

    alpha = _clamp(120.0 / frame.sample_rate_hz, 0.001, 0.02)
    clarity = _db_to_linear(_clamp(presence_db, 0.0, 3.0)) - 1.0
    bass = _db_to_linear(_clamp(bass_db, 0.0, 3.0)) - 1.0
    width = _clamp(stereo_width, 0.75, 1.35)
    low_left = frame.samples[0][0]
    low_right = frame.samples[0][1]
    output: list[tuple[float, float]] = []

    for left, right in frame.samples:
        low_left += alpha * (left - low_left)
        low_right += alpha * (right - low_right)
        high_left = left - low_left
        high_right = right - low_right
        shaped_left = left + low_left * bass + high_left * clarity
        shaped_right = right + low_right * bass + high_right * clarity
        mid = (shaped_left + shaped_right) * 0.5
        side = (shaped_left - shaped_right) * 0.5 * width
        output.append(
            (
                mid + side,
                mid - side,
            )
        )

    return frame.with_samples(tuple(output))


def apply_true_peak_limiter(
    frame: AudioFrame,
    *,
    ceiling_dbfs: float = -1.0,
) -> AudioFrame:
    limited, _ = true_peak_limit(frame, ceiling_dbfs=ceiling_dbfs)
    return limited


def true_peak_limit(
    frame: AudioFrame,
    *,
    ceiling_dbfs: float = -1.0,
) -> tuple[AudioFrame, int]:
    ceiling = _db_to_linear(ceiling_dbfs)
    limited_samples: list[tuple[float, float]] = []
    limited_count = 0
    for left, right in frame.samples:
        limited_left = _clamp(left, -ceiling, ceiling)
        limited_right = _clamp(right, -ceiling, ceiling)
        limited_count += int(limited_left != left) + int(limited_right != right)
        limited_samples.append((limited_left, limited_right))
    return frame.with_samples(tuple(limited_samples)), limited_count
