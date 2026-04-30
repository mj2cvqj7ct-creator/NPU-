"""Small, deterministic DSP building blocks for the prototype enhancer.

The production Windows app would run equivalent operations in a real-time
audio callback. These functions keep the math explicit and dependency-free so
they can be tested on CI without Snapdragon-specific hardware.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

SampleFrame = tuple[float, float]


def clamp_sample(value: float, limit: float = 0.999) -> float:
    """Clamp one normalized floating-point PCM sample."""

    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float:
    if gain <= 0:
        return -120.0
    return 20.0 * math.log10(gain)


class AudioStats:
    """Frame-level measurements used by service profiles and tests."""

    def __init__(
        self,
        rms_dbfs: float,
        peak_dbfs: float,
        crest_factor_db: float,
        stereo_balance: float,
    ) -> None:
        self.rms_dbfs = rms_dbfs
        self.peak_dbfs = peak_dbfs
        self.crest_factor_db = crest_factor_db
        self.stereo_balance = stereo_balance


def measure(frames: Sequence[SampleFrame]) -> AudioStats:
    """Measure simple loudness, peak, crest factor, and L/R balance."""

    if not frames:
        return AudioStats(-120.0, -120.0, 0.0, 0.0)

    sum_squares = 0.0
    peak = 0.0
    left_energy = 0.0
    right_energy = 0.0

    for left, right in frames:
        sum_squares += left * left + right * right
        left_energy += left * left
        right_energy += right * right
        peak = max(peak, abs(left), abs(right))

    rms = math.sqrt(sum_squares / (len(frames) * 2))
    rms_dbfs = gain_to_db(rms)
    peak_dbfs = gain_to_db(peak)
    return AudioStats(
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        crest_factor_db=peak_dbfs - rms_dbfs,
        stereo_balance=(left_energy - right_energy) / max(left_energy + right_energy, 1e-12),
    )


def normalize_loudness(
    frames: Sequence[SampleFrame],
    target_rms_dbfs: float,
    max_gain_db: float,
) -> list[SampleFrame]:
    stats = measure(frames)
    requested_gain_db = target_rms_dbfs - stats.rms_dbfs
    gain = db_to_gain(max(min(requested_gain_db, max_gain_db), -max_gain_db))
    return [(left * gain, right * gain) for left, right in frames]


def enhance_presence(
    frames: Sequence[SampleFrame],
    presence_gain_db: float,
    bass_tighten: float,
) -> list[SampleFrame]:
    """Apply a lightweight high-shelf-like detail lift and low smear control."""

    if not frames:
        return []

    detail_gain = db_to_gain(presence_gain_db) - 1.0
    previous_mid = 0.0
    enhanced: list[SampleFrame] = []

    for left, right in frames:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        transient = mid - previous_mid
        previous_mid = mid

        controlled_mid = mid - bass_tighten * transient
        lifted_mid = controlled_mid + transient * detail_gain
        enhanced.append((lifted_mid + side, lifted_mid - side))

    return enhanced


def apply_stereo_width(frames: Sequence[SampleFrame], width: float) -> list[SampleFrame]:
    widened: list[SampleFrame] = []
    safe_width = max(0.0, min(width, 1.12))
    for left, right in frames:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * safe_width
        widened.append((mid + side, mid - side))
    return widened


def true_peak_limiter(
    frames: Sequence[SampleFrame],
    ceiling: float,
    release: float = 0.995,
) -> list[SampleFrame]:
    """Simple look-ahead-free limiter suitable for offline tests."""

    gain = 1.0
    limited: list[SampleFrame] = []
    for left, right in frames:
        peak = max(abs(left), abs(right))
        target_gain = min(1.0, ceiling / peak) if peak > 0 else 1.0
        if target_gain < gain:
            gain = target_gain
        else:
            gain = min(1.0, gain / release)
        limited.append((clamp_sample(left * gain, ceiling), clamp_sample(right * gain, ceiling)))
    return limited


def enhance_frames(
    frames: Sequence[SampleFrame],
    profile: object,
    npu_detail_gain: float = 0.0,
) -> list[SampleFrame]:
    """Run the prototype enhancement chain for one block of stereo frames."""

    presence_gain_db = profile.presence_gain_db + npu_detail_gain
    normalized = normalize_loudness(
        frames,
        profile.target_rms_dbfs,
        profile.max_gain_db,
    )
    present = enhance_presence(
        normalized,
        presence_gain_db,
        profile.bass_tighten,
    )
    widened = apply_stereo_width(present, profile.stereo_width)
    return true_peak_limiter(widened, profile.limiter_ceiling)


def chunk_frames(frames: Sequence[SampleFrame], size: int) -> Iterable[Sequence[SampleFrame]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for offset in range(0, len(frames), size):
        yield frames[offset : offset + size]
