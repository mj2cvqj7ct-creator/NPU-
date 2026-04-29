from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .profiles import DynamicEqProfile


AudioFrame = list[list[float]]


MAX_TRUE_PEAK = 0.98


class AudioFrameError(ValueError):
    """Raised when PCM frames are malformed or unsafe to process."""


@dataclass(frozen=True)
class FrameStats:
    channels: int
    samples_per_channel: int
    peak: float
    rms: float
    loudness_db: float


@dataclass(frozen=True)
class ProcessingStats:
    input: FrameStats
    output: FrameStats
    loudness_gain_db: float


def validate_stereo_frame(frame: Iterable[Iterable[float]]) -> AudioFrame:
    channels = [list(channel) for channel in frame]
    if len(channels) != 2:
        raise AudioFrameError("expected 48 kHz stereo PCM with exactly two channels")

    sample_count = len(channels[0])
    if sample_count == 0:
        raise AudioFrameError("audio frame must contain at least one sample")

    for channel in channels:
        if len(channel) != sample_count:
            raise AudioFrameError("stereo channels must have the same sample count")
        for sample in channel:
            if not isfinite(sample):
                raise AudioFrameError("audio frame contains a non-finite sample")

    return channels


def frame_stats(frame: Iterable[Iterable[float]]) -> FrameStats:
    channels = validate_stereo_frame(frame)
    flattened = [sample for channel in channels for sample in channel]
    peak = max(abs(sample) for sample in flattened)
    mean_square = sum(sample * sample for sample in flattened) / len(flattened)
    rms = mean_square**0.5
    loudness_db = -120.0 if rms == 0 else 20.0 * _log10(rms)
    return FrameStats(
        channels=2,
        samples_per_channel=len(channels[0]),
        peak=peak,
        rms=rms,
        loudness_db=loudness_db,
    )


def apply_gain(frame: Iterable[Iterable[float]], gain: float) -> AudioFrame:
    channels = validate_stereo_frame(frame)
    if not isfinite(gain):
        raise AudioFrameError("gain must be finite")
    return [[sample * gain for sample in channel] for channel in channels]


def soft_knee_limiter(
    frame: Iterable[Iterable[float]], threshold: float = MAX_TRUE_PEAK
) -> AudioFrame:
    channels = validate_stereo_frame(frame)
    if threshold <= 0.0 or threshold > 1.0:
        raise AudioFrameError("limiter threshold must be in the range (0.0, 1.0]")

    limited: list[list[float]] = []
    for channel in channels:
        limited_channel = []
        for sample in channel:
            magnitude = abs(sample)
            if magnitude <= threshold:
                limited_channel.append(sample)
                continue
            sign = 1.0 if sample >= 0.0 else -1.0
            overshoot = magnitude - threshold
            limited_channel.append(sign * (threshold + overshoot / (1.0 + overshoot)))
        limited.append(limited_channel)
    return normalize_peak(limited, threshold)


def normalize_peak(
    frame: Iterable[Iterable[float]], target_peak: float = MAX_TRUE_PEAK
) -> AudioFrame:
    channels = validate_stereo_frame(frame)
    peak = frame_stats(channels).peak
    if peak == 0.0 or peak <= target_peak:
        return channels
    return apply_gain(channels, target_peak / peak)


def dynamic_stereo_width(
    frame: Iterable[Iterable[float]], width: float
) -> AudioFrame:
    channels = validate_stereo_frame(frame)
    if width < 0.0 or width > 1.5:
        raise AudioFrameError("stereo width must be between 0.0 and 1.5")

    left, right = channels
    widened_left: list[float] = []
    widened_right: list[float] = []
    for left_sample, right_sample in zip(left, right):
        mid = (left_sample + right_sample) * 0.5
        side = (left_sample - right_sample) * 0.5 * width
        widened_left.append(mid + side)
        widened_right.append(mid - side)
    return normalize_peak([widened_left, widened_right])


def clarity_shelf(frame: Iterable[Iterable[float]], amount: float) -> AudioFrame:
    channels = validate_stereo_frame(frame)
    if amount < 0.0 or amount > 1.0:
        raise AudioFrameError("clarity amount must be between 0.0 and 1.0")

    # A single-pole high-passed exciter approximates vocal-presence enhancement
    # without requiring FFT dependencies in the real-time safety tests.
    enhanced: list[list[float]] = []
    for channel in channels:
        previous = 0.0
        output_channel = []
        for sample in channel:
            high = sample - previous
            previous = sample
            output_channel.append(sample + high * 0.22 * amount)
        enhanced.append(output_channel)
    return normalize_peak(enhanced)


def normalize_loudness(
    frame: Iterable[Iterable[float]], target_lufs: float, max_gain_db: float
) -> tuple[AudioFrame, ProcessingStats]:
    input_stats = frame_stats(frame)
    if max_gain_db < 0.0:
        raise AudioFrameError("max_gain_db must be non-negative")

    desired_gain_db = target_lufs - input_stats.loudness_db
    clamped_gain_db = max(-max_gain_db, min(max_gain_db, desired_gain_db))
    gained = apply_gain(frame, db_to_linear(clamped_gain_db))
    limited = normalize_peak(gained)
    output_stats = frame_stats(limited)
    return limited, ProcessingStats(
        input=input_stats,
        output=output_stats,
        loudness_gain_db=clamped_gain_db,
    )


def apply_dynamic_eq(frame: Iterable[Iterable[float]], eq: DynamicEqProfile) -> AudioFrame:
    channels = validate_stereo_frame(frame)
    bass_amount = _gain_db_to_amount(eq.bass_gain_db, scale=3.0)
    clarity_amount = _gain_db_to_amount(eq.presence_gain_db + eq.air_gain_db, scale=4.0)

    bass_balanced = _bass_warmth(channels, bass_amount)
    clarified = clarity_shelf(bass_balanced, clarity_amount)
    widened = dynamic_stereo_width(clarified, eq.stereo_width)
    restored = _transient_restore(widened, eq.transient_restore)
    return normalize_peak(restored)


def true_peak_limit(frame: Iterable[Iterable[float]], ceiling_dbfs: float) -> AudioFrame:
    if ceiling_dbfs >= 0.0:
        raise AudioFrameError("ceiling_dbfs must leave headroom below 0 dBFS")
    return soft_knee_limiter(frame, threshold=db_to_linear(ceiling_dbfs))


def _bass_warmth(frame: AudioFrame, amount: float) -> AudioFrame:
    enhanced: AudioFrame = []
    for channel in frame:
        low_state = 0.0
        output_channel = []
        for sample in channel:
            low_state = low_state * 0.94 + sample * 0.06
            output_channel.append(sample + low_state * 0.18 * amount)
        enhanced.append(output_channel)
    return normalize_peak(enhanced)


def _transient_restore(frame: AudioFrame, amount: float) -> AudioFrame:
    if not 0.0 <= amount <= 1.0:
        raise AudioFrameError("transient restore amount must be between 0.0 and 1.0")

    enhanced: AudioFrame = []
    for channel in frame:
        previous = 0.0
        output_channel = []
        for sample in channel:
            edge = sample - previous
            previous = sample
            output_channel.append(sample + edge * 0.12 * amount)
        enhanced.append(output_channel)
    return normalize_peak(enhanced)


def db_to_linear(db_value: float) -> float:
    return 10.0 ** (db_value / 20.0)


def _gain_db_to_amount(db_value: float, scale: float) -> float:
    return max(0.0, min(1.0, db_value / scale))


def _log10(value: float) -> float:
    # Avoid importing cmath or numpy; math.log10 is intentionally localized here.
    import math

    return math.log10(value)
