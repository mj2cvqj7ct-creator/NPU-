"""Low-latency DSP primitives for the audio enhancer prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .audio import AudioBuffer, Sample, db_to_linear, linear_to_db


EPSILON = 1.0e-12


@dataclass(frozen=True)
class AudioFeatures:
    """Short-window features used by local enhancement models."""

    rms_dbfs: float
    peak_dbfs: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float
    stereo_width: float
    clipping_ratio: float


@dataclass(frozen=True)
class EnhancementProfile:
    """User and device controls for conservative real-time enhancement."""

    target_rms_dbfs: float = -18.0
    max_gain_db: float = 9.0
    bass_boost_db: float = 1.5
    presence_boost_db: float = 1.25
    air_boost_db: float = 0.75
    stereo_width: float = 1.05
    limiter_ceiling_dbfs: float = -1.0


def extract_features(buffer: AudioBuffer) -> AudioFeatures:
    """Extract lightweight features without storing or identifying content."""

    samples = [value for frame in buffer.samples for value in frame]
    if not samples:
        return AudioFeatures(-120.0, -120.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    square_sum = sum(sample * sample for sample in samples)
    rms = math.sqrt(square_sum / len(samples))
    peak = max(abs(sample) for sample in samples)
    clipping = sum(1 for sample in samples if abs(sample) >= 0.999) / len(samples)

    low = mid = high = 0.0
    previous_left = 0.0
    previous_right = 0.0
    for left, right in buffer.samples:
        for sample, previous in ((left, previous_left), (right, previous_right)):
            delta = sample - previous
            low += sample * sample
            mid += delta * delta
            high += (sample - 2.0 * previous) ** 2
        previous_left = left
        previous_right = right

    total = max(low + mid + high, EPSILON)
    side_energy = 0.0
    mid_energy = 0.0
    for left, right in buffer.samples:
        mid_sample = (left + right) * 0.5
        side_sample = (left - right) * 0.5
        mid_energy += mid_sample * mid_sample
        side_energy += side_sample * side_sample
    stereo = math.sqrt(side_energy / max(mid_energy, EPSILON))

    return AudioFeatures(
        rms_dbfs=max(linear_to_db(rms), -120.0),
        peak_dbfs=max(linear_to_db(peak), -120.0),
        low_band_energy=low / total,
        mid_band_energy=mid / total,
        high_band_energy=high / total,
        stereo_width=stereo,
        clipping_ratio=clipping,
    )


def normalize_loudness(buffer: AudioBuffer, profile: EnhancementProfile) -> AudioBuffer:
    """Apply bounded RMS normalization similar to a low-latency loudness stage."""

    features = extract_features(buffer)
    gain_db = profile.target_rms_dbfs - features.rms_dbfs
    gain_db = max(-profile.max_gain_db, min(profile.max_gain_db, gain_db))
    return buffer.apply_gain(db_to_linear(gain_db))


def tone_shape(
    buffer: AudioBuffer,
    profile: EnhancementProfile,
    *,
    clarity: float,
    warmth: float,
    air: float,
) -> AudioBuffer:
    """Apply a simple three-band tilt using one-pole filters."""

    low_alpha = min(1.0, 180.0 / buffer.sample_rate)
    high_alpha = min(1.0, 4000.0 / buffer.sample_rate)
    low_state = [0.0, 0.0]
    high_state = [0.0, 0.0]
    shaped_samples: list[Sample] = []

    bass_gain = db_to_linear(profile.bass_boost_db * warmth)
    presence_gain = db_to_linear(profile.presence_boost_db * clarity)
    air_gain = db_to_linear(profile.air_boost_db * air)

    for frame in buffer.samples:
        shaped_frame = []
        for channel, sample in enumerate(frame):
            low_state[channel] += low_alpha * (sample - low_state[channel])
            high_state[channel] += high_alpha * (sample - high_state[channel])
            low = low_state[channel]
            high = sample - high_state[channel]
            mid = sample - low - high
            shaped_frame.append((low * bass_gain) + (mid * presence_gain) + (high * air_gain))
        shaped_samples.append((shaped_frame[0], shaped_frame[1]))

    return AudioBuffer(buffer.sample_rate, shaped_samples)


def enhance_transients(buffer: AudioBuffer, amount: float) -> AudioBuffer:
    """Restore a small amount of attack on heavily compressed material."""

    if amount <= 0.0:
        return buffer

    previous = (0.0, 0.0)
    enhanced: list[Sample] = []
    for left, right in buffer.samples:
        left_attack = left - previous[0]
        right_attack = right - previous[1]
        enhanced.append((left + left_attack * amount, right + right_attack * amount))
        previous = (left, right)
    return AudioBuffer(buffer.sample_rate, enhanced)


def adjust_stereo_width(buffer: AudioBuffer, width: float) -> AudioBuffer:
    """Adjust stereo width through mid/side processing."""

    adjusted: list[Sample] = []
    for left, right in buffer.samples:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        adjusted.append((mid + side, mid - side))
    return AudioBuffer(buffer.sample_rate, adjusted)


def true_peak_limiter(buffer: AudioBuffer, ceiling_dbfs: float = -1.0) -> AudioBuffer:
    """Limit peaks with a soft knee and final hard ceiling."""

    ceiling = db_to_linear(ceiling_dbfs)
    limited_samples: list[Sample] = []
    for frame in buffer.samples:
        limited_frame = []
        for sample in frame:
            sign = -1.0 if sample < 0.0 else 1.0
            magnitude = abs(sample)
            if magnitude > ceiling:
                overshoot = magnitude - ceiling
                magnitude = ceiling + math.tanh(overshoot) * (1.0 - ceiling)
            limited_frame.append(sign * min(magnitude, ceiling))
        limited_samples.append((limited_frame[0], limited_frame[1]))
    return AudioBuffer(buffer.sample_rate, limited_samples)
