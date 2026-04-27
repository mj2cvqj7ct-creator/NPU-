"""Rule-based DSP blocks used around the NPU inference boundary."""

from __future__ import annotations

import math

from .audio_types import AudioBuffer, StereoFrame


def normalize_loudness(buffer: AudioBuffer, target_dbfs: float) -> tuple[AudioBuffer, float]:
    rms = buffer.rms
    current_dbfs = linear_to_db(rms)
    gain_db = max(-9.0, min(9.0, target_dbfs - current_dbfs))
    return apply_gain(buffer, db_to_linear(gain_db)), gain_db


def apply_tone_shape(
    buffer: AudioBuffer,
    bass_gain_db: float,
    presence_gain_db: float,
    air_gain_db: float,
) -> AudioBuffer:
    low_alpha = one_pole_alpha(180.0, buffer.sample_rate)
    presence_alpha = one_pole_alpha(2600.0, buffer.sample_rate)
    air_alpha = one_pole_alpha(7600.0, buffer.sample_rate)
    low_state = [0.0, 0.0]
    presence_state = [0.0, 0.0]
    air_state = [0.0, 0.0]
    bass_amount = db_to_linear(bass_gain_db) - 1.0
    presence_amount = db_to_linear(presence_gain_db) - 1.0
    air_amount = db_to_linear(air_gain_db) - 1.0

    shaped: list[StereoFrame] = []
    for left, right in buffer.frames:
        output_channels: list[float] = []
        for channel, sample in enumerate((left, right)):
            low_state[channel] += low_alpha * (sample - low_state[channel])
            presence_state[channel] += presence_alpha * (sample - presence_state[channel])
            air_state[channel] += air_alpha * (sample - air_state[channel])
            presence_band = presence_state[channel] - low_state[channel]
            air_band = sample - air_state[channel]
            output_channels.append(
                sample
                + bass_amount * low_state[channel]
                + presence_amount * presence_band
                + air_amount * air_band
            )
        shaped.append((output_channels[0], output_channels[1]))
    return AudioBuffer(sample_rate=buffer.sample_rate, frames=tuple(shaped))


def apply_transient_restore(buffer: AudioBuffer, amount: float) -> AudioBuffer:
    if not buffer.frames or amount <= 0.0:
        return buffer
    restored: list[StereoFrame] = []
    previous = (0.0, 0.0)
    for left, right in buffer.frames:
        transient_left = left - previous[0]
        transient_right = right - previous[1]
        restored.append((left + transient_left * amount, right + transient_right * amount))
        previous = (left, right)
    return AudioBuffer(sample_rate=buffer.sample_rate, frames=tuple(restored))


def apply_stereo_width(buffer: AudioBuffer, width: float) -> AudioBuffer:
    widened: list[StereoFrame] = []
    safe_width = max(0.85, min(1.18, width))
    for left, right in buffer.frames:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * safe_width
        widened.append((mid + side, mid - side))
    return AudioBuffer(sample_rate=buffer.sample_rate, frames=tuple(widened))


def apply_soft_limiter(buffer: AudioBuffer, ceiling: float) -> tuple[AudioBuffer, int]:
    limited: list[StereoFrame] = []
    reductions = 0
    for left, right in buffer.frames:
        limited_left, reduced_left = _limit_sample(left, ceiling)
        limited_right, reduced_right = _limit_sample(right, ceiling)
        reductions += int(reduced_left) + int(reduced_right)
        limited.append((limited_left, limited_right))
    peak = max((abs(sample) for frame in limited for sample in frame), default=0.0)
    if peak > ceiling:
        scale = ceiling / peak
        limited = [(left * scale, right * scale) for left, right in limited]
    return AudioBuffer(sample_rate=buffer.sample_rate, frames=tuple(limited)), reductions


def apply_gain(buffer: AudioBuffer, gain: float) -> AudioBuffer:
    return AudioBuffer(
        sample_rate=buffer.sample_rate,
        frames=tuple((left * gain, right * gain) for left, right in buffer.frames),
    )


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)


def db_to_linear(value: float) -> float:
    return 10.0 ** (value / 20.0)


def one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    return dt / (rc + dt)


def _limit_sample(sample: float, ceiling: float) -> tuple[float, bool]:
    if abs(sample) <= ceiling:
        return sample, False
    sign = 1.0 if sample >= 0.0 else -1.0
    overshoot = abs(sample) - ceiling
    return sign * (ceiling + math.tanh(overshoot) * (1.0 - ceiling)), True
