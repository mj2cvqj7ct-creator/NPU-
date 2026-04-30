"""Realtime-safe DSP primitives for PCM music enhancement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .audio import AudioBuffer


@dataclass(frozen=True)
class EnhancementProfile:
    """User-tunable parameters for conservative music playback enhancement."""

    target_rms_db: float = -18.0
    max_gain_db: float = 9.0
    low_gain_db: float = 1.5
    presence_gain_db: float = 1.2
    air_gain_db: float = 0.8
    stereo_width: float = 1.05
    limiter_ceiling_db: float = -1.0

    @property
    def vocal_clarity(self) -> float:
        return self.presence_gain_db

    @property
    def bass_weight(self) -> float:
        return self.low_gain_db

    @property
    def air(self) -> float:
        return self.air_gain_db


@dataclass(frozen=True)
class AnalysisFrame:
    rms_db: float
    peak_db: float
    low_energy: float
    mid_energy: float
    high_energy: float


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)


def analyze(buffer: AudioBuffer) -> AnalysisFrame:
    """Extract inexpensive features that can be shared with DSP and NPU stages."""

    samples = [abs(sample) for frame in buffer.frames for sample in frame]
    if not samples:
        return AnalysisFrame(-120.0, -120.0, 0.0, 0.0, 0.0)

    square_sum = sum(sample * sample for sample in samples)
    rms = math.sqrt(square_sum / len(samples))
    peak = max(samples)

    low, mid, high = _estimate_band_energy(buffer)
    total = low + mid + high
    if total <= 1e-12:
        return AnalysisFrame(linear_to_db(rms), linear_to_db(peak), 0.0, 0.0, 0.0)

    return AnalysisFrame(
        rms_db=linear_to_db(rms),
        peak_db=linear_to_db(peak),
        low_energy=low / total,
        mid_energy=mid / total,
        high_energy=high / total,
    )


def enhance(buffer: AudioBuffer, profile: EnhancementProfile | None = None) -> AudioBuffer:
    """Apply a low-latency enhancement chain to interleaved PCM frames."""

    active_profile = profile or EnhancementProfile()
    features = analyze(buffer)
    frames = _copy_frames(buffer.frames)

    frames = _automatic_gain(frames, features, active_profile)
    frames = _three_band_tone(frames, buffer.sample_rate, active_profile)
    frames = _adjust_stereo_width(frames, active_profile.stereo_width)
    frames = _soft_limiter(frames, active_profile.limiter_ceiling_db)
    return AudioBuffer(sample_rate=buffer.sample_rate, frames=_as_stereo_frames(frames))


def _copy_frames(frames: Iterable[Iterable[float]]) -> list[list[float]]:
    return [[float(sample) for sample in frame] for frame in frames]


def _automatic_gain(
    frames: list[list[float]], features: AnalysisFrame, profile: EnhancementProfile
) -> list[list[float]]:
    if features.rms_db <= -90.0:
        return frames

    gain_db = max(-profile.max_gain_db, min(profile.max_gain_db, profile.target_rms_db - features.rms_db))
    gain = db_to_linear(gain_db)
    return [[sample * gain for sample in frame] for frame in frames]


def _three_band_tone(
    frames: list[list[float]], sample_rate: int, profile: EnhancementProfile
) -> list[list[float]]:
    """Use one-pole crossovers to approximate low/presence/air shelves cheaply."""

    if not frames:
        return frames

    low_alpha = _one_pole_alpha(180.0, sample_rate)
    presence_alpha = _one_pole_alpha(3800.0, sample_rate)
    low_gain = db_to_linear(profile.low_gain_db)
    presence_gain = db_to_linear(profile.presence_gain_db)
    air_gain = db_to_linear(profile.air_gain_db)
    low_state = [0.0 for _ in frames[0]]
    presence_state = [0.0 for _ in frames[0]]
    processed: list[list[float]] = []

    for frame in frames:
        out_frame: list[float] = []
        for channel, sample in enumerate(frame):
            low_state[channel] += low_alpha * (sample - low_state[channel])
            low = low_state[channel]
            high_passed = sample - low
            presence_state[channel] += presence_alpha * (high_passed - presence_state[channel])
            presence = presence_state[channel]
            air = high_passed - presence
            out_frame.append((low * low_gain) + (presence * presence_gain) + (air * air_gain))
        processed.append(out_frame)

    return processed


def _adjust_stereo_width(frames: list[list[float]], width: float) -> list[list[float]]:
    if not frames or len(frames[0]) != 2:
        return frames

    constrained_width = max(0.0, min(width, 1.35))
    widened: list[list[float]] = []
    for left, right in frames:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * constrained_width
        widened.append([mid + side, mid - side])
    return widened


def _soft_limiter(frames: list[list[float]], ceiling_db: float) -> list[list[float]]:
    ceiling = db_to_linear(ceiling_db)
    if ceiling <= 0.0:
        ceiling = 0.8912509381337456

    limited: list[list[float]] = []
    for frame in frames:
        limited.append([ceiling * math.tanh(sample / ceiling) for sample in frame])
    return limited


def _one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    cutoff = max(1.0, min(cutoff_hz, sample_rate * 0.45))
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sample_rate
    return dt / (rc + dt)


def _estimate_band_energy(buffer: AudioBuffer) -> tuple[float, float, float]:
    if not buffer.frames:
        return 0.0, 0.0, 0.0

    low_alpha = _one_pole_alpha(180.0, buffer.sample_rate)
    mid_alpha = _one_pole_alpha(3800.0, buffer.sample_rate)
    low_state = [0.0 for _ in range(buffer.channels)]
    mid_state = [0.0 for _ in range(buffer.channels)]
    low_energy = 0.0
    mid_energy = 0.0
    high_energy = 0.0

    for frame in buffer.frames:
        for channel, sample in enumerate(frame):
            low_state[channel] += low_alpha * (sample - low_state[channel])
            low = low_state[channel]
            high_passed = sample - low
            mid_state[channel] += mid_alpha * (high_passed - mid_state[channel])
            mid = mid_state[channel]
            high = high_passed - mid
            low_energy += low * low
            mid_energy += mid * mid
            high_energy += high * high

    return low_energy, mid_energy, high_energy


def _as_stereo_frames(frames: Iterable[Iterable[float]]) -> tuple[tuple[float, float], ...]:
    stereo: list[tuple[float, float]] = []
    for frame in frames:
        values = tuple(float(sample) for sample in frame)
        if len(values) != 2:
            raise ValueError("internal DSP pipeline expects stereo frames")
        stereo.append((values[0], values[1]))
    return tuple(stereo)
