"""Low-latency DSP primitives for streaming music enhancement."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .audio_frame import AudioFrame, clamp_sample


@dataclass(frozen=True)
class LoudnessStats:
    """Short-window loudness statistics used by the adaptive pipeline."""

    rms_dbfs: float
    peak_dbfs: float
    gain_db: float


@dataclass(frozen=True)
class EnhancementControls:
    """Control values emitted by NPU or CPU inference for DSP postprocessing."""

    vocal_clarity: float = 0.0
    bass_weight: float = 0.0
    transient_restore: float = 0.0
    stereo_width: float = 0.0
    air: float = 0.0

    def bounded(self) -> "EnhancementControls":
        return EnhancementControls(
            vocal_clarity=_clamp_unit(self.vocal_clarity),
            bass_weight=_clamp_unit(self.bass_weight),
            transient_restore=_clamp_unit(self.transient_restore),
            stereo_width=_clamp_unit(self.stereo_width),
            air=_clamp_unit(self.air),
        )


class LoudnessNormalizer:
    """EBU R128-inspired gain smoother for frame-sized streaming blocks."""

    def __init__(
        self,
        target_dbfs: float = -18.0,
        max_boost_db: float = 6.0,
        max_cut_db: float = 12.0,
        smoothing: float = 0.15,
    ) -> None:
        self.target_dbfs = target_dbfs
        self.max_boost_db = max_boost_db
        self.max_cut_db = max_cut_db
        self.smoothing = smoothing
        self._smoothed_gain_db = 0.0

    def analyze(self, frame: AudioFrame) -> LoudnessStats:
        samples = [sample for stereo in frame.samples for sample in stereo]
        if not samples:
            return LoudnessStats(rms_dbfs=-120.0, peak_dbfs=-120.0, gain_db=0.0)

        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        peak = max(abs(sample) for sample in samples)
        rms_dbfs = linear_to_db(rms)
        peak_dbfs = linear_to_db(peak)
        desired_gain = self.target_dbfs - rms_dbfs
        desired_gain = min(self.max_boost_db, max(-self.max_cut_db, desired_gain))
        self._smoothed_gain_db += (desired_gain - self._smoothed_gain_db) * self.smoothing
        return LoudnessStats(rms_dbfs=rms_dbfs, peak_dbfs=peak_dbfs, gain_db=self._smoothed_gain_db)

    def apply(self, frame: AudioFrame) -> tuple[AudioFrame, LoudnessStats]:
        stats = self.analyze(frame)
        return frame.apply_gain_db(stats.gain_db), stats


class DynamicEqualizer:
    """Small spectral-shaping EQ tuned for compressed streaming music."""

    def __init__(self) -> None:
        self._prev_left = 0.0
        self._prev_right = 0.0

    def process(
        self,
        frame: AudioFrame,
        controls: EnhancementControls,
        service_tilt: float = 1.0,
    ) -> AudioFrame:
        controls = controls.bounded()
        enhanced: list[tuple[float, float]] = []
        bass_gain = 1.0 + 0.10 * controls.bass_weight * service_tilt
        clarity_gain = 1.0 + 0.08 * controls.vocal_clarity * service_tilt
        air_gain = 1.0 + 0.05 * controls.air

        for left, right in frame.samples:
            low_left = 0.985 * self._prev_left + 0.015 * left
            low_right = 0.985 * self._prev_right + 0.015 * right
            high_left = left - low_left
            high_right = right - low_right
            mid_left = left - high_left * 0.35
            mid_right = right - high_right * 0.35

            shaped_left = (low_left * bass_gain) + (mid_left * clarity_gain - low_left) + high_left * air_gain
            shaped_right = (low_right * bass_gain) + (mid_right * clarity_gain - low_right) + high_right * air_gain

            self._prev_left = low_left
            self._prev_right = low_right
            enhanced.append((clamp_sample(shaped_left), clamp_sample(shaped_right)))

        return frame.with_samples(enhanced)


class TransientProtector:
    """Adds conservative attack restoration while avoiding harsh clipping."""

    def __init__(self) -> None:
        self._previous_mid = 0.0

    def process(self, frame: AudioFrame, controls: EnhancementControls) -> AudioFrame:
        amount = 0.06 * _clamp_unit(controls.transient_restore)
        if amount <= 0.0:
            return frame

        restored: list[tuple[float, float]] = []
        for left, right in frame.samples:
            mid = (left + right) * 0.5
            attack = mid - self._previous_mid
            self._previous_mid = mid
            restored.append((clamp_sample(left + attack * amount), clamp_sample(right + attack * amount)))
        return frame.with_samples(restored)


class StereoWidener:
    """Mid-side stereo width adjustment with mono bass protection."""

    def process(self, frame: AudioFrame, controls: EnhancementControls) -> AudioFrame:
        width = 1.0 + 0.12 * _clamp_unit(controls.stereo_width)
        widened: list[tuple[float, float]] = []
        for left, right in frame.samples:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * width
            widened.append((clamp_sample(mid + side), clamp_sample(mid - side)))
        return frame.with_samples(widened)


class TruePeakLimiter:
    """Simple look-ahead-free limiter suitable for 10-20 ms blocks."""

    def __init__(self, ceiling_dbfs: float = -1.0, release: float = 0.08) -> None:
        self.ceiling = db_to_linear(ceiling_dbfs)
        self.release = release
        self._gain = 1.0

    def process(self, frame: AudioFrame) -> AudioFrame:
        limited: list[tuple[float, float]] = []
        for left, right in frame.samples:
            peak = max(abs(left), abs(right), 1e-12)
            target_gain = min(1.0, self.ceiling / peak)
            if target_gain < self._gain:
                self._gain = target_gain
            else:
                self._gain = min(target_gain, self._gain + (1.0 - self._gain) * self.release)
            limited.append((clamp_sample(left * self._gain), clamp_sample(right * self._gain)))
        return frame.with_samples(limited)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    if value <= 1e-12:
        return -120.0
    return 20.0 * math.log10(value)


def mean_abs(samples: Iterable[float]) -> float:
    values = list(samples)
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))
