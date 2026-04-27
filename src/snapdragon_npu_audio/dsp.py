"""Low-latency DSP primitives for service-agnostic music enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .backends import InferenceBackend
from .frames import AudioFrame


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class EnhancementProfile:
    """User and device tuning applied consistently across music services."""

    bass_boost: float = 0.0
    vocal_clarity: float = 0.0
    stereo_width: float = 1.0
    target_rms: float = 0.18
    limiter_ceiling: float = 0.98

    def __post_init__(self) -> None:
        if self.target_rms <= 0.0:
            raise ValueError("target_rms must be positive")
        if not 0.0 < self.limiter_ceiling <= 1.0:
            raise ValueError("limiter_ceiling must be in the interval (0, 1]")


class AudioEnhancer:
    """Frame-based enhancement chain intended for 10-20 ms realtime buffers."""

    def __init__(
        self,
        backend: InferenceBackend,
        profile: EnhancementProfile | None = None,
    ) -> None:
        self.backend = backend
        self.profile = profile or EnhancementProfile()

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Enhance one PCM frame without changing its shape or sample rate."""

        inferred = self.backend.process(frame)
        tuned_profile = self._profile_from_signal(inferred)
        samples = self._normalize_loudness(frame.samples, tuned_profile)
        samples = self._apply_tone_shape(samples, frame.channels, tuned_profile)
        samples = self._adjust_stereo_width(samples, tuned_profile.stereo_width)
        samples = self._limit(samples, tuned_profile.limiter_ceiling)
        return frame.with_samples(samples)

    def _profile_from_signal(self, frame: AudioFrame) -> EnhancementProfile:
        peak = max((abs(sample) for sample in frame.samples), default=0.0)
        rms = self._rms(frame.samples)
        density = _clamp(rms / peak if peak > 1e-9 else 0.0, 0.0, 1.0)
        brightness = self._zero_crossing_rate(frame)
        vocal_presence = 1.0 - abs(0.5 - brightness) * 2.0

        return EnhancementProfile(
            bass_boost=_clamp(self.profile.bass_boost + (0.5 - density) * 0.08, -0.25, 0.25),
            vocal_clarity=_clamp(
                self.profile.vocal_clarity + (vocal_presence - brightness) * 0.10,
                -0.20,
                0.20,
            ),
            stereo_width=_clamp(self.profile.stereo_width + (0.5 - density) * 0.12, 0.75, 1.25),
            target_rms=self.profile.target_rms,
            limiter_ceiling=self.profile.limiter_ceiling,
        )

    def _rms(self, samples: tuple[float, ...]) -> float:
        if not samples:
            return 0.0
        return sqrt(sum(sample * sample for sample in samples) / len(samples))

    def _zero_crossing_rate(self, frame: AudioFrame) -> float:
        crossings = 0
        comparisons = 0
        for channel in range(frame.channels):
            previous = None
            for sample in frame.channel_samples(channel):
                if previous is not None:
                    crossings += int((previous < 0.0 <= sample) or (sample < 0.0 <= previous))
                    comparisons += 1
                previous = sample
        if comparisons == 0:
            return 0.0
        return _clamp(crossings / comparisons, 0.0, 1.0)

    def _normalize_loudness(
        self,
        samples: tuple[float, ...],
        profile: EnhancementProfile,
    ) -> tuple[float, ...]:
        if not samples:
            return samples

        rms = self._rms(samples)
        if rms <= 1e-9:
            return samples

        gain = _clamp(profile.target_rms / rms, 0.25, 4.0)
        return tuple(_clamp(sample * gain, -1.0, 1.0) for sample in samples)

    def _apply_tone_shape(
        self,
        samples: tuple[float, ...],
        channels: int,
        profile: EnhancementProfile,
    ) -> tuple[float, ...]:
        if not samples:
            return samples

        shaped: list[float] = []
        previous_by_channel = list(samples[:channels])
        low_mix = _clamp(profile.bass_boost, -0.25, 0.25)
        clarity_mix = _clamp(profile.vocal_clarity, -0.20, 0.20)

        for index, sample in enumerate(samples):
            channel = index % channels
            previous = previous_by_channel[channel]
            low_band = (sample + previous) * 0.5
            transient = sample - low_band
            shaped.append(_clamp(sample + low_band * low_mix + transient * clarity_mix, -1.0, 1.0))
            previous_by_channel[channel] = sample

        return tuple(shaped)

    def _adjust_stereo_width(
        self,
        samples: tuple[float, ...],
        width: float,
    ) -> tuple[float, ...]:
        if len(samples) < 2:
            return samples

        adjusted: list[float] = []
        safe_width = _clamp(width, 0.5, 1.5)
        iterator = iter(samples)
        for left, right in zip(iterator, iterator):
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * safe_width
            adjusted.extend((_clamp(mid + side, -1.0, 1.0), _clamp(mid - side, -1.0, 1.0)))

        return tuple(adjusted)

    def _limit(
        self,
        samples: tuple[float, ...],
        ceiling: float,
    ) -> tuple[float, ...]:
        peak = max((abs(sample) for sample in samples), default=0.0)
        if peak <= ceiling or peak <= 1e-9:
            return samples

        gain = ceiling / peak
        return tuple(sample * gain for sample in samples)
