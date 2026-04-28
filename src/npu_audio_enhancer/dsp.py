"""Low-latency stereo DSP primitives for the audio enhancement prototype."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


StereoFrame = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EnhancementSettings:
    """Rule-based controls that can run before or after NPU inference."""

    target_rms_dbfs: float = -18.0
    max_gain_db: float = 9.0
    bass_gain_db: float = 1.5
    presence_gain_db: float = 1.0
    stereo_width: float = 1.05
    true_peak_ceiling: float = 0.98


@dataclass(frozen=True)
class FrameAnalysis:
    rms: float
    peak: float
    clipping_samples: int


class AudioEnhancementPipeline:
    """Service-agnostic frame processor for 48 kHz stereo float PCM.

    The implementation intentionally keeps the first stage deterministic and
    dependency-free so it can be tested on non-Snapdragon CI runners. Optional
    NPU models can be inserted later between the preprocess and limiter stages.
    """

    def __init__(
        self,
        settings: EnhancementSettings | None = None,
        sample_rate: int = 48_000,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.settings = settings or EnhancementSettings()
        self.sample_rate = sample_rate
        self._bass_l = 0.0
        self._bass_r = 0.0
        self._presence_l = 0.0
        self._presence_r = 0.0

    def process(self, frame: Iterable[tuple[float, float]]) -> StereoFrame:
        """Enhance one short stereo frame and keep samples inside the ceiling."""

        samples = tuple((float(left), float(right)) for left, right in frame)
        if not samples:
            return ()

        normalized = self._normalize_loudness(samples)
        equalized = self._dynamic_eq(normalized)
        widened = self._protective_stereo_width(equalized)
        return self._limit(widened)

    def analyze(self, frame: Iterable[tuple[float, float]]) -> FrameAnalysis:
        samples = tuple(frame)
        if not samples:
            return FrameAnalysis(rms=0.0, peak=0.0, clipping_samples=0)

        square_sum = 0.0
        peak = 0.0
        clipping = 0
        for left, right in samples:
            for value in (left, right):
                abs_value = abs(value)
                square_sum += value * value
                peak = max(peak, abs_value)
                if abs_value >= 1.0:
                    clipping += 1

        rms = sqrt(square_sum / (len(samples) * 2))
        return FrameAnalysis(rms=rms, peak=peak, clipping_samples=clipping)

    def _normalize_loudness(self, samples: StereoFrame) -> StereoFrame:
        analysis = self.analyze(samples)
        if analysis.rms <= 1e-9:
            return samples

        target = 10 ** (self.settings.target_rms_dbfs / 20.0)
        desired_gain = target / analysis.rms
        max_gain = 10 ** (self.settings.max_gain_db / 20.0)
        gain = min(desired_gain, max_gain)

        # Avoid pushing already hot frames into the limiter too aggressively.
        if analysis.peak * gain > self.settings.true_peak_ceiling:
            gain = self.settings.true_peak_ceiling / analysis.peak

        return tuple((left * gain, right * gain) for left, right in samples)

    def _dynamic_eq(self, samples: StereoFrame) -> StereoFrame:
        bass_gain = 10 ** (self.settings.bass_gain_db / 20.0)
        presence_gain = 10 ** (self.settings.presence_gain_db / 20.0)
        bass_alpha = self._one_pole_alpha(180.0)
        presence_alpha = self._one_pole_alpha(3_000.0)

        processed: list[tuple[float, float]] = []
        for left, right in samples:
            self._bass_l = self._bass_l + bass_alpha * (left - self._bass_l)
            self._bass_r = self._bass_r + bass_alpha * (right - self._bass_r)
            self._presence_l = self._presence_l + presence_alpha * (left - self._presence_l)
            self._presence_r = self._presence_r + presence_alpha * (right - self._presence_r)

            bass_l = self._bass_l
            bass_r = self._bass_r
            presence_l = left - self._presence_l
            presence_r = right - self._presence_r

            enhanced_l = left + (bass_l * (bass_gain - 1.0)) + (presence_l * (presence_gain - 1.0))
            enhanced_r = right + (bass_r * (bass_gain - 1.0)) + (presence_r * (presence_gain - 1.0))
            processed.append((enhanced_l, enhanced_r))

        return tuple(processed)

    def _protective_stereo_width(self, samples: StereoFrame) -> StereoFrame:
        width = max(0.0, min(self.settings.stereo_width, 1.25))
        widened: list[tuple[float, float]] = []
        for left, right in samples:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * width
            widened.append((mid + side, mid - side))
        return tuple(widened)

    def _limit(self, samples: StereoFrame) -> StereoFrame:
        ceiling = max(0.0, min(self.settings.true_peak_ceiling, 1.0))
        peak = self.analyze(samples).peak
        if peak <= ceiling or peak <= 1e-9:
            return samples
        gain = ceiling / peak
        return tuple((left * gain, right * gain) for left, right in samples)

    def _one_pole_alpha(self, cutoff_hz: float) -> float:
        # Stable approximation for short audio frames; exact filter shape is not
        # critical at this rule-based stage.
        return min(1.0, max(0.0, cutoff_hz / self.sample_rate))
