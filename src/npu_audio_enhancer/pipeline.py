from __future__ import annotations

from dataclasses import dataclass
from math import pow

from .audio import AudioFrame, clamp_sample
from .inference import EnhancementControls, InferenceBackend, create_backend
from .profiles import EnhancementProfile, service_profile


def db_to_linear(db: float) -> float:
    return pow(10.0, db / 20.0)


@dataclass(frozen=True)
class EnhancementReport:
    backend: str
    profile: str
    input_rms_db: float
    input_peak_db: float
    output_rms_db: float
    output_peak_db: float
    loudness_gain_db: float
    bass_gain_db: float
    clarity_gain_db: float
    compression_amount: float


class EnhancementPipeline:
    """Low-latency DSP chain controlled by an adaptive inference backend."""

    def __init__(
        self,
        profile: EnhancementProfile | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.profile = profile or service_profile(None)
        self.backend = backend or create_backend()
        self._bass_state_left = 0.0
        self._bass_state_right = 0.0
        self._clarity_state_left = 0.0
        self._clarity_state_right = 0.0

    @classmethod
    def for_service(cls, service: str | None, backend: InferenceBackend | None = None) -> "EnhancementPipeline":
        return cls(profile=service_profile(service), backend=backend)

    def process(self, frame: AudioFrame) -> tuple[AudioFrame, EnhancementReport]:
        features, controls = self.backend.infer(frame, self.profile)
        enhanced = self._apply_controls(frame, controls)
        report = EnhancementReport(
            backend=self.backend.name,
            profile=self.profile.name,
            input_rms_db=features.rms_db,
            input_peak_db=features.peak_db,
            output_rms_db=enhanced.rms_db,
            output_peak_db=enhanced.peak_db,
            loudness_gain_db=controls.loudness_gain_db,
            bass_gain_db=controls.bass_gain_db,
            clarity_gain_db=controls.clarity_gain_db,
            compression_amount=controls.compression_amount,
        )
        return enhanced, report

    def _apply_controls(self, frame: AudioFrame, controls: EnhancementControls) -> AudioFrame:
        gain = db_to_linear(controls.loudness_gain_db)
        bass_mix = db_to_linear(controls.bass_gain_db) - 1.0
        clarity_mix = db_to_linear(controls.clarity_gain_db) - 1.0
        ceiling = db_to_linear(self.profile.limiter_ceiling_db)
        alpha_bass = 0.025
        alpha_clarity = 0.18
        threshold = 0.62
        ratio = 1.0 + controls.compression_amount * 3.0
        processed: list[tuple[float, float]] = []

        for left, right in frame.samples:
            left, right = self._apply_stereo_width(left, right, controls.stereo_width)

            self._bass_state_left += alpha_bass * (left - self._bass_state_left)
            self._bass_state_right += alpha_bass * (right - self._bass_state_right)
            bass_left = self._bass_state_left
            bass_right = self._bass_state_right

            self._clarity_state_left += alpha_clarity * (left - self._clarity_state_left)
            self._clarity_state_right += alpha_clarity * (right - self._clarity_state_right)
            high_left = left - self._clarity_state_left
            high_right = right - self._clarity_state_right

            left = (left + bass_left * bass_mix + high_left * clarity_mix) * gain
            right = (right + bass_right * bass_mix + high_right * clarity_mix) * gain
            left = self._compress(left, threshold, ratio)
            right = self._compress(right, threshold, ratio)
            processed.append((self._limit(left, ceiling), self._limit(right, ceiling)))

        return AudioFrame(sample_rate=frame.sample_rate, samples=tuple(processed))

    @staticmethod
    def _apply_stereo_width(left: float, right: float, width: float) -> tuple[float, float]:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        return mid + side, mid - side

    @staticmethod
    def _compress(value: float, threshold: float, ratio: float) -> float:
        magnitude = abs(value)
        if magnitude <= threshold:
            return value
        reduced = threshold + (magnitude - threshold) / ratio
        return reduced if value >= 0.0 else -reduced

    @staticmethod
    def _limit(value: float, ceiling: float) -> float:
        return clamp_sample(max(-ceiling, min(ceiling, value)))
