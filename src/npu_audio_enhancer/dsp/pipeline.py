from __future__ import annotations

from dataclasses import dataclass

from npu_audio_enhancer.dsp.eq import DynamicToneShaper, ToneProfile
from npu_audio_enhancer.dsp.frame import AudioFrame
from npu_audio_enhancer.dsp.loudness import LoudnessNormalizer, LoudnessStats
from npu_audio_enhancer.dsp.limiter import TruePeakLimiter


@dataclass(frozen=True)
class EnhancementReport:
    input_loudness: LoudnessStats
    applied_gain_db: float
    limited_samples: int
    service_profile: str
    npu_backend: str


@dataclass(frozen=True)
class EnhancementConfig:
    target_lufs: float = -16.0
    max_gain_db: float = 6.0
    true_peak_ceiling: float = 0.98
    presence_db: float = 1.5
    bass_db: float = 0.8
    air_db: float = 0.7
    stereo_width: float = 1.04


class AudioEnhancementPipeline:
    """Low-latency deterministic DSP chain used before/after NPU inference."""

    def __init__(self, config: EnhancementConfig | None = None) -> None:
        self.config = config or EnhancementConfig()
        self._normalizer = LoudnessNormalizer(
            target_lufs=self.config.target_lufs,
            max_gain_db=self.config.max_gain_db,
        )
        self._eq = DynamicToneShaper(
            ToneProfile(
                bass_db=self.config.bass_db,
                presence_db=self.config.presence_db,
                air_db=self.config.air_db,
                stereo_width=self.config.stereo_width,
            )
        )
        self._limiter = TruePeakLimiter(ceiling=self.config.true_peak_ceiling)

    def process(
        self,
        frame: AudioFrame,
        *,
        service_profile: str,
        npu_backend: str,
        neural_gain: float = 1.0,
    ) -> tuple[AudioFrame, EnhancementReport]:
        normalized, loudness, gain_db = self._normalizer.process(frame)
        equalized = self._eq.process(normalized, clarity_hint=max(0.0, neural_gain - 1.0))
        neural_scaled = equalized.with_samples(
            [sample * neural_gain for sample in channel] for channel in equalized.samples
        )
        limited, limited_samples = self._limiter.process(neural_scaled)
        report = EnhancementReport(
            input_loudness=loudness,
            applied_gain_db=gain_db,
            limited_samples=limited_samples,
            service_profile=service_profile,
            npu_backend=npu_backend,
        )
        return limited, report
