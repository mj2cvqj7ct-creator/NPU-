from __future__ import annotations

from dataclasses import dataclass

from .audio_frame import AudioFrame
from .dsp import DspChain, EnhancementSettings, LoudnessStats
from .inference import EnhancementHints, InferenceProvider, select_provider


@dataclass(frozen=True)
class EnhancementConfig:
    target_rms_dbfs: float = -18.0
    max_loudness_gain_db: float = 9.0
    limiter_ceiling_dbfs: float = -1.0
    model_path: str | None = None


@dataclass(frozen=True)
class ProcessingReport:
    provider: str
    loudness: LoudnessStats
    hints: EnhancementHints


class EnhancementPipeline:
    """Service-neutral PCM post-processing pipeline for streaming apps."""

    def __init__(
        self,
        config: EnhancementConfig | None = None,
        provider: InferenceProvider | None = None,
    ) -> None:
        self.config = config or EnhancementConfig()
        self.provider = provider or select_provider(self.config.model_path)
        self._last_report: ProcessingReport | None = None
        self._chain: DspChain | None = None

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Enhance one 10-20 ms frame while preserving format metadata."""
        frame.validate()
        hints = self.provider.predict(frame)
        settings = self._make_settings(hints)
        chain = self._get_chain(settings, frame.sample_rate, frame.channels)
        processed, loudness = chain.process(frame)
        self._last_report = ProcessingReport(
            provider=self.provider.kind.value,
            loudness=loudness,
            hints=hints,
        )
        return processed

    @property
    def last_report(self) -> ProcessingReport | None:
        return self._last_report

    def _get_chain(
        self,
        settings: EnhancementSettings,
        sample_rate: int = 48_000,
        channels: int = 2,
    ) -> DspChain:
        if (
            self._chain is None
            or self._chain.sample_rate != sample_rate
            or self._chain.channels != channels
            or self._chain.settings != settings
        ):
            self._chain = DspChain(sample_rate=sample_rate, channels=channels, settings=settings)
        return self._chain

    def _make_settings(self, hints: EnhancementHints) -> EnhancementSettings:
        return EnhancementSettings(
            target_rms_dbfs=self.config.target_rms_dbfs,
            max_loudness_gain_db=self.config.max_loudness_gain_db,
            limiter_ceiling_dbfs=hints.limiter_ceiling_dbfs,
            stereo_width=hints.stereo_width,
            vocal_presence_gain_db=hints.clarity_db,
            bass_shelf_gain_db=hints.bass_db,
        )
