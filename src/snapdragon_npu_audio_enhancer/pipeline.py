from __future__ import annotations

from dataclasses import dataclass

from .audio_frame import AudioFrame
from .dsp import EnhancementControls, FeatureExtractor, RuleBasedDSP
from .inference import InferenceBackend, InferenceResult, select_backend
from .service_policy import MusicService, ServicePolicy, get_policy


@dataclass(frozen=True)
class EnhancementReport:
    service: str
    backend: str
    input_rms_db: float
    output_rms_db: float
    peak_dbfs: float
    controls: EnhancementControls


class EnhancementPipeline:
    """Service-aware local PCM enhancer for streaming app output.

    The pipeline never talks to Spotify, Apple Music, or YouTube Music APIs.
    It only accepts already-decoded PCM blocks from an OS capture layer.
    """

    def __init__(
        self,
        service: MusicService | str | None = None,
        inference_backend: InferenceBackend | None = None,
        policy: ServicePolicy | None = None,
    ) -> None:
        self.policy = policy or get_policy(service)
        self.inference_backend = inference_backend or select_backend()
        self.extractor = FeatureExtractor()
        self.dsp = RuleBasedDSP()

    @classmethod
    def for_environment(
        cls,
        service: MusicService | str | None = None,
        enable_npu: bool = True,
        model_path: str | None = None,
    ) -> "EnhancementPipeline":
        backend = select_backend(str(model_path) if enable_npu and model_path else None)
        return cls(service=service, inference_backend=backend)

    def process(self, frame: AudioFrame) -> tuple[AudioFrame, EnhancementReport]:
        input_rms = frame.rms_db()
        features = self.extractor.analyze(frame)
        rule_controls = self.dsp.controls_for(features, user_taste=self.policy.bass_weight)
        npu_controls = self.inference_backend.infer(frame)
        controls = self._merge_controls(rule_controls, npu_controls)
        enhanced = self.dsp.process(frame, controls)
        report = EnhancementReport(
            service=self.policy.service.value,
            backend=self.inference_backend.kind.value,
            input_rms_db=input_rms,
            output_rms_db=enhanced.rms_db(),
            peak_dbfs=self.extractor.analyze(enhanced).peak_dbfs,
            controls=controls,
        )
        return enhanced, report

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        enhanced, _ = self.process(frame)
        return enhanced

    def _merge_controls(
        self,
        rule_controls: EnhancementControls,
        npu_controls: InferenceResult,
    ) -> EnhancementControls:
        gain_db = self.policy.loudness_target_lufs - (-18.0) + rule_controls.gain_db
        return EnhancementControls(
            gain_db=max(-6.0, min(6.0, gain_db)),
            low_shelf_db=max(-2.0, min(3.0, rule_controls.low_shelf_db + self.policy.bass_weight + npu_controls.warmth_gain_db)),
            presence_db=max(
                -1.0,
                min(3.0, rule_controls.presence_db + self.policy.clarity * npu_controls.clarity_gain_db),
            ),
            air_db=max(-1.5, min(2.0, rule_controls.air_db + 0.35 * npu_controls.transient_boost_db)),
            stereo_width=max(
                0.8,
                min(1.2, rule_controls.stereo_width + self.policy.stereo_width + (npu_controls.stereo_width - 1.0)),
            ),
            limiter_ceiling_dbfs=self.policy.limiter_ceiling_dbfs,
        )
