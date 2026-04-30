from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio_frame import AudioFrame, ensure_stereo
from .dsp import AudioFeatures, EnhancementControls, FeatureExtractor, RuleBasedEnhancer, merge_controls
from .inference import InferenceBackend, build_backend
from .service_profiles import ServiceProfile, get_service_profile


@dataclass
class EnhancementPipeline:
    """Low-latency enhancement chain for WASAPI-style PCM frames."""

    inference_backend: InferenceBackend = field(default_factory=build_backend)
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    enhancer: RuleBasedEnhancer = field(default_factory=RuleBasedEnhancer)
    service_profile: ServiceProfile = field(default_factory=lambda: get_service_profile(None))
    npu_mix: float | None = None

    last_features: AudioFeatures | None = None
    last_controls: EnhancementControls | None = None

    def __post_init__(self) -> None:
        self.enhancer.target_rms_db = self.service_profile.target_rms_db

    @property
    def effective_npu_mix(self) -> float:
        """Return the configured NPU-control blend for the active profile."""

        return self.service_profile.npu_mix if self.npu_mix is None else self.npu_mix

    @classmethod
    def for_service(
        cls,
        service: str,
        inference_backend: InferenceBackend | None = None,
        npu_mix: float | None = None,
    ) -> "EnhancementPipeline":
        """Create a pipeline with Spotify/Apple/YouTube Music tuning."""

        kwargs = {
            "service_profile": get_service_profile(service),
            "npu_mix": npu_mix,
        }
        if inference_backend is not None:
            kwargs["inference_backend"] = inference_backend
        return cls(**kwargs)

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        frame = ensure_stereo(frame)
        features = self.extractor.extract(frame)
        base_controls = self.enhancer.derive_controls(features)
        inferred_controls = self.inference_backend.infer(features)
        controls = self.service_profile.apply(
            merge_controls(base_controls, inferred_controls, self.effective_npu_mix)
        )

        self.last_features = features
        self.last_controls = controls
        return self.enhancer.process(frame, controls)

    def process(self, frame: AudioFrame, block_size: int = 960) -> AudioFrame:
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        processed = []
        for start in range(0, frame.frame_count, block_size):
            chunk = AudioFrame(frame.samples[start : start + block_size], frame.sample_rate)
            processed.append(self.process_frame(chunk).samples)
        return AudioFrame(samples=np.concatenate(processed, axis=0), sample_rate=frame.sample_rate)
