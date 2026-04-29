from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio_frame import AudioFrame, ensure_stereo
from .dsp import AudioFeatures, EnhancementControls, FeatureExtractor, RuleBasedEnhancer, merge_controls
from .inference import InferenceBackend, build_backend
from .service_profiles import ServiceEnhancementProfile, get_service_profile


@dataclass
class EnhancementPipeline:
    """Low-latency enhancement chain for WASAPI-style PCM frames."""

    inference_backend: InferenceBackend = field(default_factory=build_backend)
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    enhancer: RuleBasedEnhancer = field(default_factory=RuleBasedEnhancer)
    service_profile: ServiceEnhancementProfile = field(default_factory=lambda: get_service_profile("neutral"))

    last_features: AudioFeatures | None = None
    last_controls: EnhancementControls | None = None

    def __post_init__(self) -> None:
        self.enhancer.target_rms_db = self.service_profile.target_rms_db

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        frame = ensure_stereo(frame)
        features = self.extractor.extract(frame)
        base_controls = self.enhancer.derive_controls(features)
        inferred_controls = self.inference_backend.infer(features)
        controls = merge_controls(base_controls, inferred_controls, self.service_profile.npu_mix)
        controls = self.service_profile.tune_controls(controls, features)

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
