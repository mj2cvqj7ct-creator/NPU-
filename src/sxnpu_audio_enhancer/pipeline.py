"""Frame-oriented enhancement pipeline for captured PCM audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import EnhancerConfig
from .dsp import DynamicEq, LoudnessNormalizer, TruePeakLimiter, analyze_audio
from .inference import (
    BackendKind,
    EnhancementBackend,
    InferenceConfig,
    InferenceProvider,
    create_inference_backend,
)


class AudioEnhancer:
    """Applies deterministic DSP plus optional NPU-assisted enhancement."""

    def __init__(
        self,
        config: EnhancerConfig | None = None,
        backend: EnhancementBackend | None = None,
    ) -> None:
        self.config = config or EnhancerConfig()
        self.backend = backend or create_inference_backend(
            InferenceConfig(model_path=self.config.model_path)
        )
        self._normalizer = LoudnessNormalizer(self.config)
        self._eq = DynamicEq(self.config)
        self._limiter = TruePeakLimiter(self.config)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Enhance a stereo float PCM buffer and keep it under true peak."""

        frame = np.asarray(audio, dtype=np.float32)
        metrics = analyze_audio(frame)
        normalized = self._normalizer.process(frame, metrics)
        neural = self.backend.enhance(normalized)
        enhanced = self._eq.process(neural)
        return self._limiter.process(enhanced)


class AudioEnhancementPipeline(AudioEnhancer):
    """Compatibility wrapper for CLI-oriented provider selection."""

    def __init__(
        self,
        config: EnhancerConfig | None = None,
        backend: EnhancementBackend | None = None,
        *,
        provider_preference: InferenceProvider = InferenceProvider.AUTO,
        model_path: str | Path | None = None,
    ) -> None:
        config = config or EnhancerConfig()
        if backend is None:
            backend_kind = BackendKind(provider_preference.value)
            backend = create_inference_backend(
                InferenceConfig(
                    preferred_backend=backend_kind,
                    model_path=model_path or config.model_path,
                )
            )
        super().__init__(config=config, backend=backend)


EnhancementPipeline = AudioEnhancer
