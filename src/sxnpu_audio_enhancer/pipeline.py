"""Frame-oriented enhancement pipeline for captured PCM audio."""

from __future__ import annotations

import numpy as np

from .config import EnhancementConfig
from .dsp import AudioMetrics, DynamicEq, FrameLimiter, LoudnessNormalizer, analyze_audio
from .inference import InferenceBackend, InferenceResult


class AudioEnhancementPipeline:
    """Applies safe DSP plus optional NPU-assisted control inference."""

    def __init__(
        self,
        config: EnhancementConfig | None = None,
        inference_backend: InferenceBackend | None = None,
    ) -> None:
        self.config = config or EnhancementConfig()
        self.inference_backend = inference_backend
        self._normalizer = LoudnessNormalizer(self.config)
        self._eq = DynamicEq(self.config)
        self._limiter = FrameLimiter(self.config)

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, AudioMetrics, InferenceResult | None]:
        """Enhance a mono or stereo float PCM buffer.

        The returned buffer is always clipped to the configured true peak ceiling.
        """

        frame = np.asarray(audio, dtype=np.float32)
        metrics = analyze_audio(frame)
        inference = self.inference_backend.run(metrics) if self.inference_backend else None

        normalized = self._normalizer.process(frame, metrics)
        eq_curve = inference.eq_curve if inference else None
        enhanced = self._eq.process(normalized, eq_curve=eq_curve)
        limited = self._limiter.process(enhanced)
        return limited, analyze_audio(limited), inference
