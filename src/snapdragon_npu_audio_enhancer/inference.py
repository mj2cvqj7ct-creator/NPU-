from __future__ import annotations

import math

from .models import AudioAnalysis, EnhancementConfig, Frame


class InferenceProvider:
    """Provider interface for local audio-scene inference."""

    name = "provider"
    available = False

    def enhance(self, frame: Frame, config: EnhancementConfig, analysis: AudioAnalysis) -> Frame:
        raise NotImplementedError


class HeuristicInferenceProvider(InferenceProvider):
    """Deterministic fallback that mirrors intended NPU model controls.

    The production Snapdragon path should load an ONNX model through the QNN
    Execution Provider and write the same conservative DSP control contract.
    """

    name = "heuristic-fallback"
    available = True

    def enhance(self, frame: Frame, config: EnhancementConfig, analysis: AudioAnalysis) -> Frame:
        crest = analysis.peak / max(analysis.rms, 1e-9)
        compression_hint = 1.0 - min(crest / 8.0, 1.0)
        transient_amount = _clamp(config.transient_restore + compression_hint * 0.18, 0.0, 0.35)
        if transient_amount <= 0.0:
            return frame

        enhanced: Frame = []
        previous_mid = (frame[0][0] + frame[0][1]) * 0.5
        for left, right in frame:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            transient = mid - previous_mid
            shaped_mid = mid + transient * transient_amount
            enhanced.append((shaped_mid + side, shaped_mid - side))
            previous_mid = mid
        return enhanced


class SnapdragonNpuProvider(InferenceProvider):
    """Runtime-selecting provider for Snapdragon X deployments.

    This class intentionally avoids importing ONNX Runtime at module import time
    so the repository stays testable on non-ARM64 CI machines. When a model path
    and QNN-capable onnxruntime build are supplied, this is where QNN session
    construction should happen.
    """
    name = "snapdragon-qnn"
    npu_accelerated = False

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._fallback = HeuristicInferenceProvider()
        self.available = self._has_qnn_provider()
        self.npu_accelerated = self.available and self._model_path is not None

    def enhance(self, frame: Frame, config: EnhancementConfig, analysis: AudioAnalysis) -> Frame:
        if not self.available or not self._model_path:
            return self._fallback.enhance(frame, config, analysis)

        # Model-specific tensor names are not part of this prototype yet. Keep
        # the behavior explicit so a missing model cannot silently alter audio.
        return self._fallback.enhance(frame, config, analysis)

    def _has_qnn_provider(self) -> bool:
        if not self._model_path:
            return False

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return False

        providers = getattr(ort, "get_available_providers", lambda: [])()
        return "QNNExecutionProvider" in providers


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
