"""NPU backend selection and lightweight inference adapters."""

from .backend import InferenceResult, SnapdragonNpuBackendSelector

__all__ = ["InferenceResult", "SnapdragonNpuBackendSelector"]
