"""Snapdragon X NPU audio enhancement prototype.

The package exposes a service-agnostic PCM post-processing pipeline. It does
not integrate with or modify Spotify, Apple Music, YouTube Music, or their
protected streams; callers provide decoded stereo PCM frames captured from the
OS audio path.
"""

from .dsp import AudioEnhancementPipeline, EnhancementSettings, StereoFrame
from .inference import (
    InferenceBackend,
    InferenceRequest,
    run_personalization_inference,
    select_backend,
)
from .profiles import ServiceProfile, get_service_profile

__all__ = [
    "AudioEnhancementPipeline",
    "EnhancementSettings",
    "InferenceBackend",
    "InferenceRequest",
    "ServiceProfile",
    "StereoFrame",
    "get_service_profile",
    "run_personalization_inference",
    "select_backend",
]
