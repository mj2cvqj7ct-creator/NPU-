from __future__ import annotations

import math

import numpy as np

from npu_audio_enhancer.audio import AudioBuffer
from npu_audio_enhancer.inference import InferenceBackend, NpuFeatureExtractor
from npu_audio_enhancer.pipeline import EnhancementPipeline, EnhancementSettings


def _sine_buffer(amplitude: float = 0.2, seconds: float = 0.2) -> AudioBuffer:
    sample_rate = 48_000
    frames = int(sample_rate * seconds)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    left = amplitude * np.sin(2.0 * math.pi * 440.0 * t)
    right = amplitude * np.sin(2.0 * math.pi * 880.0 * t)
    return AudioBuffer(np.column_stack([left, right]).astype(np.float32), sample_rate)


def test_cpu_fallback_extracts_bounded_features() -> None:
    features = NpuFeatureExtractor(prefer_npu=False).extract(_sine_buffer())

    assert features.backend is InferenceBackend.CPU
    assert 0.0 <= features.clarity <= 1.0
    assert 0.0 <= features.density <= 1.0
    assert 0.0 <= features.bass_weight <= 1.0
    assert 0.0 <= features.transient_risk <= 1.0


def test_runtime_selector_uses_cpu_fallback_without_qnn_model() -> None:
    extractor = NpuFeatureExtractor(prefer_npu=True)

    assert extractor.backend is InferenceBackend.CPU


def test_pipeline_limits_peak_and_preserves_shape() -> None:
    source = _sine_buffer(amplitude=0.95)
    pipeline = EnhancementPipeline(EnhancementSettings(service="spotify"))

    result = pipeline.process(source)

    assert result.audio.samples.shape == source.samples.shape
    assert result.audio.sample_rate_hz == source.sample_rate_hz
    assert np.max(np.abs(result.audio.samples)) <= 0.985
    assert result.metrics.peak_after <= 0.985
    assert result.features.backend is InferenceBackend.CPU


def test_service_profiles_change_rendered_audio() -> None:
    source = _sine_buffer(amplitude=0.15)

    spotify = EnhancementPipeline(EnhancementSettings(service="spotify")).process(source).audio
    youtube = EnhancementPipeline(EnhancementSettings(service="youtube_music")).process(source).audio

    assert not np.allclose(spotify.samples, youtube.samples)
