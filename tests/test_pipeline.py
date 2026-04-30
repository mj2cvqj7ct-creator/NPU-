import numpy as np

from snapdragon_audio_enhancer.config import EnhancementConfig
from snapdragon_audio_enhancer.inference import CpuFallbackBackend
from snapdragon_audio_enhancer.pipeline import EnhancementPipeline, extract_features


def _sine(sample_rate: int = 48_000, seconds: float = 0.08) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    left = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    right = 0.20 * np.sin(2.0 * np.pi * 880.0 * t)
    return np.column_stack((left, right)).astype(np.float32)


def test_pipeline_keeps_shape_and_limits_peak() -> None:
    config = EnhancementConfig.for_service("spotify")
    pipeline = EnhancementPipeline(config, CpuFallbackBackend())
    result = pipeline.process(_sine(), sample_rate=48_000)

    assert result.shape == (3840, 2)
    assert result.dtype == np.float32
    assert np.max(np.abs(result)) <= 10 ** (config.true_peak_dbfs / 20.0) + 1e-6


def test_pipeline_supports_mono_audio() -> None:
    config = EnhancementConfig.for_service("apple_music")
    pipeline = EnhancementPipeline(config, CpuFallbackBackend())
    mono = _sine()[:, 0]
    result = pipeline.process(mono, sample_rate=48_000)

    assert result.shape == (3840, 1)
    assert np.isfinite(result).all()


def test_extract_features_reports_bounded_density() -> None:
    features = extract_features(_sine(), 48_000)

    assert -120.0 < features["rms_dbfs"] < 0.0
    assert 0.0 <= features["spectral_density"] <= 1.0
