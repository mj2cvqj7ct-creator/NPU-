import numpy as np

from sxnpu_audio_enhancer.config import EnhancementConfig
from sxnpu_audio_enhancer.dsp import (
    DynamicEq,
    LoudnessNormalizer,
    TruePeakLimiter,
    db_to_linear,
)
from sxnpu_audio_enhancer.pipeline import EnhancementPipeline


def stereo_sine(amplitude: float = 0.25, frames: int = 4800) -> np.ndarray:
    t = np.arange(frames, dtype=np.float32) / 48_000.0
    wave = amplitude * np.sin(2.0 * np.pi * 440.0 * t)
    return np.column_stack([wave, wave]).astype(np.float32)


def test_true_peak_limiter_caps_output() -> None:
    limiter = TruePeakLimiter(ceiling_dbfs=-1.0)
    audio = np.array([[1.2, -1.2], [0.5, -0.5]], dtype=np.float32)

    limited = limiter.process(audio)

    assert np.max(np.abs(limited)) <= db_to_linear(-1.0) + 1e-6


def test_loudness_normalizer_raises_quiet_signal_without_clipping() -> None:
    normalizer = LoudnessNormalizer(target_lufs=-18.0, max_gain_db=12.0)
    quiet = stereo_sine(amplitude=0.02)

    processed = normalizer.process(quiet)

    assert np.sqrt(np.mean(processed**2)) > np.sqrt(np.mean(quiet**2))
    assert np.max(np.abs(processed)) <= 1.0


def test_dynamic_eq_preserves_shape_and_float32() -> None:
    eq = DynamicEq(sample_rate=48_000, low_shelf_db=2.0, presence_db=1.5, air_db=1.0)
    audio = stereo_sine()

    processed = eq.process(audio)

    assert processed.shape == audio.shape
    assert processed.dtype == np.float32
    assert np.all(np.isfinite(processed))


def test_pipeline_uses_cpu_fallback_when_model_is_absent() -> None:
    config = EnhancementConfig(model_path=None)
    pipeline = EnhancementPipeline(config)
    audio = stereo_sine(amplitude=0.1)

    processed = pipeline.process(audio)

    assert processed.shape == audio.shape
    assert processed.dtype == np.float32
    assert pipeline.backend.name == "cpu"
    assert np.max(np.abs(processed)) <= db_to_linear(config.limiter_ceiling_dbfs) + 1e-6
