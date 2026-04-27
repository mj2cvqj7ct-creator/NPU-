from __future__ import annotations

import numpy as np

from sxnpu_audio_enhancer.config import EnhancerConfig
from sxnpu_audio_enhancer.inference import NoopInferenceBackend
from sxnpu_audio_enhancer.pipeline import AudioEnhancer


def test_pipeline_preserves_shape_and_limits_true_peak() -> None:
    cfg = EnhancerConfig(target_lufs=-18.0, true_peak_limit_db=-3.0)
    enhancer = AudioEnhancer(cfg, NoopInferenceBackend())
    signal = np.ones((4096, 2), dtype=np.float32) * 0.95

    out = enhancer.process(signal)

    assert out.shape == signal.shape
    assert np.max(np.abs(out)) <= cfg.true_peak_linear + 1e-6


def test_pipeline_rejects_mono_input() -> None:
    enhancer = AudioEnhancer(EnhancerConfig(), NoopInferenceBackend())
    mono = np.zeros(256, dtype=np.float32)

    try:
        enhancer.process(mono)
    except ValueError as exc:
        assert "stereo" in str(exc)
    else:
        raise AssertionError("Expected mono input to fail")
