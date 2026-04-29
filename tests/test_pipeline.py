from __future__ import annotations

import numpy as np

from snapdragon_npu_audio_enhancer import AudioFrame, EnhancementPipeline, MusicService
from snapdragon_npu_audio_enhancer.inference import HeuristicNpuSurrogate
from snapdragon_npu_audio_enhancer.profiles import get_service_profile


def _tone(freq: float = 440.0, seconds: float = 0.08, sample_rate: int = 48_000) -> AudioFrame:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    left = 0.18 * np.sin(2.0 * np.pi * freq * t)
    right = 0.16 * np.sin(2.0 * np.pi * (freq * 1.01) * t)
    return AudioFrame(np.column_stack([left, right]), sample_rate)


def test_pipeline_preserves_shape_and_limits_peak() -> None:
    pipeline = EnhancementPipeline(service=MusicService.SPOTIFY)
    output = pipeline.process(_tone(), block_size=960)

    assert output.samples.shape == (3840, 2)
    assert output.peak() <= 10 ** (-1.0 / 20.0) + 1e-6
    assert pipeline.last_profile is not None
    assert pipeline.last_profile.service is MusicService.SPOTIFY


def test_service_profiles_change_inferred_controls() -> None:
    extractor_frame = _tone(freq=120.0)
    pipeline = EnhancementPipeline(service=MusicService.SPOTIFY)
    features = pipeline.extractor.extract(extractor_frame)
    backend = HeuristicNpuSurrogate()

    spotify = backend.infer(features, get_service_profile(MusicService.SPOTIFY))
    apple = backend.infer(features, get_service_profile(MusicService.APPLE_MUSIC))

    assert spotify != apple
    assert spotify.limiter_ceiling_db == -1.0

