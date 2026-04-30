import numpy as np

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame
from snapdragon_npu_audio_enhancer.dsp import EnhancementControls
from snapdragon_npu_audio_enhancer.pipeline import EnhancementPipeline
from snapdragon_npu_audio_enhancer.service_profiles import get_service_profile


class StaticBackend:
    def infer(self, features):
        return EnhancementControls(
            pre_gain_db=0.5,
            bass_gain_db=1.0,
            presence_gain_db=0.7,
            air_gain_db=0.4,
            stereo_width=1.02,
            limiter_ceiling_db=-1.0,
        )


def test_pipeline_preserves_shape_and_limits_peak():
    t = np.linspace(0.0, 1.0, 48000, endpoint=False)
    mono = 0.9 * np.sin(2 * np.pi * 440.0 * t)
    frame = AudioFrame(np.column_stack([mono, mono]), sample_rate=48000)

    pipeline = EnhancementPipeline(inference_backend=StaticBackend())
    result = pipeline.process_frame(frame)

    assert result.samples.shape == frame.samples.shape
    assert result.sample_rate == 48000
    assert np.max(np.abs(result.samples)) <= 1.0


def test_pipeline_handles_silence_without_nan():
    frame = AudioFrame(np.zeros((1024, 2), dtype=np.float32), sample_rate=48000)
    pipeline = EnhancementPipeline()

    result = pipeline.process_frame(frame)

    assert np.all(np.isfinite(result.samples))
    assert np.max(np.abs(result.samples)) == 0.0


def test_service_profile_biases_controls_without_clipping():
    t = np.linspace(0.0, 0.1, 4800, endpoint=False)
    left = 0.65 * np.sin(2 * np.pi * 120.0 * t)
    right = 0.55 * np.sin(2 * np.pi * 123.0 * t)
    frame = AudioFrame(np.column_stack([left, right]), sample_rate=48000)

    pipeline = EnhancementPipeline(
        inference_backend=StaticBackend(),
        service_profile=get_service_profile("youtube_music"),
    )
    result = pipeline.process_frame(frame)

    assert pipeline.enhancer.target_rms_db == -16.5
    assert pipeline.last_controls is not None
    assert pipeline.last_controls.presence_gain_db > 0.7
    assert result.samples.shape == frame.samples.shape
    assert np.max(np.abs(result.samples)) <= 1.0
