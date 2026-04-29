import numpy as np

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame
from snapdragon_npu_audio_enhancer.dsp import EnhancementControls
from snapdragon_npu_audio_enhancer.pipeline import EnhancementPipeline
from snapdragon_npu_audio_enhancer.profiles import get_service_profile


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


def test_service_profile_changes_snapdragon_npu_control_mix():
    t = np.linspace(0.0, 0.2, 9600, endpoint=False)
    left = 0.45 * np.sin(2 * np.pi * 120.0 * t) + 0.1 * np.sin(2 * np.pi * 3200.0 * t)
    right = 0.42 * np.sin(2 * np.pi * 120.0 * t + 0.05)
    frame = AudioFrame(np.column_stack([left, right]), sample_rate=48000)

    balanced = EnhancementPipeline(
        inference_backend=StaticBackend(),
        service_profile=get_service_profile("balanced"),
    )
    snapdragon = EnhancementPipeline(
        inference_backend=StaticBackend(),
        service_profile=get_service_profile("snapdragon-x-npu"),
    )

    balanced.process_frame(frame)
    snapdragon.process_frame(frame)

    assert snapdragon.last_controls is not None
    assert balanced.last_controls is not None
    assert snapdragon.last_controls.presence_gain_db > balanced.last_controls.presence_gain_db
    assert snapdragon.last_controls.transient_gain_db > balanced.last_controls.transient_gain_db
    assert snapdragon.last_controls.stereo_width > balanced.last_controls.stereo_width
