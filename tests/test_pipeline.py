import numpy as np

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame
from snapdragon_npu_audio_enhancer.dsp import EnhancementControls
from snapdragon_npu_audio_enhancer.pipeline import EnhancementPipeline


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


def test_pipeline_uses_service_profile_defaults():
    frame = AudioFrame(np.full((1024, 2), 0.2, dtype=np.float32), sample_rate=48000)
    pipeline = EnhancementPipeline.for_service("youtube-music")

    result = pipeline.process_frame(frame)

    assert np.all(np.isfinite(result.samples))
    assert pipeline.service_profile.service.value == "youtube-music"
    assert pipeline.effective_npu_mix == pipeline.service_profile.npu_mix
    assert pipeline.last_controls is not None
    assert pipeline.last_controls.limiter_ceiling_db <= pipeline.service_profile.limiter_ceiling_db
