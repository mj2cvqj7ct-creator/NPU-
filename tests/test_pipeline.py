import math
import unittest

from snapdragon_audio_enhancer import AudioEnhancementPipeline, MusicService, get_profile
from snapdragon_audio_enhancer.dsp import db_to_linear, measure_frame
from snapdragon_audio_enhancer.npu import EnhancementControls


class FixedModel:
    def __init__(self, controls: EnhancementControls) -> None:
        self.controls = controls

    def infer(self, features):
        return self.controls


def sine_frame(length: int = 960, amplitude: float = 0.08):
    return [
        (
            amplitude * math.sin(2.0 * math.pi * index / 48.0),
            amplitude * math.sin(2.0 * math.pi * index / 48.0 + 0.2),
        )
        for index in range(length)
    ]


class AudioEnhancementPipelineTests(unittest.TestCase):
    def test_pipeline_preserves_frame_length_and_limiter_ceiling(self):
        controls = EnhancementControls(
            clarity=0.8,
            warmth=0.6,
            stereo_width=1.08,
            transient_restore=0.28,
            loudness_boost_db=2.0,
        )
        pipeline = AudioEnhancementPipeline(
            service=MusicService.SPOTIFY,
            npu_model=FixedModel(controls),
        )

        result = pipeline.process_frame(sine_frame(amplitude=0.9))

        self.assertEqual(len(result.frame), 960)
        ceiling = db_to_linear(result.applied_settings.limiter_ceiling_dbfs)
        peak = max(abs(sample) for pair in result.frame for sample in pair)
        self.assertLessEqual(peak, ceiling)

    def test_empty_frame_is_supported(self):
        pipeline = AudioEnhancementPipeline(service="unknown-service")

        result = pipeline.process_frame([])

        self.assertEqual(result.frame, [])
        self.assertEqual(result.features.rms, 0.0)

    def test_service_profile_changes_loudness_target(self):
        spotify = get_profile("spotify")
        apple_music = get_profile("apple_music")

        self.assertIs(spotify.service, MusicService.SPOTIFY)
        self.assertIs(apple_music.service, MusicService.APPLE_MUSIC)
        self.assertGreater(spotify.loudness_target_lufs, apple_music.loudness_target_lufs)

    def test_measurement_extracts_channel_imbalance_and_density(self):
        frame = [(0.25, 0.05), (-0.25, -0.05)] * 100

        metrics = measure_frame(frame)

        self.assertGreater(metrics.channel_imbalance, 0.7)
        self.assertGreaterEqual(metrics.spectral_density, 0.0)
        self.assertLessEqual(metrics.spectral_density, 1.0)


if __name__ == "__main__":
    unittest.main()
