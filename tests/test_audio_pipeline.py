import math
import unittest

from snapdragon_npu_audio import (
    AudioEnhancementPipeline,
    AudioFrame,
    EnhancementPlan,
    get_service_profile,
)


class FixedBackend:
    name = "fixed-test-backend"

    def __init__(self, plan: EnhancementPlan) -> None:
        self.plan = plan
        self.calls = 0

    def infer(self, features, profile):
        self.calls += 1
        return self.plan


def sine_frame(amplitude: float = 0.2, length: int = 480) -> AudioFrame:
    pairs = []
    for index in range(length):
        phase = (index / length) * math.tau * 4.0
        left = math.sin(phase) * amplitude
        right = math.sin(phase + 0.2) * amplitude
        pairs.append((left, right))
    return AudioFrame.from_stereo_pairs(pairs)


class AudioPipelineTests(unittest.TestCase):
    def test_service_aliases_resolve_to_supported_profiles(self) -> None:
        self.assertEqual(get_service_profile("Spotify").name, "spotify")
        self.assertEqual(get_service_profile("apple music").name, "apple_music")
        self.assertEqual(get_service_profile("YouTubeMusic").name, "youtube_music")
        self.assertEqual(get_service_profile("unknown").name, "generic")

    def test_pipeline_uses_backend_and_preserves_frame_shape(self) -> None:
        plan = EnhancementPlan(
            loudness_gain_db=2.0,
            low_shelf_db=1.0,
            presence_gain_db=0.5,
            compression_ratio=1.2,
            transient_restore=0.1,
            stereo_width=1.03,
        )
        backend = FixedBackend(plan)
        pipeline = AudioEnhancementPipeline(service="spotify", backend=backend)

        result = pipeline.process(sine_frame())

        self.assertEqual(backend.calls, 1)
        self.assertEqual(result.service, "spotify")
        self.assertEqual(result.backend, "fixed-test-backend")
        self.assertEqual(len(result.frame.samples), 960)
        self.assertLessEqual(result.true_peak, get_service_profile("spotify").true_peak_ceiling)

    def test_limiter_keeps_hot_input_under_service_ceiling(self) -> None:
        pipeline = AudioEnhancementPipeline(service="youtube_music")
        frame = AudioFrame.from_stereo_pairs([(1.8, -1.8), (-1.5, 1.5)] * 240)

        result = pipeline.process(frame)

        self.assertLessEqual(result.true_peak, get_service_profile("youtube_music").true_peak_ceiling)

    def test_heuristic_backend_generates_different_service_tuning(self) -> None:
        frame = sine_frame(amplitude=0.08)

        spotify = AudioEnhancementPipeline(service="spotify").process(frame)
        apple = AudioEnhancementPipeline(service="apple_music").process(frame)

        self.assertEqual(spotify.service, "spotify")
        self.assertEqual(apple.service, "apple_music")
        self.assertNotEqual(spotify.low_shelf_db, apple.low_shelf_db)
        self.assertNotEqual(spotify.stereo_width, apple.stereo_width)


if __name__ == "__main__":
    unittest.main()
