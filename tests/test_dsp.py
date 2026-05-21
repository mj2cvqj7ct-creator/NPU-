import math
import unittest

from npu_audio_enhancer.audio_types import AudioFrame
from npu_audio_enhancer.dsp import AudioEnhancer, EnhancerConfig
from npu_audio_enhancer.service_profiles import MusicService, service_config


class AudioEnhancerTests(unittest.TestCase):
    def test_process_preserves_duration_and_limits_peak(self) -> None:
        frame = AudioFrame.from_iterable(_sine_wave(amplitude=0.95, samples=960))
        enhancer = AudioEnhancer(
            EnhancerConfig(target_loudness_db=-6.0, max_gain_db=12.0, limiter_ceiling=0.9)
        )

        result = enhancer.process(frame)

        self.assertEqual(len(result.samples), len(frame.samples))
        self.assertLessEqual(result.peak(), 0.9000001)

    def test_quiet_frame_receives_bounded_gain(self) -> None:
        frame = AudioFrame.from_iterable(_sine_wave(amplitude=0.02, samples=960))
        enhancer = AudioEnhancer(EnhancerConfig(max_gain_db=6.0))

        result = enhancer.process(frame)

        self.assertGreater(result.peak(), frame.peak())
        self.assertLess(result.peak(), 0.1)

    def test_service_profiles_are_distinct(self) -> None:
        spotify = service_config(MusicService.SPOTIFY)
        apple = service_config(MusicService.APPLE_MUSIC)
        youtube = service_config(MusicService.YOUTUBE_MUSIC)

        self.assertNotEqual(spotify, apple)
        self.assertNotEqual(youtube, apple)


def _sine_wave(amplitude: float, samples: int) -> list[tuple[float, float]]:
    output = []
    for index in range(samples):
        value = amplitude * math.sin(2.0 * math.pi * 440.0 * index / 48_000.0)
        output.append((value, value))
    return output


if __name__ == "__main__":
    unittest.main()
