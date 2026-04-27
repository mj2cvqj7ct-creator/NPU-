from __future__ import annotations

import math
import unittest

from npu_audio_enhancer import AudioFrame, CpuAdaptiveBackend, EnhancementPipeline, service_profile
from npu_audio_enhancer.inference import BackendMode, create_backend


def sine_frame(amplitude: float, frequency: float = 440.0, seconds: float = 0.05) -> AudioFrame:
    sample_rate = 48_000
    count = int(sample_rate * seconds)
    samples = [amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate) for index in range(count)]
    return AudioFrame.from_mono(sample_rate, samples)


class EnhancementPipelineTests(unittest.TestCase):
    def test_quiet_audio_is_lifted_without_clipping(self) -> None:
        frame = sine_frame(0.04)
        pipeline = EnhancementPipeline.for_service("spotify", backend=CpuAdaptiveBackend())

        enhanced, report = pipeline.process(frame)

        self.assertGreater(enhanced.rms, frame.rms)
        self.assertLessEqual(enhanced.peak, 10 ** (pipeline.profile.limiter_ceiling_db / 20.0) + 1e-9)
        self.assertEqual(report.backend, "cpu-adaptive")
        self.assertEqual(report.profile, "spotify")

    def test_limiter_keeps_hot_audio_below_ceiling(self) -> None:
        frame = sine_frame(0.98)
        pipeline = EnhancementPipeline.for_service("youtube-music", backend=CpuAdaptiveBackend())

        enhanced, _ = pipeline.process(frame)

        ceiling = 10 ** (pipeline.profile.limiter_ceiling_db / 20.0)
        self.assertLessEqual(enhanced.peak, ceiling + 1e-9)

    def test_service_profiles_are_distinct(self) -> None:
        spotify = service_profile("spotify")
        apple = service_profile("apple-music")
        youtube = service_profile("youtube-music")

        self.assertNotEqual(spotify.target_rms_db, apple.target_rms_db)
        self.assertGreater(youtube.clarity_boost_db, apple.clarity_boost_db)

    def test_auto_backend_uses_cpu_without_qnn_flag(self) -> None:
        backend = create_backend(BackendMode.AUTO)

        self.assertEqual(backend.name, "cpu-adaptive")


if __name__ == "__main__":
    unittest.main()
