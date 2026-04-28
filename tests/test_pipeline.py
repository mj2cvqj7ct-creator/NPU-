from __future__ import annotations

import math
import unittest

from npu_audio_enhancer import AudioEnhancer, EnhancerConfig, ServiceProfile
from npu_audio_enhancer.dsp import FeatureExtractor, TruePeakLimiter
from npu_audio_enhancer.inference import BackendKind


def sine_frame(samples: int = 960, amplitude: float = 0.1) -> list[tuple[float, float]]:
    return [
        (
            amplitude * math.sin(index * 2.0 * math.pi / 48.0),
            amplitude * math.sin(index * 2.0 * math.pi / 48.0),
        )
        for index in range(samples)
    ]


class PipelineTests(unittest.TestCase):
    def test_pipeline_boosts_quiet_audio_without_clipping(self) -> None:
        enhancer = AudioEnhancer(EnhancerConfig(service=ServiceProfile.SPOTIFY))
        processed, report = enhancer.process_frame(sine_frame(amplitude=0.05))

        self.assertEqual(len(processed), 960)
        self.assertIs(report.service, ServiceProfile.SPOTIFY)
        self.assertLessEqual(report.output_peak, 0.98)
        self.assertGreater(
            FeatureExtractor().analyze(processed).rms,
            FeatureExtractor().analyze(sine_frame(amplitude=0.05)).rms,
        )

    def test_pipeline_leaves_silence_silent(self) -> None:
        enhancer = AudioEnhancer()
        processed, report = enhancer.process_frame([(0.0, 0.0)] * 128)

        self.assertEqual(processed, [(0.0, 0.0)] * 128)
        self.assertEqual(report.decision.gain, 1.0)

    def test_limiter_scales_stereo_frame_instead_of_hard_clipping(self) -> None:
        limited = TruePeakLimiter(ceiling=0.8).process([(1.0, 0.5), (-0.25, -2.0)])

        self.assertEqual(limited, [(0.8, 0.4), (-0.1, -0.8)])

    def test_forced_qnn_falls_back_to_cpu_when_provider_is_missing(self) -> None:
        enhancer = AudioEnhancer(EnhancerConfig(preferred_backend=BackendKind.QNN))

        self.assertIn(enhancer.inference_backend.kind, {BackendKind.QNN, BackendKind.CPU})
        if enhancer.inference_backend.kind is BackendKind.CPU:
            self.assertIn("QNN provider unavailable", enhancer.inference_backend.reason)

    def test_service_profiles_are_distinct(self) -> None:
        spotify = AudioEnhancer(EnhancerConfig(service=ServiceProfile.SPOTIFY))
        apple = AudioEnhancer(EnhancerConfig(service=ServiceProfile.APPLE_MUSIC))
        frame = sine_frame(amplitude=0.08)

        _, spotify_report = spotify.process_frame(frame)
        _, apple_report = apple.process_frame(frame)

        self.assertNotEqual(spotify_report.decision.bass_tilt, apple_report.decision.bass_tilt)
        self.assertNotEqual(spotify_report.decision.presence_tilt, apple_report.decision.presence_tilt)


if __name__ == "__main__":
    unittest.main()
