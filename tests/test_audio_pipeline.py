import math
import unittest

from npu_audio_enhancer import EnhancementSettings, StreamingEnhancer
from npu_audio_enhancer.dsp.frame import AudioFrame
from npu_audio_enhancer.dsp.limiter import TruePeakLimiter
from npu_audio_enhancer.inference.backend import SnapdragonNpuBackendSelector
from npu_audio_enhancer.profile.model import ListeningPreference, MusicService


class AudioPipelineTests(unittest.TestCase):
    def test_audio_frame_round_trips_interleaved_stereo(self) -> None:
        frame = AudioFrame.from_interleaved([0.1, -0.1, 0.2, -0.2], channels=2)

        self.assertEqual(frame.channels, 2)
        self.assertEqual(frame.sample_count, 2)
        self.assertEqual(frame.to_interleaved(), [0.1, -0.1, 0.2, -0.2])

    def test_limiter_scales_frame_below_true_peak_ceiling(self) -> None:
        frame = AudioFrame([[1.2, -0.4], [-1.1, 0.2]])
        limited, limited_samples = TruePeakLimiter(ceiling=0.95).process(frame)

        self.assertEqual(limited_samples, 2)
        self.assertLessEqual(
            max(abs(sample) for channel in limited.samples for sample in channel),
            0.95,
        )

    def test_streaming_enhancer_preserves_shape_and_limits_output(self) -> None:
        samples = []
        for index in range(480):
            value = math.sin(index / 12.0) * 0.25
            samples.extend([value, value * 0.9])

        enhancer = StreamingEnhancer(
            EnhancementSettings(service_name="spotify"),
            profile=ListeningPreference(
                service=MusicService.SPOTIFY,
                bass_preference=0.2,
                vocal_clarity_preference=0.5,
            ),
        )
        processed, report = enhancer.process_interleaved(samples)

        self.assertEqual(len(processed), len(samples))
        self.assertLessEqual(max(abs(sample) for sample in processed), 0.98)
        self.assertEqual(report.service_profile, "spotify")
        self.assertIn(
            report.npu_backend,
            {"deterministic-cpu", "onnxruntime-cpu", "onnxruntime-qnn"},
        )

    def test_backend_selector_uses_cpu_fallback_when_npu_not_preferred(self) -> None:
        selector = SnapdragonNpuBackendSelector(prefer_npu=False)

        self.assertIn(selector.select_backend_name(), {"deterministic-cpu", "onnxruntime-cpu"})
        self.assertGreater(
            selector.infer(AudioFrame([[0.1, 0.1], [0.1, 0.1]]), "youtube_music").neural_gain,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
