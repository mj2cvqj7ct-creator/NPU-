import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from npu_audio_enhancer import AudioEnhancementPipeline, AudioFrame, MusicService
from npu_audio_enhancer.dsp import analyze_frame
from npu_audio_enhancer.inference import BackendKind, EnhancementBackend, InferenceResult


class StubBackend(EnhancementBackend):
    kind = BackendKind.SNAPDRAGON_QNN
    reason = "test backend"

    def infer(self, frame, features=None, profile=None):
        return InferenceResult(
            clarity_boost=0.8,
            warmth_boost=0.6,
            transient_restore=0.5,
            stereo_expansion=0.4,
            gain_trim_db=-1.5,
            backend=self.kind,
        )


def sine_frame(amplitude=0.2, frequency=1000.0, frames=480):
    samples = []
    for index in range(frames):
        value = amplitude * math.sin(2.0 * math.pi * frequency * index / 48_000)
        samples.append((value, value))
    return AudioFrame(samples=samples)


class EnhancementPipelineTests(unittest.TestCase):
    def test_pipeline_preserves_frame_shape_and_limits_true_peak(self):
        frame = sine_frame(amplitude=1.2)
        pipeline = AudioEnhancementPipeline(service=MusicService.SPOTIFY, backend=StubBackend())

        result = pipeline.process(frame)

        self.assertEqual(len(result.frame.samples), len(frame.samples))
        self.assertEqual(result.backend, BackendKind.SNAPDRAGON_QNN.value)
        self.assertLessEqual(analyze_frame(result.frame).peak_dbfs, -0.99)
        for left, right in result.frame.samples:
            self.assertLessEqual(abs(left), 0.892)
            self.assertLessEqual(abs(right), 0.892)

    def test_service_profiles_produce_distinct_outputs(self):
        frame = sine_frame(amplitude=0.3, frequency=180.0)

        spotify = AudioEnhancementPipeline(
            service=MusicService.SPOTIFY,
            backend=StubBackend(),
        ).process(frame).frame.samples
        youtube = AudioEnhancementPipeline(
            service=MusicService.YOUTUBE_MUSIC,
            backend=StubBackend(),
        ).process(frame).frame.samples

        self.assertNotEqual(spotify, youtube)

    def test_invalid_frame_rejected(self):
        with self.assertRaises(ValueError):
            AudioFrame(samples=[(0.0, 0.0)], sample_rate=44_100)

    def test_interleaved_round_trip(self):
        frame = AudioFrame.from_interleaved((0.1, -0.1, 0.2, -0.2))

        self.assertEqual(frame.samples, ((0.1, -0.1), (0.2, -0.2)))
        self.assertEqual(frame.to_interleaved(), (0.1, -0.1, 0.2, -0.2))


if __name__ == "__main__":
    unittest.main()
