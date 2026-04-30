from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snapdragon_npu_audio.audio_frame import AudioFrame
from snapdragon_npu_audio.inference import BackendKind, InferenceConfig, available_backend_kinds
from snapdragon_npu_audio.pipeline import EnhancementConfig, EnhancementPipeline, StreamingEnhancer
from snapdragon_npu_audio.service_profiles import MusicService, get_service_profile


def _sine_frame(sample_rate: int = 48_000, frames: int = 960, amplitude: float = 0.4) -> AudioFrame:
    samples = []
    for index in range(frames):
        value = math.sin(2.0 * math.pi * 440.0 * index / sample_rate) * amplitude
        samples.append((value, value * 0.92))
    return AudioFrame(sample_rate=sample_rate, samples=tuple(samples))


class EnhancementPipelineTest(unittest.TestCase):
    def test_pipeline_limits_peak_and_reports_cpu_fallback(self) -> None:
        pipeline = EnhancementPipeline(
            EnhancementConfig(
                service=MusicService.SPOTIFY,
                intensity=1.0,
                inference=InferenceConfig(preferred_backends=(BackendKind.CPU,)),
            )
        )

        result = pipeline.process(_sine_frame(amplitude=0.95))

        self.assertEqual(result.backend_kind, BackendKind.CPU.value)
        self.assertLessEqual(result.frame.peak, 10 ** (-1.0 / 20.0) + 1e-9)
        self.assertEqual(result.service, MusicService.SPOTIFY)
        self.assertGreaterEqual(result.controls.vocal_clarity, 0.0)
        self.assertLessEqual(result.controls.vocal_clarity, 1.0)

    def test_streaming_enhancer_emits_fixed_frames_and_flushes_remainder(self) -> None:
        pipeline = EnhancementPipeline(
            EnhancementConfig(inference=InferenceConfig(preferred_backends=(BackendKind.CPU,)))
        )
        enhancer = StreamingEnhancer(pipeline, frame_size=128)
        input_frame = _sine_frame(frames=300, amplitude=0.2)

        results = enhancer.push(input_frame)
        flushed = enhancer.flush(input_frame.sample_rate)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(result.frame.samples) == 128 for result in results))
        self.assertIsNotNone(flushed)
        assert flushed is not None
        self.assertEqual(len(flushed.frame.samples), 44)

    def test_service_profile_falls_back_to_generic_for_unknown_services(self) -> None:
        self.assertEqual(get_service_profile("unknown").service, MusicService.GENERIC)
        self.assertGreater(
            get_service_profile("youtube_music").presence_db,
            get_service_profile("apple_music").presence_db,
        )

    def test_available_backend_kinds_always_include_cpu(self) -> None:
        self.assertIn(BackendKind.CPU, available_backend_kinds())


if __name__ == "__main__":
    unittest.main()
