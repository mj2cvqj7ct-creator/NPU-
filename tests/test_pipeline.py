from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from npu_audio_enhancer import EnhancementPipeline
from npu_audio_enhancer.audio_frame import AudioFrame
from npu_audio_enhancer.wav_io import iter_wav_frames, read_wav, write_wav


def sine_frame(amplitude: float = 0.2, frames: int = 960) -> AudioFrame:
    samples: list[list[float]] = []
    for index in range(frames):
        value = amplitude * math.sin(2.0 * math.pi * 440.0 * index / 48_000)
        samples.append([value, value])
    return AudioFrame(samples=samples, sample_rate=48_000)


class EnhancementPipelineTests(unittest.TestCase):
    def test_pipeline_preserves_shape_and_limits_peak(self) -> None:
        pipeline = EnhancementPipeline()
        processed = pipeline.process(sine_frame(amplitude=0.95))

        self.assertEqual(processed.sample_rate, 48_000)
        self.assertEqual(processed.frame_count, 960)
        self.assertEqual(processed.channels, 2)
        self.assertLessEqual(processed.peak(), 10 ** (-1.2 / 20.0) + 1.0e-9)
        self.assertIsNotNone(pipeline.last_report)
        self.assertEqual(pipeline.last_report.provider, "cpu")

    def test_pipeline_boosts_quiet_content_without_clipping(self) -> None:
        pipeline = EnhancementPipeline()
        original = sine_frame(amplitude=0.02)
        processed = pipeline.process(original)

        self.assertGreater(processed.rms(), original.rms())
        self.assertLessEqual(processed.peak(), 1.0)

    def test_wav_round_trip_and_frame_split(self) -> None:
        source = sine_frame(frames=1_000)
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            wav_path = tmp_path / "source.wav"
            out_path = tmp_path / "out.wav"

            write_wav(wav_path, source)
            frames = iter_wav_frames(wav_path, frame_ms=10.0)
            write_wav(out_path, frames)
            restored = read_wav(out_path)

        self.assertEqual(len(frames), 3)
        self.assertEqual(restored.sample_rate, source.sample_rate)
        self.assertEqual(restored.frame_count, source.frame_count)
        self.assertEqual(restored.channels, 2)


if __name__ == "__main__":
    unittest.main()
