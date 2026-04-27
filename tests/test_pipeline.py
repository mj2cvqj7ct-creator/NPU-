from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from snapdragon_audio_enhancer import AudioBuffer, EnhancementPipeline, get_service_profile
from snapdragon_audio_enhancer.inference import CpuHeuristicBackend, QnnOnnxBackend, select_backend
from snapdragon_audio_enhancer.wav_io import read_wav, write_wav


def sine_buffer(sample_rate: int = 48_000, frames: int = 960, amplitude: float = 0.2) -> AudioBuffer:
    generated = []
    for index in range(frames):
        sample = math.sin(2.0 * math.pi * 440.0 * index / sample_rate) * amplitude
        generated.append((sample, sample * 0.9))
    return AudioBuffer(sample_rate=sample_rate, frames=tuple(generated))


class EnhancementPipelineTests(unittest.TestCase):
    def test_pipeline_preserves_frame_count_and_limits_peak(self) -> None:
        source = sine_buffer(amplitude=0.85)
        pipeline = EnhancementPipeline(profile=get_service_profile("spotify"))

        enhanced, telemetry = pipeline.process_with_telemetry(source)

        self.assertEqual(enhanced.frame_count, source.frame_count)
        self.assertLessEqual(enhanced.peak, 0.980001)
        self.assertEqual(telemetry.service, "spotify")
        self.assertEqual(telemetry.backend, "cpu-heuristic")
        self.assertFalse(telemetry.used_npu)

    def test_service_profiles_produce_distinct_tuning(self) -> None:
        source = sine_buffer()
        spotify = EnhancementPipeline(profile=get_service_profile("spotify")).process(source)
        apple = EnhancementPipeline(profile=get_service_profile("apple_music")).process(source)

        self.assertNotEqual(spotify.frames, apple.frames)

    def test_wav_roundtrip_uses_stereo_pcm(self) -> None:
        source = sine_buffer(frames=64, amplitude=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.wav"
            write_wav(path, source)
            loaded = read_wav(path)

        self.assertEqual(loaded.sample_rate, source.sample_rate)
        self.assertEqual(loaded.channels, 2)
        self.assertEqual(loaded.frame_count, source.frame_count)
        self.assertAlmostEqual(loaded.frames[1][0], source.frames[1][0], places=4)

class InferenceSelectionTests(unittest.TestCase):
    def test_select_backend_falls_back_to_cpu_without_model(self) -> None:
        self.assertIsInstance(select_backend(model_path=None, prefer_npu=True), CpuHeuristicBackend)

    def test_qnn_backend_reports_unavailable_without_onnxruntime(self) -> None:
        backend = QnnOnnxBackend("model.onnx")
        if not backend.available:
            self.assertIsInstance(select_backend("model.onnx", prefer_npu=True), CpuHeuristicBackend)


if __name__ == "__main__":
    unittest.main()
