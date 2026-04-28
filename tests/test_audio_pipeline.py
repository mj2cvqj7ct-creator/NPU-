from __future__ import annotations

import os
import tempfile
import unittest

from npu_audio_enhancer import available_profiles, enhance_audio
from npu_audio_enhancer.audio import generate_demo_buffer, read_wav, write_wav
from npu_audio_enhancer.dsp import AudioBuffer, analyze
from npu_audio_enhancer.inference import BackendSelection, create_backend, select_backend
from npu_audio_enhancer.profiles import get_profile


class AudioPipelineTest(unittest.TestCase):
    def test_demo_buffer_matches_internal_stream_format(self) -> None:
        buffer = generate_demo_buffer(duration_seconds=0.1)

        self.assertEqual(buffer.sample_rate, 48_000)
        self.assertEqual(buffer.channels, 2)
        self.assertEqual(buffer.frames, 4_800)

    def test_service_profiles_are_available(self) -> None:
        profiles = available_profiles()

        self.assertIn("spotify", profiles)
        self.assertIn("apple-music", profiles)
        self.assertIn("youtube-music", profiles)
        self.assertIn("snapdragon-x-npu", profiles)

    def test_enhancement_limits_true_peak(self) -> None:
        source = AudioBuffer(48_000, 1, (-1.2, -0.8, 0.3, 1.1))
        profile = get_profile("youtube-music")

        result = enhance_audio(source, profile, backend="cpu")

        self.assertEqual(result.backend_name, "cpu-dsp")
        self.assertLessEqual(result.output_metrics.peak, profile.true_peak_ceiling)
        self.assertEqual(result.audio.sample_rate, source.sample_rate)
        self.assertEqual(result.audio.channels, source.channels)

    def test_snapdragon_profile_uses_qnn_when_available(self) -> None:
        source = generate_demo_buffer(duration_seconds=0.05)
        profile = get_profile("snapdragon-x-npu")

        backend = create_backend("qnn", BackendSelection(qnn_available=True))
        result = enhance_audio(source, profile, backend=backend)

        self.assertEqual(result.backend_name, "onnxruntime-qnn")
        self.assertLessEqual(result.output_metrics.peak, profile.true_peak_ceiling)
        self.assertNotEqual(result.audio.samples, source.samples)

    def test_backend_falls_back_to_cpu_without_qnn(self) -> None:
        backend = select_backend(get_profile("spotify"))

        self.assertEqual(backend.name, "cpu-dsp")
        self.assertIn("fallback", backend.description)

    def test_wav_round_trip_and_enhance(self) -> None:
        source = generate_demo_buffer(duration_seconds=0.05)

        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, "input.wav")
            output_path = os.path.join(tmp, "output.wav")
            write_wav(input_path, source)

            decoded = read_wav(input_path)
            result = enhance_audio(decoded, get_profile("spotify"), backend="cpu")
            write_wav(output_path, result.audio)
            enhanced = read_wav(output_path)

        self.assertEqual(decoded.frames, source.frames)
        self.assertEqual(enhanced.sample_rate, 48_000)
        self.assertLessEqual(analyze(enhanced).peak, get_profile("spotify").true_peak_ceiling + 1e-4)


if __name__ == "__main__":
    unittest.main()
