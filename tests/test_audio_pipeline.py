import math
from pathlib import Path
import tempfile
import unittest
import wave

from snapdragon_npu_audio_enhancer.audio import AudioBuffer, read_wav, write_wav
from snapdragon_npu_audio_enhancer.dsp import extract_features, true_peak_limiter
from snapdragon_npu_audio_enhancer.pipeline import EnhancementConfig, enhance_samples


class AudioPipelineTests(unittest.TestCase):
    def test_pipeline_normalizes_quiet_audio_without_clipping(self) -> None:
        samples = [(0.04 * math.sin(i / 9.0), 0.035 * math.sin(i / 9.0)) for i in range(4800)]
        audio = AudioBuffer(sample_rate=48_000, samples=samples)

        result = enhance_samples(audio, EnhancementConfig(service="spotify"))

        self.assertEqual(result.service, "spotify")
        self.assertLessEqual(result.audio.peak, 10 ** (-1.0 / 20.0) + 1.0e-9)
        self.assertGreater(result.audio.rms, audio.rms)


    def test_true_peak_limiter_enforces_ceiling(self) -> None:
        audio = AudioBuffer(sample_rate=48_000, samples=[(1.5, -1.2), (0.2, -0.3)])

        limited = true_peak_limiter(audio, ceiling_dbfs=-6.0)

        self.assertLessEqual(limited.peak, 10 ** (-6.0 / 20.0) + 1.0e-9)


    def test_feature_extraction_detects_clipping(self) -> None:
        audio = AudioBuffer(sample_rate=48_000, samples=[(1.0, -1.0), (0.0, 0.0)])

        features = extract_features(audio)

        self.assertEqual(features.clipping_ratio, 0.5)
        self.assertEqual(features.peak_dbfs, 0.0)


    def test_wav_round_trip_writes_stereo_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "out.wav"
            source = AudioBuffer(sample_rate=44_100, samples=[(-0.5, 0.5), (0.25, -0.25)])

            write_wav(str(path), source)
            restored = read_wav(str(path))

            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 44_100)

            self.assertEqual(restored.sample_rate, source.sample_rate)
            self.assertEqual(len(restored.samples), len(source.samples))
            self.assertAlmostEqual(restored.samples[0][0], -0.5, delta=4.0e-5)
            self.assertAlmostEqual(restored.samples[0][1], 0.5, delta=4.0e-5)


if __name__ == "__main__":
    unittest.main()
