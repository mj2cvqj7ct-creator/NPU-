import os
import tempfile
import unittest
import wave

from npu_audio_enhancer.audio import AudioBuffer, enhance_audio, generate_demo_buffer, read_wav, write_wav
from npu_audio_enhancer.profiles import get_profile


class AudioPipelineTest(unittest.TestCase):
    def test_generate_demo_buffer_is_stereo_48khz(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.1)

        self.assertEqual(audio.sample_rate, 48_000)
        self.assertEqual(audio.channels, 2)
        self.assertEqual(len(audio.samples), 9_600)

    def test_enhance_audio_limits_peak(self) -> None:
        audio = AudioBuffer(sample_rate=48_000, channels=1, samples=[-0.9, 0.9, 0.2])

        enhanced = enhance_audio(audio, get_profile("snapdragon-x-npu"))
        peak = max(abs(sample) for sample in enhanced.samples)

        self.assertLessEqual(peak, 0.921)
        self.assertEqual(enhanced.sample_rate, audio.sample_rate)
        self.assertEqual(enhanced.channels, audio.channels)

    def test_wav_round_trip(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.05)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo.wav")
            write_wav(path, audio)
            decoded = read_wav(path)

            with wave.open(path, "rb") as wav:
                self.assertEqual(wav.getframerate(), 48_000)
                self.assertEqual(wav.getnchannels(), 2)

        self.assertEqual(decoded.sample_rate, audio.sample_rate)
        self.assertEqual(decoded.channels, audio.channels)
        self.assertEqual(len(decoded.samples), len(audio.samples))


if __name__ == "__main__":
    unittest.main()
