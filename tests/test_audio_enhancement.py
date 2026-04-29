import math
import os
import tempfile
import unittest
import wave

from npu_audio_enhancer.audio import (
    AudioBuffer,
    enhance_audio,
    generate_demo_buffer,
    read_wav,
    write_wav,
)
from npu_audio_enhancer.npu import HeuristicNpuModel, NpuFeatures
from npu_audio_enhancer.profiles import get_profile, service_profile_name


class AudioEnhancementTest(unittest.TestCase):
    def test_demo_buffer_is_48khz_stereo(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.05)

        self.assertEqual(audio.sample_rate, 48_000)
        self.assertEqual(audio.channels, 2)
        self.assertEqual(len(audio.samples), 4_800)

    def test_spotify_profile_limits_peak_and_changes_signal(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.08)

        enhanced = enhance_audio(audio, get_profile("spotify-npu"))
        output_peak = max(abs(sample) for sample in enhanced.samples)

        self.assertEqual(enhanced.sample_rate, audio.sample_rate)
        self.assertEqual(enhanced.channels, audio.channels)
        self.assertEqual(len(enhanced.samples), len(audio.samples))
        self.assertLessEqual(output_peak, 0.911)
        self.assertNotEqual(enhanced.samples[200], audio.samples[200])

    def test_service_profile_aliases_match_major_music_apps(self) -> None:
        self.assertEqual(service_profile_name("spotify"), "spotify-npu")
        self.assertEqual(service_profile_name("apple-music"), "apple-lossless-npu")
        self.assertEqual(service_profile_name("youtube-music"), "youtube-music-npu")

    def test_fixed_npu_features_are_applied_to_dsp(self) -> None:
        audio = AudioBuffer(
            sample_rate=48_000,
            channels=2,
            samples=[0.10, 0.05, 0.20, -0.10, -0.35, -0.20, 0.15, 0.25] * 32,
        )
        features = NpuFeatures(
            clarity=1.0,
            bass_tightness=0.0,
            transient_restore=1.0,
            stereo_focus=1.0,
            noise_floor=0.0,
            vocal_presence=1.0,
        )

        class FixedModel:
            def infer(self, _frame, _profile):
                return features

        enhanced = enhance_audio(
            audio,
            get_profile("snapdragon-x-npu"),
            npu_model=FixedModel(),
        )

        self.assertEqual(len(enhanced.samples), len(audio.samples))
        self.assertLessEqual(max(abs(sample) for sample in enhanced.samples), 0.921)
        self.assertNotEqual(enhanced.samples[0], audio.samples[0])

    def test_heuristic_model_detects_more_noise_on_silent_hash(self) -> None:
        noisy = [0.03 * math.sin(index * 2.2) for index in range(512)]
        clean = [0.60 * math.sin(index * 0.04) for index in range(512)]
        model = HeuristicNpuModel()

        noisy_features = model.infer(noisy, get_profile("youtube-music-npu"))
        clean_features = model.infer(clean, get_profile("youtube-music-npu"))

        self.assertGreater(noisy_features.noise_floor, clean_features.noise_floor)

    def test_wav_round_trip_and_cli_ready_format(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.02)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo.wav")
            write_wav(path, audio)
            decoded = read_wav(path)

            with wave.open(path, "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 48_000)
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getsampwidth(), 2)

        self.assertEqual(decoded.sample_rate, audio.sample_rate)
        self.assertEqual(decoded.channels, audio.channels)
        self.assertEqual(len(decoded.samples), len(audio.samples))


if __name__ == "__main__":
    unittest.main()
