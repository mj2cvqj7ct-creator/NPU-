import os
import tempfile
import unittest
import wave
from unittest import mock

from npu_audio_enhancer.audio import AudioBuffer, enhance_audio, generate_demo_buffer, read_wav, write_wav
from npu_audio_enhancer.profiles import get_profile
from npu_audio_enhancer.recommender import RecommendationEngine, build_demo_catalog, build_recommendation_status
from npu_audio_enhancer.reports import build_status_text
from npu_audio_enhancer.realtime import ServiceState, build_realtime_status


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

    def test_holographic_profile_preserves_stereo_shape(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.05)

        enhanced = enhance_audio(audio, get_profile("holographic-vocal-stage"))

        self.assertEqual(enhanced.channels, 2)
        self.assertEqual(len(enhanced.samples), len(audio.samples))
        self.assertLessEqual(max(abs(sample) for sample in enhanced.samples), 0.901)
        self.assertNotEqual(enhanced.samples[0], enhanced.samples[1])

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

    def test_status_text_includes_profile_and_peaks(self) -> None:
        report = mock.Mock()
        report.profile.name = "snapdragon-x-npu"
        report.profile.target_backend = "onnxruntime-qnn"
        report.samples = 96_000
        report.input_peak = 0.878
        report.output_peak = 0.763

        status = build_status_text(report)

        self.assertIn("Profile: snapdragon-x-npu", status)
        self.assertIn("Target backend: onnxruntime-qnn", status)
        self.assertIn("Samples: 96000", status)
        self.assertIn("Output peak: 0.7630", status)

    def test_realtime_status_mentions_npu_streaming_requirements(self) -> None:
        status = build_realtime_status(
            ServiceState(
                spotify=True,
                apple_music=True,
                youtube_music=True,
                profile="snapdragon-x-npu",
            ),
            active=True,
        )

        self.assertIn("Spotify, Apple Music, YouTube Music", status)
        self.assertIn("snapdragon-x-npu", status)
        self.assertIn("Windows ARM64 + Snapdragon X NPU", status)
        self.assertIn("ONNX Runtime QNN Execution Provider", status)
        self.assertIn("ASIO exclusive output", status)
        self.assertIn("holographic imaging", status)
        self.assertIn("XMOS USB DAC Driver Control Panel", status)
        self.assertIn("target 32 samples", status)

    def test_recommendation_engine_ranks_unheard_tracks(self) -> None:
        engine = RecommendationEngine(build_demo_catalog())
        result = engine.recommend(
            recent_track_ids=("spotify:aurora-drive", "apple:glass-voice"),
            service_targets=("Spotify", "Apple Music", "YouTube Music"),
            limit=3,
        )

        recommended_ids = [item.track.track_id for item in result.tracks]
        status = build_recommendation_status(result)

        self.assertEqual(len(result.tracks), 3)
        self.assertNotIn("spotify:aurora-drive", recommended_ids)
        self.assertIn("NPU target: Snapdragon X NPU", status)
        self.assertIn("Realtime reflection: service queues, smart playlists, API sync payloads", status)
        self.assertIn("Realtime update tick: #1", status)
        self.assertIn("Top realtime picks:", status)
        self.assertIn("Spotify / Apple Music / YouTube Music", status)


if __name__ == "__main__":
    unittest.main()
