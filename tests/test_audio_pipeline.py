import os
import tempfile
import unittest
import wave

from npu_audio_enhancer.audio import (
    AudioBuffer,
    enhance_wav,
    generate_demo_buffer,
    read_wav,
    write_wav,
)
from npu_audio_enhancer.npu import build_backend_plan, describe_backend_plan
from npu_audio_enhancer.pipeline import enhance_audio
from npu_audio_enhancer.profiles import get_profile, get_service_profile
from npu_audio_enhancer.recommendation import ListeningSignal, build_local_sound_preference


class AudioPipelineTest(unittest.TestCase):
    def test_generate_demo_buffer_is_stereo_48khz(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.1)

        self.assertEqual(audio.sample_rate, 48_000)
        self.assertEqual(audio.channels, 2)
        self.assertEqual(audio.frame_count, 4_800)
        self.assertEqual(len(audio.samples), 9_600)

    def test_enhance_audio_limits_peak_and_changes_signal(self) -> None:
        audio = AudioBuffer(sample_rate=48_000, channels=1, samples=(-0.9, 0.9, 0.2, -0.1))

        result = enhance_audio(
            audio,
            get_profile("snapdragon-x-npu"),
            services=(get_service_profile("spotify"),),
        )

        self.assertLessEqual(result.metrics.output_peak, 0.91)
        self.assertEqual(result.audio.sample_rate, audio.sample_rate)
        self.assertEqual(result.audio.channels, audio.channels)
        self.assertNotEqual(result.audio.samples, audio.samples)

    def test_holographic_profile_preserves_stereo_shape(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.05)

        result = enhance_audio(
            audio,
            get_profile("holographic-vocal-stage"),
            services=(get_service_profile("apple-music"),),
        )

        self.assertEqual(result.audio.channels, 2)
        self.assertEqual(len(result.audio.samples), len(audio.samples))
        self.assertLessEqual(result.metrics.output_peak, 0.91)
        stereo_delta = sum(
            abs(result.audio.samples[index] - result.audio.samples[index + 1])
            for index in range(0, min(200, len(result.audio.samples)), 2)
        )
        self.assertGreater(stereo_delta, 0.0)

    def test_wav_round_trip_and_report(self) -> None:
        audio = generate_demo_buffer(duration_seconds=0.05)

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "demo.wav")
            output = os.path.join(tmp, "enhanced.wav")
            write_wav(source, audio)
            decoded = read_wav(source)
            report = enhance_wav(
                source,
                output,
                profile_name="snapdragon-x-npu",
                service_name="youtube-music",
            )

            with wave.open(output, "rb") as wav:
                self.assertEqual(wav.getframerate(), 48_000)
                self.assertEqual(wav.getnchannels(), 2)

        self.assertEqual(decoded.sample_rate, audio.sample_rate)
        self.assertEqual(decoded.channels, audio.channels)
        self.assertEqual(report.profile_name, "snapdragon-x-npu+youtube-music")
        self.assertEqual(report.service.slug, "youtube-music")
        self.assertGreater(report.frames, 0)
        self.assertLessEqual(report.latency_ms, 20.0)

    def test_backend_plan_prefers_qnn_on_arm64(self) -> None:
        plan = build_backend_plan(prefer_npu=True, frame_ms=10, machine="arm64")
        text = describe_backend_plan(plan)

        self.assertEqual(plan.backend, "snapdragon-x-npu")
        self.assertIn("ONNX Runtime QNN Execution Provider", text)
        self.assertIn("Frame size: 10 ms", text)

    def test_backend_plan_uses_cpu_reference_on_non_arm64(self) -> None:
        plan = build_backend_plan(prefer_npu=True, machine="x86_64")

        self.assertEqual(plan.backend, "cpu-reference")
        self.assertEqual(plan.status, "portable-validation-mode")

    def test_local_preference_uses_only_local_events(self) -> None:
        preference = build_local_sound_preference(
            [
                ListeningSignal("spotify", "holographic-vocal-stage", volume=0.45),
                ListeningSignal("apple-music", "holographic-vocal-stage", volume=0.50),
                ListeningSignal("youtube-music", "balanced", volume=0.35, skipped=True),
            ]
        )

        self.assertGreater(preference.vocal_presence, 0.55)
        self.assertLess(preference.loudness_bias, 0.6)
        self.assertGreater(preference.transient_detail, 0.5)


if __name__ == "__main__":
    unittest.main()
