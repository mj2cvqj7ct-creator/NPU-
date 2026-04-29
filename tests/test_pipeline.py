import math
import unittest

from snapdragon_npu_audio import AudioEnhancementPipeline, AudioFrame, ServiceProfile
from snapdragon_npu_audio.dsp import (
    apply_true_peak_limiter,
    estimate_loudness_lufs,
    normalize_loudness,
)
from snapdragon_npu_audio.inference import BackendKind, select_backend


class PipelineTests(unittest.TestCase):
    def test_limiter_keeps_samples_below_true_peak(self) -> None:
        frame = AudioFrame(
            sample_rate_hz=48_000,
            channels=2,
            samples=((1.5, -1.2), (0.5, -0.5)),
        )

        limited = apply_true_peak_limiter(frame, ceiling_dbfs=-1.0)

        self.assertLessEqual(limited.peak, 10 ** (-1.0 / 20.0) + 1e-12)

    def test_loudness_normalization_raises_quiet_input(self) -> None:
        frame = AudioFrame(
            sample_rate_hz=48_000,
            channels=2,
            samples=tuple((0.02 * math.sin(i / 5), 0.02 * math.sin(i / 5)) for i in range(240)),
        )

        normalized = normalize_loudness(frame, target_lufs=-16.0, max_gain_db=12.0)

        self.assertGreater(estimate_loudness_lufs(normalized), estimate_loudness_lufs(frame))
        self.assertLessEqual(normalized.peak, 1.0)

    def test_pipeline_changes_music_service_frame_safely(self) -> None:
        pipeline = AudioEnhancementPipeline.for_service(ServiceProfile.SPOTIFY)
        frame = AudioFrame(
            sample_rate_hz=48_000,
            channels=2,
            samples=((0.1, -0.1), (0.12, -0.11), (0.08, -0.09)) * 64,
        )

        result = pipeline.process(frame)

        self.assertEqual(result.frame.sample_rate_hz, 48_000)
        self.assertEqual(result.frame.channels, 2)
        self.assertEqual(len(result.frame.samples), len(frame.samples))
        self.assertLessEqual(result.frame.peak, 10 ** (-1.0 / 20.0) + 1e-12)
        self.assertIn(result.backend.kind, {BackendKind.QNN_NPU, BackendKind.DIRECTML, BackendKind.CPU})
        self.assertEqual(result.decision.backend_name, result.backend.kind.value)

    def test_backend_selection_uses_cpu_when_npu_disabled(self) -> None:
        backend = select_backend(prefer_npu=False, force_cpu=True)

        self.assertEqual(backend.status.kind, BackendKind.CPU)
        self.assertFalse(backend.status.accelerated)


if __name__ == "__main__":
    unittest.main()
