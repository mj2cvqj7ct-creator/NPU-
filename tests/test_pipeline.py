import unittest

from snapdragon_audio_enhancer import AudioEnhancementPipeline, EnhancementProfile
from snapdragon_audio_enhancer.inference import (
    BackendKind,
    InferenceEngine,
    PresenceEnhancementModel,
    RuntimeCapabilities,
)


class PipelineTests(unittest.TestCase):
    def test_pipeline_keeps_processed_audio_under_true_peak_ceiling(self) -> None:
        pipeline = AudioEnhancementPipeline(
            profile=EnhancementProfile(
                bass_boost_db=4.0,
                presence_boost_db=3.0,
                stereo_width=1.2,
                target_loudness_dbfs=-10.0,
                true_peak_ceiling=0.75,
            )
        )
        hot_frame = tuple((0.95, -0.95) for _ in range(64))

        result = pipeline.process(hot_frame)

        self.assertTrue(result.report.peak_limited)
        self.assertLessEqual(
            max(abs(sample) for frame in result.frames for sample in frame),
            0.75,
        )

    def test_pipeline_sanitizes_non_finite_samples(self) -> None:
        pipeline = AudioEnhancementPipeline()

        result = pipeline.process(((float("nan"), float("inf")), (0.25, -0.25)))

        self.assertEqual(result.report.frame_count, 2)
        self.assertTrue(
            all(
                -1.0 <= sample <= 1.0
                for frame in result.frames
                for sample in frame
            )
        )

    def test_inference_engine_reports_selected_backend(self) -> None:
        engine = InferenceEngine(
            capabilities=RuntimeCapabilities(
                is_arm64=True,
                has_snapdragon_x_npu=True,
                has_qnn_execution_provider=True,
                has_directml=False,
            ),
            model=PresenceEnhancementModel(),
        )
        pipeline = AudioEnhancementPipeline(inference_engine=engine)

        result = pipeline.process(tuple((0.1, 0.05) for _ in range(32)))

        self.assertEqual(result.report.inference.backend, BackendKind.QNN_NPU)
        self.assertGreater(result.report.inference.presence_weight, 0.85)


if __name__ == "__main__":
    unittest.main()
