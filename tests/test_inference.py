import unittest

from snapdragon_audio_enhancer.inference import (
    BackendKind,
    InferenceConfig,
    InferenceEngine,
    PresenceEnhancementModel,
    RuntimeCapabilities,
    choose_backend,
)


class BackendSelectionTests(unittest.TestCase):
    def test_prefers_qnn_npu_on_snapdragon_x_arm64(self) -> None:
        capabilities = RuntimeCapabilities(
            is_arm64=True,
            has_snapdragon_x_npu=True,
            has_qnn_execution_provider=True,
            has_directml=True,
        )

        self.assertEqual(choose_backend(capabilities), BackendKind.QNN_NPU)

    def test_uses_directml_when_qnn_npu_is_unavailable(self) -> None:
        capabilities = RuntimeCapabilities(
            is_arm64=True,
            has_snapdragon_x_npu=True,
            has_qnn_execution_provider=False,
            has_directml=True,
        )

        self.assertEqual(choose_backend(capabilities), BackendKind.DIRECTML)

    def test_falls_back_to_cpu_without_accelerators(self) -> None:
        capabilities = RuntimeCapabilities(
            is_arm64=False,
            has_snapdragon_x_npu=False,
            has_qnn_execution_provider=False,
            has_directml=False,
        )

        self.assertEqual(choose_backend(capabilities), BackendKind.CPU)


class InferenceModelTests(unittest.TestCase):
    def test_rejects_frames_that_exceed_latency_budget(self) -> None:
        engine = InferenceEngine(
            capabilities=RuntimeCapabilities(
                is_arm64=True,
                has_snapdragon_x_npu=True,
                has_qnn_execution_provider=True,
                has_directml=False,
            ),
            model=PresenceEnhancementModel(),
            config=InferenceConfig(frame_size=2),
        )

        with self.assertRaisesRegex(ValueError, "low-latency frame size"):
            engine.enhance_control_frame([0.1, 0.2, 0.3], {})

    def test_presence_model_keeps_samples_bounded(self) -> None:
        model = PresenceEnhancementModel(InferenceConfig(enhancement_mix=1.0))

        output = model.infer([0.9, -0.9, 0.9], {"spectral_density": 1.0})

        self.assertTrue(all(-1.0 <= sample <= 1.0 for sample in output))


if __name__ == "__main__":
    unittest.main()
