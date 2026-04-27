import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_enhancer.inference import InferenceBackend, ordered_provider_names, select_provider


class InferenceProviderTests(unittest.TestCase):
    def test_selects_qnn_when_snapdragon_npu_is_available(self) -> None:
        provider = select_provider(
            ["CPUExecutionProvider", "DmlExecutionProvider", "QNNExecutionProvider"]
        )

        self.assertIs(provider.backend, InferenceBackend.QNN_NPU)
        self.assertEqual(provider.onnx_provider_name, "QNNExecutionProvider")
        self.assertEqual(ordered_provider_names(provider)[0], "QNNExecutionProvider")

    def test_directml_is_fallback_when_qnn_is_missing(self) -> None:
        provider = select_provider(["CPUExecutionProvider", "DmlExecutionProvider"])

        self.assertIs(provider.backend, InferenceBackend.DIRECTML)
        self.assertEqual(provider.onnx_provider_name, "DmlExecutionProvider")

    def test_cpu_is_final_safe_fallback(self) -> None:
        provider = select_provider(["CPUExecutionProvider"])

        self.assertIs(provider.backend, InferenceBackend.CPU)

    def test_can_disable_npu_for_battery_or_diagnostics(self) -> None:
        provider = select_provider(
            ["CPUExecutionProvider", "DmlExecutionProvider", "QNNExecutionProvider"],
            environment={"AUDIO_ENHANCER_DISABLE_NPU": "1"},
        )

        self.assertIs(provider.backend, InferenceBackend.DIRECTML)


if __name__ == "__main__":
    unittest.main()
