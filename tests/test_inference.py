import unittest

from snapdragon_npu_audio.inference import ProviderKind, ProviderRequest, select_provider


class InferenceProviderTests(unittest.TestCase):
    def test_selects_qnn_when_preferred_and_available(self) -> None:
        provider = select_provider(
            ProviderRequest(available=("CPUExecutionProvider", "QNNExecutionProvider"))
        )

        self.assertEqual(provider.kind, ProviderKind.QNN_NPU)
        self.assertTrue(provider.accelerated)

    def test_uses_directml_when_qnn_unavailable(self) -> None:
        provider = select_provider(ProviderRequest(available=("DmlExecutionProvider",)))

        self.assertEqual(provider.kind, ProviderKind.DIRECTML)
        self.assertTrue(provider.accelerated)

    def test_uses_cpu_when_accelerators_unavailable(self) -> None:
        provider = select_provider(ProviderRequest(available=("CPUExecutionProvider",)))

        self.assertEqual(provider.kind, ProviderKind.CPU)
        self.assertFalse(provider.accelerated)


if __name__ == "__main__":
    unittest.main()
