import unittest
from unittest.mock import patch

from npu_audio_enhancer.inference import BackendChoice, HeuristicFeatureModel, InferenceBackendSelector
from npu_audio_enhancer.dsp import FrameFeatures


class InferenceBackendSelectorTests(unittest.TestCase):
    def test_prefers_qnn_on_snapdragon_arm64(self) -> None:
        selector = InferenceBackendSelector(["CPUExecutionProvider", "QNNExecutionProvider"])
        with patch.object(selector, "_looks_like_snapdragon_arm64", return_value=True):
            status = selector.select()

        self.assertEqual(status.choice, BackendChoice.QNN_NPU)
        self.assertEqual(status.provider, "QNNExecutionProvider")

    def test_uses_directml_when_qnn_is_not_available(self) -> None:
        selector = InferenceBackendSelector(["DmlExecutionProvider", "CPUExecutionProvider"])

        status = selector.select()

        self.assertEqual(status.choice, BackendChoice.DIRECTML)

    def test_heuristic_model_bounds_controls(self) -> None:
        controls = HeuristicFeatureModel().infer(
            FrameFeatures(
                loudness_db=-18.0,
                peak=0.7,
                crest_factor_db=6.0,
                stereo_correlation=0.99,
                low_band_energy=0.1,
                mid_band_energy=0.1,
                high_band_energy=0.8,
            )
        )

        self.assertLessEqual(controls["bass_gain_db"], 2.0)
        self.assertGreaterEqual(controls["presence_gain_db"], -1.5)
        self.assertLessEqual(controls["stereo_width"], 1.08)


if __name__ == "__main__":
    unittest.main()
