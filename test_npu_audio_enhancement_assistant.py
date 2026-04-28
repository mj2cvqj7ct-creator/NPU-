import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import npu_audio_enhancement_assistant as nae


class NpuAudioEnhancementAssistantTest(unittest.TestCase):
    def test_ldac_plan_is_ai_enhanced_not_true_lossless(self):
        status = nae.NpuStatus(True, "QNNExecutionProvider", "test provider")
        plan = nae.build_enhancement_plan("LDAC", "flac-24-96", npu_status=status)

        self.assertEqual(plan.source_codec, "ldac")
        self.assertEqual(plan.target_container, "flac")
        self.assertEqual(plan.target_bit_depth, 24)
        self.assertEqual(plan.target_sample_rate_hz, 96000)
        self.assertTrue(plan.npu_available)
        self.assertFalse(plan.true_lossless_restoration)
        self.assertEqual(plan.output_label, "ai-enhanced-high-res-preservation")
        self.assertIn("cannot prove or restore", plan.warning)

    def test_cpu_fallback_when_onnxruntime_is_absent(self):
        status = nae.NpuStatus(False, "CPUExecutionProvider", "test fallback")
        plan = nae.build_enhancement_plan("sbc", "wav-24-96", npu_status=status)

        self.assertEqual(plan.acceleration, "CPU fallback")
        self.assertFalse(plan.npu_available)

    def test_forced_npu_provider_environment(self):
        with mock.patch.dict(
            os.environ,
            {"AUDIO_ASSISTANT_NPU_PROVIDER": "OpenVINOExecutionProvider"},
            clear=False,
        ):
            status = nae.detect_npu_status()

        self.assertTrue(status.available)
        self.assertEqual(status.provider, "OpenVINOExecutionProvider")

    def test_rejects_unknown_high_res_target(self):
        with self.assertRaises(ValueError):
            nae.build_enhancement_plan("ldac", "mp3")

    def test_plan_can_be_written_as_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "enhancement.json"
            args = type(
                "Args",
                (),
                {
                    "source_codec": "aptx hd",
                    "target": "flac-24-192",
                    "provider": None,
                    "output": output,
                },
            )()

            self.assertEqual(nae.plan_cmd(args), 0)

            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["source_codec"], "aptx hd")
            self.assertFalse(data["true_lossless_restoration"])


if __name__ == "__main__":
    unittest.main()
