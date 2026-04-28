import json
import tempfile
import unittest
from pathlib import Path

import audio_lossless_assistant as ala


class AudioLosslessAssistantTest(unittest.TestCase):
    def test_lossy_codec_cannot_be_restored_to_true_lossless(self):
        assessment = ala.assess_codec("MP3")

        self.assertTrue(assessment.known_codec)
        self.assertFalse(assessment.is_lossless_codec)
        self.assertFalse(assessment.can_restore_discarded_audio)
        self.assertIn("cannot restore discarded audio", assessment.message)

    def test_ldac_is_treated_as_lossy_bluetooth_codec(self):
        plan = ala.build_preservation_plan("LDAC", "FLAC")

        self.assertEqual(plan.source_codec, "ldac")
        self.assertEqual(plan.target_codec, "flac")
        self.assertFalse(plan.truly_lossless_result)
        self.assertIsNotNone(plan.warning)
        self.assertIn("preserved-from-lossy", plan.steps[-1])

    def test_lossless_source_to_lossless_target_is_true_preservation(self):
        plan = ala.build_preservation_plan("Apple Lossless", "wav")

        self.assertEqual(plan.source_codec, "alac")
        self.assertEqual(plan.target_codec, "wav")
        self.assertTrue(plan.truly_lossless_result)
        self.assertIsNone(plan.warning)

    def test_rejects_lossy_target_codec(self):
        with self.assertRaises(ValueError):
            ala.build_preservation_plan("flac", "aac")

    def test_plan_can_be_written_as_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "plan.json"
            args = type(
                "Args",
                (),
                {"source_codec": "opus", "target_codec": "flac", "output": output},
            )()

            self.assertEqual(ala.plan_cmd(args), 0)

            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["source_codec"], "opus")
            self.assertFalse(data["truly_lossless_result"])


if __name__ == "__main__":
    unittest.main()
