import unittest

import audio_desktop_app as app


class AudioDesktopAppTest(unittest.TestCase):
    def test_lossless_assessment_text_reports_lossy_mp3(self):
        text = app.build_lossless_assessment_text("mp3")

        self.assertIn("コーデック: mp3", text)
        self.assertIn("ロスレスコーデック: いいえ", text)
        self.assertIn("破棄された音声の復元: 不可", text)

    def test_lossless_plan_text_reports_true_flac_to_alac_preservation(self):
        text = app.build_lossless_plan_text("flac", "alac")

        self.assertIn("入力コーデック: flac", text)
        self.assertIn("保存先コーデック: alac", text)
        self.assertIn("真のロスレス結果: はい", text)

    def test_empty_source_codec_is_rejected(self):
        with self.assertRaises(ValueError):
            app.build_lossless_assessment_text("   ")

    def test_windows_ldac_status_text_mentions_no_native_encoder(self):
        text = app.build_ldac_status_text("Windows")

        self.assertIn("ネイティブLDAC対応: なし", text)
        self.assertIn("LDAC", text)
        self.assertIn("WASAPI排他モード", text)

    def test_npu_enhancement_text_marks_output_as_estimated(self):
        status = app.npu_enhancement.NpuStatus(
            available=True,
            provider="QNNExecutionProvider",
            detail="test provider",
        )
        text = app.build_npu_enhancement_text("ldac", "flac-24-96", status=status)

        self.assertIn("アクセラレーション: NPU via QNNExecutionProvider", text)
        self.assertIn("真のロスレス復元: いいえ", text)
        self.assertIn("ai-enhanced-high-res-preservation", text)


if __name__ == "__main__":
    unittest.main()
