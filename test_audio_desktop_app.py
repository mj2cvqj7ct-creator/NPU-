import unittest

import audio_desktop_app as app


class AudioDesktopAppTest(unittest.TestCase):
    def test_lossless_assessment_text_reports_lossy_mp3(self):
        text = app.build_lossless_assessment_text("mp3")

        self.assertIn("Codec: mp3", text)
        self.assertIn("Lossless codec: false", text)
        self.assertIn("Can restore discarded audio: false", text)

    def test_lossless_plan_text_reports_true_flac_to_alac_preservation(self):
        text = app.build_lossless_plan_text("flac", "alac")

        self.assertIn("Source codec: flac", text)
        self.assertIn("Target codec: alac", text)
        self.assertIn("Truly lossless result: true", text)

    def test_empty_source_codec_is_rejected(self):
        with self.assertRaises(ValueError):
            app.build_lossless_assessment_text("   ")

    def test_windows_ldac_status_text_mentions_no_native_encoder(self):
        text = app.build_ldac_status_text("Windows")

        self.assertIn("Native LDAC available: false", text)
        self.assertIn("LDAC", text)

    def test_npu_enhancement_text_marks_output_as_estimated(self):
        status = app.npu_enhancement.NpuStatus(
            available=True,
            provider="QNNExecutionProvider",
            detail="test provider",
        )
        text = app.build_npu_enhancement_text("ldac", "flac-24-96", status=status)

        self.assertIn("Acceleration: NPU via QNNExecutionProvider", text)
        self.assertIn("True lossless restoration: false", text)
        self.assertIn("ai-enhanced-high-res-preservation", text)

    def test_snapdragon_studio_plan_text_mentions_service_and_xmos(self):
        text = app.build_streaming_studio_plan_text(
            "spotify",
            user_id="listener-a",
            provider="QNNExecutionProvider",
        )

        self.assertIn("Service: spotify", text)
        self.assertIn("XMOS low-latency plan:", text)
        self.assertIn("provider=QNNExecutionProvider", text)
        self.assertIn("Windows EXE build:", text)

    def test_snapdragon_recommendation_update_text_returns_bias_json(self):
        text = app.build_streaming_recommendation_update_text(
            "listener-a",
            clarity=0.9,
            depth=0.8,
            vocal=0.95,
            bass=0.7,
        )

        self.assertIn("Realtime recommendation updated for user: listener-a", text)
        self.assertIn("Updates: 1", text)
        self.assertIn("Bias: acoustic=", text)


if __name__ == "__main__":
    unittest.main()
