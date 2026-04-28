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


if __name__ == "__main__":
    unittest.main()
