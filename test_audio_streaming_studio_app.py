import unittest

import audio_streaming_studio_app as app


class AudioStreamingStudioAppTest(unittest.TestCase):
    def test_build_plan_text_contains_expected_sections(self):
        text = app.build_plan_text(
            service="spotify",
            profile="immersive-reference",
            target_latency_ms=28,
            sample_rate_hz=48000,
            provider=None,
        )

        self.assertIn("Service: Spotify", text)
        self.assertIn("DAC plan:", text)
        self.assertIn("Pipeline steps:", text)

    def test_build_exe_text_contains_pyinstaller_and_desktop_copy(self):
        text = app.build_exe_text()

        self.assertIn("pyinstaller", text.lower())
        self.assertIn("Desktop", text)

    def test_json_dumps_uses_pretty_format(self):
        rendered = app.json_dumps({"a": 1})
        self.assertIn("\n", rendered)
        self.assertIn('"a"', rendered)


if __name__ == "__main__":
    unittest.main()
