import unittest

import snapdragon_streaming_studio as studio


class SnapdragonStreamingStudioTest(unittest.TestCase):
    def test_build_plan_for_spotify_contains_npu_and_dac(self):
        status = studio.npu.NpuStatus(True, "QNNExecutionProvider", "test provider")
        original_detect = studio.npu.detect_npu_status
        studio.npu.detect_npu_status = lambda preferred_provider=None: status
        try:
            plan = studio.build_realtime_audio_plan(
                service="spotify",
                profile_name="immersive-reference",
                target_latency_ms=28,
                sample_rate_hz=48000,
            )
        finally:
            studio.npu.detect_npu_status = original_detect

        self.assertEqual(plan.service, "spotify")
        self.assertEqual(plan.npu_provider, "QNNExecutionProvider")
        self.assertTrue(plan.npu_available)
        self.assertEqual(plan.dac_plan.output_device, studio.SABAJ_A20D_ES)
        self.assertGreaterEqual(plan.dac_plan.asio_buffer_samples, 128)

    def test_unknown_service_is_rejected(self):
        with self.assertRaises(ValueError):
            studio.build_realtime_audio_plan("tidal")

    def test_recommendation_state_learns_and_scores(self):
        state = studio.RealtimeRecommendationState()
        state.learn(["vocal", "clarity"], 0.8)
        state.learn(["bass"], -0.5)

        score_positive = state.score(["vocal", "clarity"])
        score_negative = state.score(["bass"])

        self.assertGreater(score_positive, 0.0)
        self.assertLess(score_negative, 0.0)
        self.assertGreaterEqual(state.updates, 2)

    def test_build_exe_commands_contains_desktop_copy_step(self):
        commands = studio.build_windows_exe_commands()
        self.assertEqual(len(commands), 3)
        self.assertIn("pyinstaller", commands[1].lower())
        self.assertIn("Desktop", commands[2])


if __name__ == "__main__":
    unittest.main()
