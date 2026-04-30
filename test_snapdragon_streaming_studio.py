import json
import tempfile
import unittest
from pathlib import Path

import snapdragon_streaming_studio as studio


class SnapdragonStreamingStudioTest(unittest.TestCase):
    def test_plan_for_spotify_contains_npu_and_xmos_sections(self):
        plan = studio.build_studio_plan("spotify", user_id="u1", provider="QNNExecutionProvider")
        rendered = studio.render_studio_plan(plan)

        self.assertEqual(plan.service, "spotify")
        self.assertEqual(plan.npu_plan.acceleration, "NPU")
        self.assertEqual(plan.npu_plan.provider, "QNNExecutionProvider")
        self.assertEqual(plan.xmos_plan.dac_model, "SABAJ A20D(ES)")
        self.assertIn("Spatial profile:", rendered)
        self.assertIn("XMOS low-latency plan:", rendered)
        self.assertIn("Windows EXE build:", rendered)

    def test_rejects_unknown_service(self):
        with self.assertRaises(ValueError):
            studio.build_studio_plan("tidal", user_id="u1")

    def test_realtime_recommendation_update_moves_embedding(self):
        state = studio.initialize_recommendation_state("user-a")
        updated = studio.update_recommendation_state(
            state,
            {"clarity": 1.0, "depth": 0.2, "vocal_presence": 0.8, "bass_control": 0.6},
            learning_rate=0.2,
        )

        self.assertEqual(updated.updates, 1)
        self.assertGreater(updated.embedding["clarity"], state.embedding["clarity"])
        self.assertLess(updated.embedding["depth"], state.embedding["depth"])

        bias = studio.recommend_next_track_bias(updated)
        self.assertIn("vocal_focus", bias)
        self.assertLessEqual(bias["vocal_focus"], 1.0)

    def test_plan_command_can_write_json_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "studio_plan.json"
            args = type(
                "Args",
                (),
                {
                    "service": "apple-music",
                    "user_id": "listener",
                    "sample_rate": 96000,
                    "frame_size": 256,
                    "provider": "QNNExecutionProvider",
                    "output": output,
                },
            )()
            rc = studio.plan_cmd(args)
            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["service"], "apple-music")
            self.assertEqual(payload["npu_plan"]["provider"], "QNNExecutionProvider")


if __name__ == "__main__":
    unittest.main()
