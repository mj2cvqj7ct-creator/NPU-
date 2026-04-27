import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_enhancer.dsp import AudioEnhancer, AudioFrame, EnhancementConfig


class AudioEnhancerTests(unittest.TestCase):
    def test_frame_round_trips_interleaved_samples(self) -> None:
        frame = AudioFrame.from_interleaved([0.1, -0.1, 0.2, -0.2])

        self.assertEqual(frame.channels, 2)
        self.assertEqual(frame.frame_samples, 2)
        self.assertEqual(frame.duration_ms, 2 * 1000 / 48_000)
        self.assertEqual(frame.to_interleaved(), [0.1, -0.1, 0.2, -0.2])

    def test_enhancer_limits_peak_below_configured_ceiling(self) -> None:
        enhancer = AudioEnhancer(
            EnhancementConfig(
                target_loudness_dbfs=0.0,
                max_gain_db=12.0,
                true_peak_ceiling_dbfs=-1.0,
                vocal_presence_db=0.0,
                low_volume_lift_db=0.0,
            )
        )
        frame = AudioFrame.from_interleaved([0.9, -0.9] * 480)

        processed, metrics = enhancer.process(frame)

        self.assertLessEqual(max(abs(sample) for sample in processed.to_interleaved()), 0.892)
        self.assertLess(metrics.limiter_gain_db, 0.0)
        self.assertLessEqual(metrics.output_peak_dbfs, -0.99)

    def test_enhancer_raises_quiet_content_without_clipping(self) -> None:
        enhancer = AudioEnhancer(
            EnhancementConfig(
                target_loudness_dbfs=-16.0,
                max_gain_db=9.0,
                true_peak_ceiling_dbfs=-1.0,
                vocal_presence_db=0.0,
                low_volume_lift_db=0.0,
            )
        )
        frame = AudioFrame.from_interleaved([0.05, -0.05] * 480)

        processed, metrics = enhancer.process(frame)

        self.assertEqual(metrics.applied_gain_db, 9.0)
        self.assertGreater(max(abs(sample) for sample in processed.to_interleaved()), 0.13)
        self.assertLess(metrics.output_peak_dbfs, -1.0)

    def test_balance_correction_reduces_channel_delta(self) -> None:
        enhancer = AudioEnhancer(
            EnhancementConfig(
                target_loudness_dbfs=-18.0,
                max_gain_db=0.0,
                true_peak_ceiling_dbfs=-1.0,
                balance_correction_strength=0.5,
                vocal_presence_db=0.0,
                low_volume_lift_db=0.0,
            )
        )
        frame = AudioFrame.from_interleaved([0.4, 0.1] * 480)

        processed, metrics = enhancer.process(frame)

        left = processed.channel_values(0)
        right = processed.channel_values(1)
        self.assertGreater(metrics.balance_delta_db, 0.0)
        self.assertLess(sum(abs(value) for value in left) / len(left), 0.4)
        self.assertGreater(sum(abs(value) for value in right) / len(right), 0.1)


if __name__ == "__main__":
    unittest.main()
