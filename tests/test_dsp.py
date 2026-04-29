from __future__ import annotations

import math
import unittest

from npu_audio_enhancer.dsp import (
    AudioFrameError,
    apply_dynamic_eq,
    frame_stats,
    normalize_loudness,
    true_peak_limit,
    validate_stereo_frame,
)
from npu_audio_enhancer.profiles import get_service_profile


class DspTests(unittest.TestCase):
    def test_validate_stereo_frame_rejects_non_finite_samples(self) -> None:
        with self.assertRaises(AudioFrameError):
            validate_stereo_frame([[0.0, math.nan], [0.0, 0.1]])

    def test_normalize_loudness_clamps_gain_and_preserves_peak_headroom(self) -> None:
        quiet_frame = [[0.02] * 128, [0.02] * 128]

        processed, stats = normalize_loudness(
            quiet_frame,
            target_lufs=-14.0,
            max_gain_db=3.0,
        )

        self.assertAlmostEqual(stats.loudness_gain_db, 3.0)
        self.assertLessEqual(frame_stats(processed).peak, 0.98)
        self.assertGreater(frame_stats(processed).rms, frame_stats(quiet_frame).rms)

    def test_true_peak_limit_keeps_samples_below_ceiling(self) -> None:
        hot_frame = [[1.3, -1.2, 0.4], [1.1, -1.4, 0.5]]

        limited = true_peak_limit(hot_frame, ceiling_dbfs=-1.0)

        self.assertLessEqual(frame_stats(limited).peak, 10 ** (-1.0 / 20.0))

    def test_dynamic_eq_keeps_shape_and_headroom(self) -> None:
        profile = get_service_profile("spotify")
        frame = [
            [0.0, 0.15, -0.15, 0.25, -0.2, 0.1],
            [0.0, -0.12, 0.12, -0.22, 0.18, -0.08],
        ]

        processed = apply_dynamic_eq(frame, profile.eq)

        self.assertEqual(len(processed), 2)
        self.assertEqual(len(processed[0]), len(frame[0]))
        self.assertEqual(len(processed[1]), len(frame[1]))
        self.assertLessEqual(frame_stats(processed).peak, 0.98)
        self.assertNotEqual(processed, frame)


if __name__ == "__main__":
    unittest.main()
