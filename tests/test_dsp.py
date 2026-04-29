from __future__ import annotations

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from snapdragon_npu_audio_enhancer import (
    EnhancementConfig,
    HeuristicInferenceProvider,
    StreamingService,
    enhance_frame,
)


def _sine_frame(samples: int = 480, *, amplitude: float = 0.2) -> list[tuple[float, float]]:
    return [
        (
            amplitude * math.sin(2.0 * math.pi * 440.0 * index / 48_000),
            amplitude * math.sin(2.0 * math.pi * 440.0 * index / 48_000 + 0.05),
        )
        for index in range(samples)
    ]


class DspTests(unittest.TestCase):
    def test_enhance_frame_limits_true_peak(self) -> None:
        frame = [(2.5, -2.5), (-2.5, 2.5)] * 240
        config = EnhancementConfig(target_lufs=0.0, true_peak_ceiling=0.5)

        output, report = enhance_frame(frame, config=config)

        self.assertTrue(output)
        self.assertLessEqual(max(max(abs(left), abs(right)) for left, right in output), 0.5 + 1e-9)
        self.assertGreater(report.limiter_reduction_db, 0.0)

    def test_service_profile_changes_reported_service(self) -> None:
        frame = _sine_frame()
        config = EnhancementConfig(service=StreamingService.SPOTIFY)

        _, report = enhance_frame(frame, config=config, provider=HeuristicInferenceProvider())

        self.assertIs(report.service, StreamingService.SPOTIFY)
        self.assertEqual(report.provider, "heuristic-fallback")

    def test_empty_frame_returns_silence_report(self) -> None:
        config = EnhancementConfig(service=StreamingService.YOUTUBE_MUSIC)

        output, report = enhance_frame([], config=config)

        self.assertEqual(output, [])
        self.assertIs(report.service, StreamingService.YOUTUBE_MUSIC)
        self.assertEqual(report.output_peak, 0.0)


if __name__ == "__main__":
    unittest.main()
