from __future__ import annotations

import unittest

from npu_audio_enhancer import EnhancementPipeline, ServiceProfileName
from npu_audio_enhancer.dsp import frame_stats


def _fixture_frame(samples: int = 960) -> list[list[float]]:
    left = []
    right = []
    for index in range(samples):
        phase = (index % 48) / 48.0
        left.append((phase - 0.5) * 0.18)
        right.append((0.5 - phase) * 0.16)
    return [left, right]


class EnhancementPipelineTest(unittest.TestCase):
    def test_processes_supported_music_services(self) -> None:
        for service in ServiceProfileName:
            with self.subTest(service=service.value):
                pipeline = EnhancementPipeline(service.value)
                result = pipeline.process(_fixture_frame())

                self.assertEqual(result.service, service.value)
                self.assertEqual(result.npu_backend, "PassthroughNpuEnhancer")
                self.assertEqual(len(result.frame), 2)
                self.assertEqual(len(result.frame[0]), 960)
                self.assertLessEqual(frame_stats(result.frame).peak, 10 ** (-1.0 / 20.0))

    def test_rejects_non_48khz_pipeline_for_initial_realtime_core(self) -> None:
        with self.assertRaises(ValueError):
            EnhancementPipeline("spotify", sample_rate=44_100)


if __name__ == "__main__":
    unittest.main()
