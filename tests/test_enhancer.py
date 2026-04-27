import math
from pathlib import Path
import unittest

from src.dsp.enhancer import AudioEnhancer, NpuEnhancementControls, load_service_profiles
from src.inference.runtime import CpuFallbackRuntime, select_runtime


ROOT = Path(__file__).resolve().parents[1]


def _sine_block(frequency=440.0, amplitude=0.2, frames=480, sample_rate=48_000):
    return [
        (
            amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate),
            amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate),
        )
        for index in range(frames)
    ]


class AudioEnhancerTests(unittest.TestCase):
    def test_loads_service_profiles(self):
        profiles = load_service_profiles(ROOT / "config" / "service_profiles.json")

        self.assertEqual(set(profiles), {"spotify", "apple_music", "youtube_music"})
        self.assertLess(profiles["apple_music"].target_lufs, profiles["youtube_music"].target_lufs)

    def test_enhancer_limits_peak_after_gain_and_eq(self):
        profiles = load_service_profiles(ROOT / "config" / "service_profiles.json")
        enhancer = AudioEnhancer(profiles["spotify"])
        loud_block = _sine_block(amplitude=1.3)

        output = enhancer.process(
            loud_block,
            NpuEnhancementControls(clarity=1.0, warmth=1.0, transient_restore=1.0),
        )

        ceiling = 10.0 ** (profiles["spotify"].limiter_ceiling_db / 20.0)
        self.assertTrue(output)
        self.assertLessEqual(max(abs(sample) for frame in output for sample in frame), ceiling)

    def test_enhancer_preserves_frame_count_and_stereo_symmetry(self):
        profiles = load_service_profiles(ROOT / "config" / "service_profiles.json")
        enhancer = AudioEnhancer(profiles["youtube_music"])
        block = _sine_block(amplitude=0.08)

        output = enhancer.process(block)

        self.assertEqual(len(output), len(block))
        self.assertTrue(all(abs(left - right) < 1e-9 for left, right in output))

    def test_cpu_runtime_returns_bounded_controls(self):
        runtime = CpuFallbackRuntime()

        controls = runtime.infer_controls(_sine_block(frequency=1800.0, amplitude=0.1))

        self.assertGreaterEqual(controls.clarity, -1.0)
        self.assertLessEqual(controls.clarity, 1.0)
        self.assertGreaterEqual(controls.warmth, -1.0)
        self.assertLessEqual(controls.warmth, 1.0)
        self.assertGreaterEqual(controls.de_mud, 0.0)
        self.assertLessEqual(controls.de_mud, 1.0)
        self.assertGreaterEqual(controls.transient_restore, 0.0)
        self.assertLessEqual(controls.transient_restore, 1.0)

    def test_runtime_selection_uses_cpu_fallback_without_model(self):
        runtime = select_runtime()

        self.assertEqual(runtime.name, "cpu-fallback")


if __name__ == "__main__":
    unittest.main()
