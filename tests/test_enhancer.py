import math
from pathlib import Path

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


def test_loads_service_profiles():
    profiles = load_service_profiles(ROOT / "config" / "service_profiles.json")

    assert set(profiles) == {"spotify", "apple_music", "youtube_music"}
    assert profiles["apple_music"].target_lufs < profiles["youtube_music"].target_lufs


def test_enhancer_limits_peak_after_gain_and_eq():
    profiles = load_service_profiles(ROOT / "config" / "service_profiles.json")
    enhancer = AudioEnhancer(profiles["spotify"])
    loud_block = _sine_block(amplitude=1.3)

    output = enhancer.process(
        loud_block,
        NpuEnhancementControls(clarity=1.0, warmth=1.0, transient_restore=1.0),
    )

    ceiling = 10.0 ** (profiles["spotify"].limiter_ceiling_db / 20.0)
    assert output
    assert max(abs(sample) for frame in output for sample in frame) <= ceiling


def test_enhancer_preserves_frame_count_and_stereo_symmetry():
    profiles = load_service_profiles(ROOT / "config" / "service_profiles.json")
    enhancer = AudioEnhancer(profiles["youtube_music"])
    block = _sine_block(amplitude=0.08)

    output = enhancer.process(block)

    assert len(output) == len(block)
    assert all(abs(left - right) < 1e-9 for left, right in output)


def test_cpu_runtime_returns_bounded_controls():
    runtime = CpuFallbackRuntime()

    controls = runtime.infer_controls(_sine_block(frequency=1800.0, amplitude=0.1))

    assert -1.0 <= controls.clarity <= 1.0
    assert -1.0 <= controls.warmth <= 1.0
    assert 0.0 <= controls.de_mud <= 1.0
    assert 0.0 <= controls.transient_restore <= 1.0


def test_runtime_selection_uses_cpu_fallback_without_model():
    runtime = select_runtime()

    assert runtime.name == "cpu-fallback"
