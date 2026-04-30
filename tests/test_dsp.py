import math

import pytest

from snapdragon_audio_enhancer.dsp import (
    TRUE_PEAK_CEILING,
    EnhancementSettings,
    analyze_frame,
    enhance_frame,
)
from snapdragon_audio_enhancer.profiles import get_service_profile


def _sine_frame(amplitude: float = 0.25, samples: int = 960) -> list[list[float]]:
    return [
        [
            amplitude * math.sin(2.0 * math.pi * 440.0 * i / 48_000),
            amplitude * math.sin(2.0 * math.pi * 440.0 * i / 48_000),
        ]
        for i in range(samples)
    ]


def test_analyze_frame_reports_expected_metrics() -> None:
    metrics = analyze_frame(_sine_frame(amplitude=0.5))

    assert -10.0 < metrics.rms_db < -8.0
    assert 0.49 < metrics.peak <= 0.5
    assert metrics.zero_crossing_rate > 0.0
    assert metrics.clipping_ratio == 0.0


def test_enhance_frame_limits_hot_audio() -> None:
    settings = EnhancementSettings.from_profile(get_service_profile("spotify"))
    hot_frame = [[1.8, -1.8], [1.5, -1.5], [-1.4, 1.4]] * 64

    enhanced, metrics = enhance_frame(hot_frame, 48_000, settings)

    assert len(enhanced) == len(hot_frame)
    assert metrics.peak <= TRUE_PEAK_CEILING + 0.02


def test_enhance_frame_rejects_bad_channel_count() -> None:
    settings = EnhancementSettings.from_profile(get_service_profile("spotify"))

    with pytest.raises(ValueError, match="two channels"):
        enhance_frame([[0.0, 0.1, 0.2]], 48_000, settings)
