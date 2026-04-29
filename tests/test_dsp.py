import numpy as np

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame
from snapdragon_npu_audio_enhancer.dsp import (
    EnhancementControls,
    FeatureExtractor,
    RuleBasedEnhancer,
    db_to_linear,
)


def _tone(freq: float, seconds: float = 0.1, sample_rate: int = 48_000) -> AudioFrame:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    mono = 0.4 * np.sin(2 * np.pi * freq * t)
    return AudioFrame(np.column_stack([mono, mono]), sample_rate=sample_rate)


def test_feature_extractor_reports_safe_ranges() -> None:
    features = FeatureExtractor().extract(_tone(440))

    assert -20.0 < features.rms_db < -5.0
    assert features.crest_factor_db > 0.0
    assert 0.0 <= features.low_band_energy <= 1.0
    assert 0.0 <= features.vocal_band_energy <= 1.0
    assert 0.0 <= features.high_band_energy <= 1.0
    assert 0.0 <= features.transient_density <= 1.0
    assert features.peak_db < 0.0


def test_rule_based_enhancer_changes_signal_and_limits_peak() -> None:
    frame = _tone(120)
    enhancer = RuleBasedEnhancer()
    controls = EnhancementControls(
        pre_gain_db=3.0,
        bass_gain_db=2.5,
        presence_gain_db=0.5,
        air_gain_db=-1.0,
        transient_gain_db=1.0,
        limiter_ceiling_db=-1.0,
    )

    enhanced = enhancer.process(frame, controls)

    assert enhanced.samples.shape == frame.samples.shape
    assert not np.allclose(enhanced.samples, frame.samples)
    assert np.max(np.abs(enhanced.samples)) <= db_to_linear(-1.0) + 1e-6


def test_clipping_derives_conservative_controls() -> None:
    frame = AudioFrame(np.full((256, 2), 1.0, dtype=np.float32), 48_000)
    features = FeatureExtractor().extract(frame)
    controls = RuleBasedEnhancer().derive_controls(features)

    assert controls.pre_gain_db <= -1.5
    assert controls.bass_gain_db <= 0.5
