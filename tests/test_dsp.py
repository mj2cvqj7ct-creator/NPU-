import numpy as np

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame
from snapdragon_npu_audio_enhancer.dsp import EnhancementControls, FeatureExtractor, RuleBasedEnhancer, db_to_linear
from snapdragon_npu_audio_enhancer.profiles import MusicService, get_service_profile


def sine(freq: float, seconds: float = 0.05, sample_rate: int = 48_000) -> AudioFrame:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    samples = 0.2 * np.sin(2.0 * np.pi * freq * t)
    return AudioFrame(np.column_stack([samples, samples]), sample_rate)


def test_feature_extractor_detects_low_frequency_energy() -> None:
    features = FeatureExtractor().extract(sine(100.0))

    assert features.low_band_energy > features.high_band_energy
    assert features.rms_db < -10.0
    assert 0.0 <= features.side_energy <= 1.0


def test_rule_based_enhancer_respects_limiter_ceiling() -> None:
    frame = sine(1000.0)
    controls = EnhancementControls(
        pre_gain_db=24.0,
        bass_gain_db=3.0,
        presence_gain_db=3.0,
        air_gain_db=2.0,
        limiter_ceiling_db=-3.0,
    )

    output = RuleBasedEnhancer().process(frame, controls)

    assert output.peak() <= db_to_linear(-3.0) + 1e-6


def test_service_profiles_change_control_tuning() -> None:
    features = FeatureExtractor().extract(sine(220.0))
    enhancer = RuleBasedEnhancer()

    spotify = enhancer.derive_controls(features, get_service_profile(MusicService.SPOTIFY))
    apple = enhancer.derive_controls(features, get_service_profile(MusicService.APPLE_MUSIC))

    assert spotify.pre_gain_db != apple.pre_gain_db
    assert spotify.bass_gain_db != apple.bass_gain_db
