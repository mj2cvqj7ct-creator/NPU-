import numpy as np
import pytest

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame, ensure_stereo


def test_audio_frame_normalizes_mono_to_2d_float32() -> None:
    frame = AudioFrame(np.array([0.0, 0.5, -0.5]), sample_rate=48_000)

    assert frame.samples.dtype == np.float32
    assert frame.samples.shape == (3, 1)
    assert frame.frame_count == 3
    assert frame.channels == 1


def test_ensure_stereo_duplicates_mono() -> None:
    frame = AudioFrame(np.array([0.25, -0.25]), sample_rate=48_000)
    stereo = ensure_stereo(frame)

    assert stereo.samples.shape == (2, 2)
    np.testing.assert_allclose(stereo.samples[:, 0], stereo.samples[:, 1])


def test_audio_frame_rejects_non_finite_samples() -> None:
    with pytest.raises(ValueError, match="finite"):
        AudioFrame(np.array([0.0, np.nan]), sample_rate=48_000)
