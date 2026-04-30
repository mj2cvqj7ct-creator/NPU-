import numpy as np
import pytest

from snapdragon_npu_audio_enhancer.audio_frame import AudioFrame, ensure_stereo


def test_audio_frame_accepts_mono_and_tracks_duration() -> None:
    frame = AudioFrame(np.zeros(480, dtype=np.float32), sample_rate=48_000)

    assert frame.channels == 1
    assert frame.frame_count == 480
    assert frame.duration_seconds == pytest.approx(0.01)


def test_ensure_stereo_duplicates_mono_input() -> None:
    mono = AudioFrame(np.array([0.25, -0.5], dtype=np.float32), sample_rate=48_000)

    stereo = ensure_stereo(mono)

    assert stereo.samples.shape == (2, 2)
    assert np.allclose(stereo.samples[:, 0], mono.samples[:, 0])
    assert np.allclose(stereo.samples[:, 1], mono.samples[:, 0])


def test_audio_frame_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AudioFrame(np.array([], dtype=np.float32), sample_rate=48_000)
