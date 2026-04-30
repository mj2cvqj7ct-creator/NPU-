from __future__ import annotations

import array
import math
import wave
from pathlib import Path

from snapdragon_audio_enhancer.audio import AudioBuffer, read_wav, write_wav
from snapdragon_audio_enhancer.inference import EnhancementHints, InferenceBackend
from snapdragon_audio_enhancer.pipeline import EnhancementPipeline


class FixedBackend(InferenceBackend):
    def infer_hints(self, buffer: AudioBuffer) -> EnhancementHints:
        return EnhancementHints(clarity=0.2, warmth=0.15, stereo_width=0.03)


def test_pipeline_keeps_audio_bounded() -> None:
    samples = tuple((0.95 * math.sin(index / 6.0), 0.95 * math.sin(index / 6.0)) for index in range(512))
    enhanced = EnhancementPipeline(inference_backend=FixedBackend()).enhance(AudioBuffer(48000, samples))

    assert enhanced.frame_count == len(samples)
    assert max(abs(sample) for frame in enhanced.samples for sample in frame) <= 1.0


def test_wav_round_trip_uses_pcm16(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    output = tmp_path / "output.wav"

    pcm = array.array("h", [0, 1000, -1000, 0, 2000, -2000, -2000, 2000])
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(pcm.tobytes())

    buffer = read_wav(source)
    write_wav(output, buffer)
    reread = read_wav(output)

    assert reread.sample_rate == 48000
    assert reread.channels == 2
    assert reread.frame_count == 4
