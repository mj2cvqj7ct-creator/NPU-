"""Small WAV reader/writer for offline DSP validation."""

from __future__ import annotations

import wave
from pathlib import Path

from .audio_types import AudioFormat, AudioFrame


def read_wav(path: str | Path) -> AudioFrame:
    """Read 16-bit PCM stereo WAV into normalized float samples."""

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        if channels != 2:
            raise ValueError("only stereo WAV files are supported")
        if sample_width != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        raw = wav_file.readframes(frame_count)

    samples = []
    for index in range(0, len(raw), 4):
        left = int.from_bytes(raw[index : index + 2], "little", signed=True) / 32768.0
        right = int.from_bytes(raw[index + 2 : index + 4], "little", signed=True) / 32768.0
        samples.append((left, right))
    return AudioFrame.from_iterable(samples, AudioFormat(sample_rate=sample_rate, channels=channels))


def write_wav(path: str | Path, frame: AudioFrame) -> None:
    """Write normalized float samples to 16-bit PCM stereo WAV."""

    frame.fmt.validate()
    payload = bytearray()
    for left, right in frame.samples:
        payload.extend(_float_to_pcm16(left).to_bytes(2, "little", signed=True))
        payload.extend(_float_to_pcm16(right).to_bytes(2, "little", signed=True))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(frame.fmt.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame.fmt.sample_rate)
        wav_file.writeframes(bytes(payload))


def _float_to_pcm16(value: float) -> int:
    clipped = max(-1.0, min(1.0, value))
    if clipped >= 0.0:
        return int(clipped * 32767.0)
    return int(clipped * 32768.0)
