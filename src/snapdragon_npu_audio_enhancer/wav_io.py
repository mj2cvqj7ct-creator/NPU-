from __future__ import annotations

import wave
from pathlib import Path

from .audio_frame import AudioFrame


def read_wav(path: str | Path) -> AudioFrame:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        payload = wav.readframes(frame_count)

    if channels != 2:
        raise ValueError(f"expected stereo WAV, got {channels} channels")

    if sample_width == 2:
        samples = []
        for index in range(0, len(payload), 2):
            value = int.from_bytes(payload[index : index + 2], "little", signed=True)
            samples.append(max(-1.0, value / 32768.0))
    elif sample_width == 4:
        samples = []
        for index in range(0, len(payload), 4):
            value = int.from_bytes(payload[index : index + 4], "little", signed=True)
            samples.append(max(-1.0, value / 2147483648.0))
    else:
        raise ValueError(f"unsupported sample width: {sample_width * 8} bit")

    return AudioFrame.from_interleaved(samples, sample_rate=sample_rate, channels=channels)


def write_wav(path: str | Path, frame: AudioFrame) -> None:
    payload = bytearray()
    for sample in frame.interleaved():
        value = int(round(max(-1.0, min(1.0, sample)) * 32767.0))
        payload.extend(value.to_bytes(2, "little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(frame.channels)
        wav.setsampwidth(2)
        wav.setframerate(frame.sample_rate)
        wav.writeframes(bytes(payload))
