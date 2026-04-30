from __future__ import annotations

from pathlib import Path
import wave

from .audio_types import AudioFrame, clamp_sample


def read_wav(path: str | Path) -> AudioFrame:
    """Read a 16-bit PCM stereo WAV into normalized float frames."""

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        if channels != 2:
            raise ValueError(f"Expected stereo WAV, got {channels} channels")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit samples")
        raw = wav_file.readframes(frame_count)

    frames: list[tuple[float, float]] = []
    scale = 32768.0
    for offset in range(0, len(raw), 4):
        left = int.from_bytes(raw[offset : offset + 2], "little", signed=True) / scale
        right = int.from_bytes(raw[offset + 2 : offset + 4], "little", signed=True) / scale
        frames.append((left, right))
    return AudioFrame(sample_rate=sample_rate, frames=tuple(frames))


def write_wav(path: str | Path, audio: AudioFrame) -> None:
    """Write normalized float frames as 16-bit PCM stereo WAV."""

    payload = bytearray()
    for left, right in audio.frames:
        payload.extend(_float_to_i16(left).to_bytes(2, "little", signed=True))
        payload.extend(_float_to_i16(right).to_bytes(2, "little", signed=True))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(audio.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(bytes(payload))


def _float_to_i16(sample: float) -> int:
    clipped = clamp_sample(sample, 0.999969482421875)
    return int(round(clipped * 32767.0))
