from __future__ import annotations

from array import array
import wave

from .audio import AudioFrame, clamp_sample


def read_wav(path: str) -> AudioFrame:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("only 16-bit PCM WAV files are supported by the prototype")

    pcm = array("h")
    pcm.frombytes(frames)
    if pcm.itemsize != 2:
        raise RuntimeError("unexpected platform sample size")
    values = [sample / 32768.0 for sample in pcm]
    return AudioFrame.from_interleaved(sample_rate=sample_rate, channels=channels, values=values)


def write_wav(path: str, frame: AudioFrame) -> None:
    pcm = array("h")
    for value in frame.interleaved():
        sample = int(round(clamp_sample(value) * 32767.0))
        pcm.append(max(-32768, min(32767, sample)))

    with wave.open(path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(frame.sample_rate)
        wav.writeframes(pcm.tobytes())
