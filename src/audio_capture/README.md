# Audio Capture Layer

The portable DSP prototype intentionally keeps WASAPI and APO code out of the core library. On ARM64 Windows, this layer should provide 48 kHz, 32-bit float stereo frames from WASAPI loopback and feed them into `snae::dsp::EnhancementPipeline`.

Planned Windows-specific responsibilities:

- Open the default render endpoint in loopback mode.
- Convert captured audio to the internal `AudioFrame` representation.
- Keep end-to-end buffering below the 40 ms target.
- Avoid recording, persisting, or redistributing service audio.
- Later replace the loopback prototype with an APO or virtual audio device integration.
