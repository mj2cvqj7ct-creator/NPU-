# Snapdragon X NPU integration plan

This repository cannot directly alter Spotify, Apple Music, or YouTube Music
streams. Those apps decrypt and render audio inside their own protected
pipelines. The practical integration point is a system audio post-processing
stage after playback has reached the OS mixer.

## Audio quality path

1. Capture the rendered system audio with the Windows audio session APIs.
2. Split the signal into short frames, for example 10-20 ms at 48 kHz.
3. Run a low-latency enhancement graph:
   - loudness normalization
   - dynamic range control
   - soft clipping protection
   - optional ONNX denoiser or spatial enhancement model
4. Route the processed audio to a virtual output device.

For Snapdragon X ARM64 devices, the neural portion should be exported to ONNX
and run through ONNX Runtime with the QNN Execution Provider. Classic DSP steps
can stay on CPU because they are deterministic and cheap.

## Recommendation assist path

Streaming recommendation engines are server-side and cannot be replaced inside
the official apps. A companion recommender can still help by importing available
data:

- Spotify Web API playlist and saved-track metadata
- Apple Music library metadata through MusicKit where available
- YouTube Music exported history or user-maintained playlists

The local assistant can calculate embeddings, cluster listening patterns, and
produce playlist suggestions without modifying the official services.

## Current demo

The Python CLI implements the deterministic audio stage with a
`snapdragon-x-npu` profile. It intentionally falls back to CPU so development
and tests work on non-Windows CI and cloud machines.
