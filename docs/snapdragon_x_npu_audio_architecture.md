# Snapdragon X NPU audio architecture

This document turns the high-level README plan into an implementation contract for a
system-wide enhancer that can improve Spotify, Apple Music, and YouTube Music output
without modifying the services themselves.

## Boundary

- Capture only PCM samples already rendered by the OS audio stack.
- Do not decrypt, save, or redistribute protected streams.
- Keep all listener profiling local to the device.
- Treat service names as profile hints, not as hooks into private application APIs.

## Runtime pipeline

```text
WASAPI loopback or APO input
  -> frame normalizer (48 kHz, stereo, float32)
  -> feature extractor
  -> adaptive inference backend
       - Snapdragon X: ONNX Runtime QNN EP or Qualcomm QNN
       - fallback: DirectML or CPU heuristics
  -> DSP controls
  -> low-latency enhancement chain
  -> limiter
  -> render endpoint or virtual device
```

The prototype in `src/npu_audio_enhancer` implements the same data flow on WAV files
so the DSP behavior can be tested before a Windows capture layer exists.

## NPU responsibilities

The NPU should not run a large generative model in the audio callback. It should run
small frame-level or short-context models that emit DSP controls:

- loudness gain target for the current passage
- bass warmth and vocal clarity deltas
- compression amount for dense masters
- stereo width safety factor
- listener preference embedding for local personalization

This keeps the real-time path deterministic and lets the CPU perform inexpensive DSP
while the NPU handles classification and control estimation.

## Latency budget

| Stage | Target |
| --- | ---: |
| Capture buffer | 5-10 ms |
| Feature extraction | < 2 ms |
| NPU/QNN inference | < 5 ms |
| DSP and limiter | < 2 ms |
| Render buffer | 5-10 ms |

The complete path should remain below 40 ms end-to-end. If NPU inference misses its
budget, the pipeline must reuse the previous controls or fall back to CPU controls
instead of blocking audio rendering.

## Service profiles

| Service | Profile purpose |
| --- | --- |
| Spotify | Smooth loudness differences across lossy and normalized tracks. |
| Apple Music | Preserve headroom for lossless tracks and apply restrained correction. |
| YouTube Music | Control volume jumps and browser playback inconsistency. |

The profiles are intentionally conservative. Dramatic improvement should come from
stable low-latency adaptation, not from permanently exaggerated EQ.

## Prototype command

```bash
PYTHONPATH=src python3 -m npu_audio_enhancer input.wav output.wav --service spotify
```

Use `--backend npu` on Snapdragon X machines only after a QNN-backed implementation is
wired in. Until then the command reports the CPU adaptive backend so tests are honest
about hardware usage.
