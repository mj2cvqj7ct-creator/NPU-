# Snapdragon X NPU audio algorithm

This document describes the local, OS-level algorithm implemented by the
prototype. It does not modify Spotify, Apple Music, or YouTube Music clients.
Instead, those apps render audio normally, and the enhancer processes the PCM
stream after the OS audio session exposes it.

## Frame contract

- Internal format: 48 kHz, stereo, 32-bit float PCM.
- Frame size: 10-20 ms for live playback; WAV tests can process any size.
- Safety target: output true peak stays below the profile ceiling.
- Privacy: no decoded samples or features are persisted by default.

## NPU feature boundary

The NPU side is represented by `NpuFeatureEstimator`. Its fallback
implementation is deterministic CPU code so that the algorithm is testable on
CI and Linux cloud runners. On Snapdragon X Windows ARM64, this boundary should
be replaced by an ONNX model through ONNX Runtime QNN Execution Provider.

The feature vector intentionally stays small:

- `loudness`: RMS-derived program energy.
- `bass_energy`: relative low-frequency content proxy.
- `presence_energy`: vocal and lead-instrument presence proxy.
- `air_energy`: high-frequency detail proxy.
- `transient_density`: short-term peak-to-RMS activity.
- `stereo_width`: side/mid balance.

These features control deterministic DSP stages instead of letting a model
write final samples directly. That keeps latency predictable and makes the
output safer for headphones.

## Enhancement stages

1. **Service-aware preset selection**
   - Spotify: stronger codec recovery, volume smoothing, and vocal clarity.
   - Apple Music: lighter processing for lossless playback and headroom.
   - YouTube Music: stronger loudness leveling for mixed browser content.
2. **Adaptive tone shaping**
   - Low shelf support is raised only when bass energy is weak.
   - Presence is lifted when vocal-band energy is low.
   - Air is restored conservatively to avoid harshness.
3. **Mid/side spatial control**
   - Stereo width is widened when the source is narrow.
   - Width expansion is reduced for already-wide or transient-heavy content.
4. **Dynamic range control**
   - A soft knee compressor keeps perceived loudness stable.
   - Makeup gain reacts to estimated loudness and service preset.
5. **Limiter**
   - Final normalization enforces the profile true-peak ceiling.

## Production integration notes

For the native Windows version, keep CPU DSP and NPU inference decoupled:

```text
WASAPI loopback frame
  -> feature extraction tensor
  -> ONNX Runtime QNN EP on Snapdragon X NPU
  -> small control vector
  -> deterministic DSP + limiter
  -> WASAPI/virtual device render
```

Use QNN only for model inference that benefits from the NPU. Cheap operations
such as deinterleaving, IIR filters, and limiting should remain on CPU to avoid
copy overhead and to preserve deterministic real-time behavior.
