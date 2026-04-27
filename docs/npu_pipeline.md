# Snapdragon X NPU Audio Pipeline

This document narrows the README plan into the first implementation seam for a
system-wide enhancer. The code intentionally starts with deterministic DSP and a
provider selector because the Windows capture layer and Qualcomm runtime are
platform-dependent.

## Processing contract

- Input: 48 kHz, 32-bit float, stereo PCM frames from WASAPI loopback.
- Frame size: 10-20 ms, represented by `AudioFrame`.
- Output: the same sample rate and channel layout, peak-limited below the
  configured true-peak ceiling.
- No audio is persisted, redistributed, or sent to a remote service.

## Current DSP chain

`AudioEnhancer` applies:

1. bounded loudness gain toward a service-wide target;
2. conservative left/right balance correction;
3. a lightweight presence/low-volume lift approximation;
4. final true-peak limiting.

These rules are deliberately conservative so Spotify, Apple Music, and YouTube
Music output can be processed uniformly without changing the apps or their
catalog/recommendation systems.

## NPU integration seam

`InferenceProviderSelector` models the runtime preference order:

1. ONNX Runtime QNN Execution Provider for Snapdragon X NPU;
2. DirectML fallback;
3. CPU fallback.

Production code should replace the environment probe with actual runtime
capability checks once the Windows ARM64 deployment target has ONNX Runtime and
Qualcomm QNN packages available.
