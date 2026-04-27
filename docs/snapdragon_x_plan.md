 # Snapdragon X NPU integration plan
 
 This project keeps the shipped code portable while defining the ARM64 Windows
 integration seam for Snapdragon X devices.
 
 ## Runtime graph
 
 ```text
 Spotify / Apple Music / YouTube Music
   -> Windows Audio Session loopback capture
   -> 48 kHz float stereo ring buffer
   -> frame slicer, 10 ms target
   -> DSP preprocess and feature extraction
   -> ONNX Runtime QNN Execution Provider
   -> DSP postprocess and true-peak limiter
   -> WASAPI exclusive / virtual audio / ASIO-capable DAC
 ```
 
 ## NPU responsibilities
 
 The first NPU-backed graph should stay small enough for predictable latency:
 
 - spectral tilt, density, transient, and vocal-presence feature estimation
 - adaptive gain controls for low shelf, presence, air, and stereo side image
 - per-service calibration values for Spotify, Apple Music, and YouTube Music
 - local preference vector inference from user-selected sound presets
 
 The Python `pipeline` module mirrors this graph with deterministic CPU logic so
 tests can validate peaks, stereo shape, and service-specific compensation before
 ONNX/QNN binaries are present.
 
 ## Backend order
 
 1. ONNX Runtime QNN Execution Provider on Windows ARM64 Snapdragon X.
 2. DirectML for compatible Windows systems without QNN availability.
 3. CPU DSP reference path, used by this repository and tests.
 
 ## Boundaries
 
 The enhancer does not bypass DRM, record streaming audio for storage, upload
 listening history, or modify the official recommendation systems. Local
 personalization only changes the post-processing profile chosen on the user's
 device.
