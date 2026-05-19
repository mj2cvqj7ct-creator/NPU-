#include "npu_audio/enhancer.hpp"
#include "npu_audio/inference_backend.hpp"
#include "npu_audio/realtime_enhancer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

npu_audio::AudioBuffer makeStereoSine(float amplitude, float frequencyHz,
                                      std::uint32_t sampleRate,
                                      std::size_t frames) {
  constexpr float pi = 3.14159265358979323846F;
  std::vector<float> samples;
  samples.reserve(frames * 2);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    const float phase =
        2.0F * pi * frequencyHz * static_cast<float>(frame) /
        static_cast<float>(sampleRate);
    const float sample = amplitude * std::sin(phase);
    samples.push_back(sample);
    samples.push_back(sample);
  }
  return npu_audio::AudioBuffer(sampleRate, 2, std::move(samples));
}

void testLimiterPreventsClipping() {
  npu_audio::AudioBuffer buffer = makeStereoSine(1.35F, 997.0F, 48000, 4800);
  npu_audio::AudioEnhancer enhancer;
  const npu_audio::EnhancementReport report = enhancer.process(buffer);

  const npu_audio::AudioFeatures output = npu_audio::analyzeAudio(buffer);
  expect(output.peakDb <= -0.85F,
         "true peak limiter should keep output below the ceiling");
  expect(report.outputPeakDb <= -0.85F,
         "enhancement report should expose the limited peak");
}

void testSilenceIsStable() {
  npu_audio::AudioBuffer buffer(48000, 2, std::vector<float>(960, 0.0F));
  npu_audio::AudioEnhancer enhancer;
  const npu_audio::EnhancementReport report = enhancer.process(buffer);

  for (const float sample : buffer.samples()) {
    expect(sample == 0.0F, "silence should remain bit-exact zero");
  }
  expect(report.loudnessGainDb == 0.0F,
         "silence should not receive loudness gain");
  expect(report.outputPeakDb <= -119.0F, "silence should report a floor peak");
}

void testBackendSelectionFallback() {
#if defined(_WIN32)
  _putenv_s("NPU_AUDIO_BACKEND", "qnn");
#else
  setenv("NPU_AUDIO_BACKEND", "qnn", 1);
#endif
  const std::unique_ptr<npu_audio::InferenceBackend> backend =
      npu_audio::InferenceBackend::createPreferred();
  const npu_audio::BackendStatus status = backend->status();

#if defined(NPU_AUDIO_ENABLE_QNN)
  expect(status.provider == npu_audio::InferenceProvider::QnnNpu,
         "QNN-enabled builds should select the QNN provider when requested");
  expect(status.accelerated, "QNN-enabled builds should report acceleration");
#else
  expect(status.provider == npu_audio::InferenceProvider::Cpu,
         "portable builds should fall back to CPU when QNN is requested");
  expect(!status.accelerated, "CPU fallback should not report acceleration");
#endif
}

void testFeatureAnalysis() {
  npu_audio::AudioBuffer buffer = makeStereoSine(0.5F, 220.0F, 48000, 4800);
  const npu_audio::AudioFeatures features = npu_audio::analyzeAudio(buffer);

  expect(!features.silent, "sine input should not be classified as silence");
  expect(features.rmsDb > -12.0F && features.rmsDb < -6.0F,
         "sine RMS should be in the expected dB range");
  expect(features.stereoCorrelation > 0.98F,
         "dual-mono sine should be highly correlated");
}

void testServiceProfiles() {
  const npu_audio::EnhancementProfile generic =
      npu_audio::profileForService("generic");
  const npu_audio::EnhancementProfile spotify =
      npu_audio::profileForService("Spotify");
  const npu_audio::EnhancementProfile apple =
      npu_audio::profileForService("apple-music");
  const npu_audio::EnhancementProfile youtube =
      npu_audio::profileForService("YouTube Music");

  expect(spotify.clarityEnhancement > generic.clarityEnhancement,
         "Spotify preset should add clarity headroom");
  expect(apple.compressorRatio < youtube.compressorRatio,
         "Apple Music preset should preserve more dynamics than YouTube Music");
  expect(youtube.limiterCeilingDb < generic.limiterCeilingDb,
         "YouTube Music preset should leave extra limiter margin");

  bool threw = false;
  try {
    (void)npu_audio::profileForService("unknown-service");
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "unknown service names should be rejected");
}

void testRealtimeFrameProcessor() {
  npu_audio::RealtimeConfig config;
  config.sampleRate = 48000;
  config.channels = 2;
  config.frameDurationMs = 10.0F;
  config.userProfile = npu_audio::profileForService("youtube-music");

  npu_audio::RealtimeEnhancer enhancer(config);
  expect(enhancer.frameSize() == 480,
         "10 ms realtime frame should map to 480 frames at 48 kHz");

  npu_audio::AudioBuffer frame = makeStereoSine(1.25F, 997.0F, 48000, 480);
  const npu_audio::RealtimeReport report = enhancer.processFrame(frame);
  const npu_audio::AudioFeatures output = npu_audio::analyzeAudio(frame);

  expect(report.processedFrames == 480,
         "realtime report should expose processed frame count");
  expect(report.frameDurationMs > 9.9F && report.frameDurationMs < 10.1F,
         "realtime report should expose the processed frame duration");
  expect(output.peakDb <= -1.15F,
         "realtime limiter should keep output below the service ceiling");
  expect(report.enhancement.outputPeakDb <= -1.15F,
         "realtime report should expose the limited output peak");
}

void testRealtimeFrameValidation() {
  bool rejectedDuration = false;
  try {
    npu_audio::RealtimeConfig config;
    config.frameDurationMs = 25.0F;
    npu_audio::RealtimeEnhancer enhancer(config);
    (void)enhancer;
  } catch (const std::invalid_argument &) {
    rejectedDuration = true;
  }
  expect(rejectedDuration,
         "realtime processor should reject frames outside the NPU budget");

  npu_audio::RealtimeEnhancer enhancer;
  npu_audio::AudioBuffer oversized =
      makeStereoSine(0.25F, 440.0F, 48000, enhancer.frameSize() + 1);
  bool rejectedOversizedFrame = false;
  try {
    (void)enhancer.processFrame(oversized);
  } catch (const std::invalid_argument &) {
    rejectedOversizedFrame = true;
  }
  expect(rejectedOversizedFrame,
         "realtime processor should reject frames over the configured size");
}

} // namespace

int main() {
  try {
    testLimiterPreventsClipping();
    testSilenceIsStable();
    testBackendSelectionFallback();
    testFeatureAnalysis();
    testServiceProfiles();
    testRealtimeFrameProcessor();
    testRealtimeFrameValidation();
  } catch (const std::exception &error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }

  std::cout << "all npu audio tests passed\n";
  return 0;
}
