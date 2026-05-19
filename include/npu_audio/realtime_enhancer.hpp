#pragma once

#include "npu_audio/audio_buffer.hpp"
#include "npu_audio/inference_backend.hpp"
#include "npu_audio/processing_types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace npu_audio {

struct RealtimeConfig {
  std::uint32_t sampleRate = 48000;
  std::uint16_t channels = 2;
  float frameDurationMs = 10.0F;
  EnhancementProfile userProfile;
};

struct RealtimeReport {
  EnhancementReport enhancement;
  std::size_t processedFrames = 0;
  float frameDurationMs = 0.0F;
};

class RealtimeEnhancer {
public:
  explicit RealtimeEnhancer(
      RealtimeConfig config = {},
      std::unique_ptr<InferenceBackend> backend =
          InferenceBackend::createPreferred());

  [[nodiscard]] std::size_t frameSize() const noexcept { return frameSize_; }
  [[nodiscard]] const RealtimeConfig &config() const noexcept { return config_; }

  RealtimeReport processFrame(AudioBuffer &frame);
  void reset();

private:
  RealtimeConfig config_;
  std::unique_ptr<InferenceBackend> backend_;
  std::size_t frameSize_ = 0;
  std::vector<float> lowState_;
  std::vector<float> highLowpassState_;
  EnhancementProfile smoothedProfile_;
  bool hasSmoothedProfile_ = false;
  float compressorEnvelope_ = 0.0F;
  float smoothedLoudnessGainDb_ = 0.0F;
  bool hasSmoothedLoudness_ = false;
};

} // namespace npu_audio
