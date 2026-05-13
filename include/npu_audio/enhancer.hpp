#pragma once

#include "npu_audio/audio_buffer.hpp"
#include "npu_audio/inference_backend.hpp"
#include "npu_audio/processing_types.hpp"

#include <memory>

namespace npu_audio {

[[nodiscard]] float linearToDb(float value) noexcept;
[[nodiscard]] float dbToLinear(float valueDb) noexcept;
[[nodiscard]] AudioFeatures analyzeAudio(const AudioBuffer &buffer);

class AudioEnhancer {
public:
  explicit AudioEnhancer(
      std::unique_ptr<InferenceBackend> backend =
          InferenceBackend::createPreferred());

  EnhancementReport process(AudioBuffer &buffer,
                            const EnhancementProfile &userProfile = {});

private:
  std::unique_ptr<InferenceBackend> backend_;
};

} // namespace npu_audio
