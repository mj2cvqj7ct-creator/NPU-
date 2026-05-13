#pragma once

#include "npu_audio/processing_types.hpp"

#include <memory>
#include <string>

namespace npu_audio {

enum class InferenceProvider {
  QnnNpu,
  DirectMl,
  Cpu,
};

struct BackendStatus {
  InferenceProvider provider = InferenceProvider::Cpu;
  std::string name = "CPU heuristic";
  bool accelerated = false;
  std::string detail;
};

class InferenceBackend {
public:
  virtual ~InferenceBackend() = default;

  [[nodiscard]] virtual BackendStatus status() const = 0;

  [[nodiscard]] virtual EnhancementProfile
  inferProfile(const AudioFeatures &features,
               const EnhancementProfile &userProfile) const = 0;

  [[nodiscard]] static std::unique_ptr<InferenceBackend> createPreferred();
};

[[nodiscard]] const char *toString(InferenceProvider provider) noexcept;

} // namespace npu_audio
