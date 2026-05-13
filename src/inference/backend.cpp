#include "npu_audio/inference_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>

namespace npu_audio {
namespace {

[[nodiscard]] float clamp(float value, float low, float high) noexcept {
  return std::max(low, std::min(value, high));
}

[[nodiscard]] std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

[[nodiscard]] bool isArm64Build() noexcept {
#if defined(_M_ARM64) || defined(__aarch64__) || defined(__ARM64_ARCH_8__)
  return true;
#else
  return false;
#endif
}

[[nodiscard]] bool isWindowsBuild() noexcept {
#if defined(_WIN32)
  return true;
#else
  return false;
#endif
}

class HeuristicInferenceBackend final : public InferenceBackend {
public:
  explicit HeuristicInferenceBackend(BackendStatus status)
      : status_(std::move(status)) {}

  [[nodiscard]] BackendStatus status() const override { return status_; }

  [[nodiscard]] EnhancementProfile
  inferProfile(const AudioFeatures &features,
               const EnhancementProfile &userProfile) const override {
    EnhancementProfile profile = userProfile;

    if (features.silent) {
      profile.bassEnhancement = 0.0F;
      profile.clarityEnhancement = 0.0F;
      profile.stereoWidth = 1.0F;
      return profile;
    }

    const float lowCorrection = (0.24F - features.lowEnergyRatio) * 0.70F;
    const float highCorrection = (0.26F - features.highEnergyRatio) * 0.85F;
    const float overCompressedBoost = features.crestFactorDb < 8.0F ? 0.08F : 0.0F;

    profile.bassEnhancement =
        clamp(profile.bassEnhancement + lowCorrection, -0.25F, 0.30F);
    profile.clarityEnhancement =
        clamp(profile.clarityEnhancement + highCorrection + overCompressedBoost,
              -0.20F, 0.35F);

    if (features.stereoCorrelation > 0.90F) {
      profile.stereoWidth = clamp(profile.stereoWidth * 1.08F, 0.80F, 1.18F);
    } else if (features.stereoCorrelation < 0.10F) {
      profile.stereoWidth = clamp(profile.stereoWidth * 0.94F, 0.80F, 1.18F);
    } else {
      profile.stereoWidth = clamp(profile.stereoWidth, 0.80F, 1.18F);
    }

    profile.compressorThresholdDb =
        features.crestFactorDb < 7.0F ? -16.0F : profile.compressorThresholdDb;
    profile.compressorRatio =
        features.peakDb > -3.0F ? std::max(profile.compressorRatio, 1.65F)
                                : profile.compressorRatio;
    profile.maxPositiveGainDb = clamp(profile.maxPositiveGainDb, 0.0F, 9.0F);
    profile.limiterCeilingDb = clamp(profile.limiterCeilingDb, -3.0F, -0.5F);

    return profile;
  }

private:
  BackendStatus status_;
};

[[nodiscard]] BackendStatus makeStatus(InferenceProvider provider,
                                       bool accelerated,
                                       std::string detail) {
  BackendStatus status;
  status.provider = provider;
  status.name = std::string(toString(provider));
  status.accelerated = accelerated;
  status.detail = std::move(detail);
  return status;
}

} // namespace

const char *toString(InferenceProvider provider) noexcept {
  switch (provider) {
  case InferenceProvider::QnnNpu:
    return "Qualcomm QNN NPU";
  case InferenceProvider::DirectMl:
    return "DirectML";
  case InferenceProvider::Cpu:
    return "CPU heuristic";
  }
  return "Unknown";
}

std::unique_ptr<InferenceBackend> InferenceBackend::createPreferred() {
  const char *requestedBackend = std::getenv("NPU_AUDIO_BACKEND");
  const std::string requested =
      requestedBackend == nullptr ? std::string{} : lower(requestedBackend);

  if (requested == "qnn" || requested == "npu" || requested == "snapdragon") {
#if defined(NPU_AUDIO_ENABLE_QNN)
    return std::make_unique<HeuristicInferenceBackend>(makeStatus(
        InferenceProvider::QnnNpu, true,
        "QNN backend requested; build is prepared for Snapdragon X NPU model "
        "execution."));
#else
    return std::make_unique<HeuristicInferenceBackend>(makeStatus(
        InferenceProvider::Cpu, false,
        "QNN backend requested, but this build was compiled without "
        "NPU_AUDIO_ENABLE_QNN."));
#endif
  }

  if (requested == "directml" || requested == "dml") {
    return std::make_unique<HeuristicInferenceBackend>(makeStatus(
        isWindowsBuild() ? InferenceProvider::DirectMl : InferenceProvider::Cpu,
        isWindowsBuild(),
        isWindowsBuild() ? "DirectML backend requested on Windows."
                         : "DirectML requested on a non-Windows build; using CPU."));
  }

#if defined(NPU_AUDIO_ENABLE_QNN)
  if (isWindowsBuild() && isArm64Build()) {
    return std::make_unique<HeuristicInferenceBackend>(makeStatus(
        InferenceProvider::QnnNpu, true,
        "ARM64 Windows build prepared for Snapdragon X QNN execution."));
  }
#endif

  return std::make_unique<HeuristicInferenceBackend>(makeStatus(
      InferenceProvider::Cpu, false,
      isArm64Build() ? "ARM64 build without QNN SDK linkage; using CPU."
                     : "Portable build; using CPU."));
}

} // namespace npu_audio
