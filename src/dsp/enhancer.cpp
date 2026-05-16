#include "npu_audio/enhancer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace npu_audio {
namespace {

constexpr float kMinDb = -120.0F;
constexpr float kPi = 3.14159265358979323846F;

[[nodiscard]] float clamp(float value, float low, float high) noexcept {
  return std::max(low, std::min(value, high));
}

[[nodiscard]] float onePoleCoefficient(float cutoffHz,
                                       float sampleRate) noexcept {
  return std::exp(-2.0F * kPi * cutoffHz / sampleRate);
}

[[nodiscard]] float safeRatio(float numerator, float denominator) noexcept {
  return denominator > std::numeric_limits<float>::epsilon()
             ? numerator / denominator
             : 0.0F;
}

[[nodiscard]] std::string normalizeServiceName(std::string_view serviceName) {
  std::string normalized;
  normalized.reserve(serviceName.size());
  for (const char value : serviceName) {
    const auto character = static_cast<unsigned char>(value);
    if (std::isspace(character) != 0 || character == '-' || character == '_') {
      continue;
    }
    normalized.push_back(static_cast<char>(std::tolower(character)));
  }
  return normalized;
}

} // namespace

float linearToDb(float value) noexcept {
  if (value <= 0.000001F) {
    return kMinDb;
  }
  return 20.0F * std::log10(value);
}

float dbToLinear(float valueDb) noexcept {
  return std::pow(10.0F, valueDb / 20.0F);
}

EnhancementProfile profileForService(std::string_view serviceName) {
  EnhancementProfile profile{};
  const std::string normalized = normalizeServiceName(serviceName);
  if (normalized.empty() || normalized == "generic" || normalized == "auto") {
    return profile;
  }

  if (normalized == "spotify") {
    profile.targetLoudnessDb = -17.5F;
    profile.maxPositiveGainDb = 7.5F;
    profile.bassEnhancement = 0.04F;
    profile.clarityEnhancement = 0.07F;
    profile.stereoWidth = 1.03F;
    profile.compressorThresholdDb = -14.0F;
    profile.compressorRatio = 1.35F;
    profile.limiterCeilingDb = -1.2F;
    return profile;
  }

  if (normalized == "apple" || normalized == "applemusic" ||
      normalized == "applelossless") {
    profile.targetLoudnessDb = -18.5F;
    profile.maxPositiveGainDb = 6.5F;
    profile.bassEnhancement = 0.02F;
    profile.clarityEnhancement = 0.04F;
    profile.stereoWidth = 1.0F;
    profile.compressorThresholdDb = -11.0F;
    profile.compressorRatio = 1.25F;
    profile.limiterCeilingDb = -1.0F;
    return profile;
  }

  if (normalized == "youtube" || normalized == "youtubemusic" ||
      normalized == "ytmusic") {
    profile.targetLoudnessDb = -17.0F;
    profile.maxPositiveGainDb = 8.0F;
    profile.bassEnhancement = 0.03F;
    profile.clarityEnhancement = 0.10F;
    profile.stereoWidth = 1.02F;
    profile.compressorThresholdDb = -15.0F;
    profile.compressorRatio = 1.55F;
    profile.limiterCeilingDb = -1.3F;
    return profile;
  }

  throw std::invalid_argument("unknown service profile: " +
                              std::string(serviceName));
}

AudioFeatures analyzeAudio(const AudioBuffer &buffer) {
  AudioFeatures features{};
  if (buffer.empty()) {
    return features;
  }

  double sumSquares = 0.0;
  float peak = 0.0F;
  for (const float sample : buffer.samples()) {
    sumSquares += static_cast<double>(sample) * static_cast<double>(sample);
    peak = std::max(peak, std::abs(sample));
  }

  const float rms =
      std::sqrt(static_cast<float>(sumSquares / buffer.sampleCount()));
  features.rmsDb = linearToDb(rms);
  features.peakDb = linearToDb(peak);
  features.crestFactorDb = features.peakDb - features.rmsDb;
  features.silent = peak <= 0.000001F;

  const float lowAlpha =
      onePoleCoefficient(220.0F, static_cast<float>(buffer.sampleRate()));
  const float highAlpha =
      onePoleCoefficient(4200.0F, static_cast<float>(buffer.sampleRate()));

  std::vector<float> lowState(buffer.channels(), 0.0F);
  std::vector<float> highLowpassState(buffer.channels(), 0.0F);
  double lowEnergy = 0.0;
  double midEnergy = 0.0;
  double highEnergy = 0.0;

  double leftRight = 0.0;
  double leftEnergy = 0.0;
  double rightEnergy = 0.0;

  for (std::size_t frame = 0; frame < buffer.frameCount(); ++frame) {
    for (std::uint16_t channel = 0; channel < buffer.channels(); ++channel) {
      const float sample = buffer.sample(frame, channel);
      lowState[channel] =
          (1.0F - lowAlpha) * sample + lowAlpha * lowState[channel];
      highLowpassState[channel] = (1.0F - highAlpha) * sample +
                                  highAlpha * highLowpassState[channel];

      const float low = lowState[channel];
      const float high = sample - highLowpassState[channel];
      const float mid = sample - low - high;

      lowEnergy += static_cast<double>(low) * low;
      midEnergy += static_cast<double>(mid) * mid;
      highEnergy += static_cast<double>(high) * high;
    }

    if (buffer.channels() >= 2) {
      const float left = buffer.sample(frame, 0);
      const float right = buffer.sample(frame, 1);
      leftRight += static_cast<double>(left) * right;
      leftEnergy += static_cast<double>(left) * left;
      rightEnergy += static_cast<double>(right) * right;
    }
  }

  const float totalBandEnergy =
      static_cast<float>(lowEnergy + midEnergy + highEnergy);
  features.lowEnergyRatio =
      safeRatio(static_cast<float>(lowEnergy), totalBandEnergy);
  features.midEnergyRatio =
      safeRatio(static_cast<float>(midEnergy), totalBandEnergy);
  features.highEnergyRatio =
      safeRatio(static_cast<float>(highEnergy), totalBandEnergy);

  if (buffer.channels() >= 2 && leftEnergy > 0.0 && rightEnergy > 0.0) {
    features.stereoCorrelation =
        static_cast<float>(leftRight / std::sqrt(leftEnergy * rightEnergy));
    features.stereoCorrelation =
        clamp(features.stereoCorrelation, -1.0F, 1.0F);
  }

  return features;
}

AudioEnhancer::AudioEnhancer(std::unique_ptr<InferenceBackend> backend)
    : backend_(std::move(backend)) {
  if (!backend_) {
    throw std::invalid_argument("inference backend must not be null");
  }
}

EnhancementReport AudioEnhancer::process(AudioBuffer &buffer,
                                         const EnhancementProfile &userProfile) {
  EnhancementReport report{};
  report.inputFeatures = analyzeAudio(buffer);
  report.inputPeakDb = report.inputFeatures.peakDb;

  const BackendStatus backendStatus = backend_->status();
  report.backendName = backendStatus.name;
  report.acceleratedBackend = backendStatus.accelerated;
  report.appliedProfile =
      backend_->inferProfile(report.inputFeatures, userProfile);

  if (buffer.empty() || report.inputFeatures.silent) {
    report.outputPeakDb = report.inputPeakDb;
    return report;
  }

  const float desiredGainDb =
      clamp(report.appliedProfile.targetLoudnessDb - report.inputFeatures.rmsDb,
            -18.0F, report.appliedProfile.maxPositiveGainDb);
  report.loudnessGainDb = desiredGainDb;
  const float loudnessGain = dbToLinear(desiredGainDb);

  const float lowAlpha =
      onePoleCoefficient(180.0F, static_cast<float>(buffer.sampleRate()));
  const float highAlpha =
      onePoleCoefficient(3600.0F, static_cast<float>(buffer.sampleRate()));
  const float bassAmount =
      clamp(report.appliedProfile.bassEnhancement, -0.45F, 0.45F);
  const float clarityAmount =
      clamp(report.appliedProfile.clarityEnhancement, -0.45F, 0.45F);
  const float sideGain =
      clamp(report.appliedProfile.stereoWidth, 0.65F, 1.35F);
  const float limiterCeiling = dbToLinear(report.appliedProfile.limiterCeilingDb);

  std::vector<float> lowState(buffer.channels(), 0.0F);
  std::vector<float> highLowpassState(buffer.channels(), 0.0F);
  float compressorEnvelope = 0.0F;
  const float attack = std::exp(-1.0F /
                                (0.005F * static_cast<float>(buffer.sampleRate())));
  const float release =
      std::exp(-1.0F / (0.080F * static_cast<float>(buffer.sampleRate())));

  for (std::size_t frame = 0; frame < buffer.frameCount(); ++frame) {
    for (std::uint16_t channel = 0; channel < buffer.channels(); ++channel) {
      const float sample = buffer.sample(frame, channel) * loudnessGain;
      lowState[channel] =
          (1.0F - lowAlpha) * sample + lowAlpha * lowState[channel];
      highLowpassState[channel] = (1.0F - highAlpha) * sample +
                                  highAlpha * highLowpassState[channel];
      const float low = lowState[channel];
      const float high = sample - highLowpassState[channel];
      buffer.setSample(frame, channel,
                       sample + (bassAmount * low) + (clarityAmount * high));
    }

    if (buffer.channels() >= 2) {
      const float left = buffer.sample(frame, 0);
      const float right = buffer.sample(frame, 1);
      const float mid = 0.5F * (left + right);
      const float side = 0.5F * (left - right) * sideGain;
      buffer.setSample(frame, 0, mid + side);
      buffer.setSample(frame, 1, mid - side);
    }

    float framePeak = 0.0F;
    for (std::uint16_t channel = 0; channel < buffer.channels(); ++channel) {
      framePeak = std::max(framePeak, std::abs(buffer.sample(frame, channel)));
    }

    const float envelopeCoefficient =
        framePeak > compressorEnvelope ? attack : release;
    compressorEnvelope =
        envelopeCoefficient * compressorEnvelope +
        (1.0F - envelopeCoefficient) * framePeak;

    const float envelopeDb = linearToDb(compressorEnvelope);
    float compressorGain = 1.0F;
    if (envelopeDb > report.appliedProfile.compressorThresholdDb) {
      const float compressedDb =
          report.appliedProfile.compressorThresholdDb +
          ((envelopeDb - report.appliedProfile.compressorThresholdDb) /
           std::max(1.0F, report.appliedProfile.compressorRatio));
      compressorGain = dbToLinear(compressedDb - envelopeDb);
    }

    framePeak = 0.0F;
    for (std::uint16_t channel = 0; channel < buffer.channels(); ++channel) {
      const float compressed = buffer.sample(frame, channel) * compressorGain;
      buffer.setSample(frame, channel, compressed);
      framePeak = std::max(framePeak, std::abs(compressed));
    }

    const float limiterGain =
        framePeak > limiterCeiling ? limiterCeiling / framePeak : 1.0F;
    for (std::uint16_t channel = 0; channel < buffer.channels(); ++channel) {
      const float limited =
          clamp(buffer.sample(frame, channel) * limiterGain, -limiterCeiling,
                limiterCeiling);
      buffer.setSample(frame, channel, limited);
    }
  }

  report.outputPeakDb = analyzeAudio(buffer).peakDb;
  return report;
}

} // namespace npu_audio
