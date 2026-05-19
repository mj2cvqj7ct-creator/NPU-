#include "npu_audio/realtime_enhancer.hpp"

#include "npu_audio/enhancer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace npu_audio {
namespace {

constexpr float kPi = 3.14159265358979323846F;

[[nodiscard]] float clamp(float value, float low, float high) noexcept {
  return std::max(low, std::min(value, high));
}

[[nodiscard]] float onePoleCoefficient(float cutoffHz,
                                       float sampleRate) noexcept {
  return std::exp(-2.0F * kPi * cutoffHz / sampleRate);
}

[[nodiscard]] float smooth(float previous, float next, float amount) noexcept {
  return previous + ((next - previous) * amount);
}

[[nodiscard]] EnhancementProfile smoothProfile(const EnhancementProfile &previous,
                                               const EnhancementProfile &next,
                                               float amount) noexcept {
  EnhancementProfile profile;
  profile.targetLoudnessDb =
      smooth(previous.targetLoudnessDb, next.targetLoudnessDb, amount);
  profile.maxPositiveGainDb =
      smooth(previous.maxPositiveGainDb, next.maxPositiveGainDb, amount);
  profile.bassEnhancement =
      smooth(previous.bassEnhancement, next.bassEnhancement, amount);
  profile.clarityEnhancement =
      smooth(previous.clarityEnhancement, next.clarityEnhancement, amount);
  profile.stereoWidth = smooth(previous.stereoWidth, next.stereoWidth, amount);
  profile.compressorThresholdDb = smooth(previous.compressorThresholdDb,
                                         next.compressorThresholdDb, amount);
  profile.compressorRatio =
      smooth(previous.compressorRatio, next.compressorRatio, amount);
  profile.limiterCeilingDb =
      smooth(previous.limiterCeilingDb, next.limiterCeilingDb, amount);
  return profile;
}

[[nodiscard]] std::size_t computeFrameSize(std::uint32_t sampleRate,
                                           float frameDurationMs) {
  if (sampleRate == 0) {
    throw std::invalid_argument("realtime sample rate must be non-zero");
  }
  if (frameDurationMs < 10.0F || frameDurationMs > 20.0F) {
    throw std::invalid_argument(
        "realtime NPU frame duration must be between 10 and 20 ms");
  }

  const auto frames =
      static_cast<std::size_t>(std::lround(static_cast<double>(sampleRate) *
                                           frameDurationMs / 1000.0));
  return std::max<std::size_t>(frames, 1U);
}

} // namespace

RealtimeEnhancer::RealtimeEnhancer(RealtimeConfig config,
                                   std::unique_ptr<InferenceBackend> backend)
    : config_(std::move(config)), backend_(std::move(backend)),
      frameSize_(computeFrameSize(config_.sampleRate,
                                  config_.frameDurationMs)),
      lowState_(config_.channels, 0.0F),
      highLowpassState_(config_.channels, 0.0F) {
  if (config_.channels == 0) {
    throw std::invalid_argument("realtime channel count must be non-zero");
  }
  if (!backend_) {
    throw std::invalid_argument("inference backend must not be null");
  }
}

void RealtimeEnhancer::reset() {
  std::fill(lowState_.begin(), lowState_.end(), 0.0F);
  std::fill(highLowpassState_.begin(), highLowpassState_.end(), 0.0F);
  compressorEnvelope_ = 0.0F;
  smoothedLoudnessGainDb_ = 0.0F;
  hasSmoothedLoudness_ = false;
  hasSmoothedProfile_ = false;
}

RealtimeReport RealtimeEnhancer::processFrame(AudioBuffer &frame) {
  if (frame.sampleRate() != config_.sampleRate ||
      frame.channels() != config_.channels) {
    throw std::invalid_argument(
        "realtime frame format does not match enhancer configuration");
  }
  if (frame.frameCount() > frameSize_) {
    throw std::invalid_argument(
        "realtime frame exceeds configured NPU frame budget");
  }

  RealtimeReport realtimeReport;
  realtimeReport.processedFrames = frame.frameCount();
  realtimeReport.frameDurationMs =
      1000.0F * static_cast<float>(frame.frameCount()) /
      static_cast<float>(config_.sampleRate);

  EnhancementReport &report = realtimeReport.enhancement;
  report.inputFeatures = analyzeAudio(frame);
  report.inputPeakDb = report.inputFeatures.peakDb;

  const BackendStatus backendStatus = backend_->status();
  report.backendName = backendStatus.name;
  report.acceleratedBackend = backendStatus.accelerated;

  const EnhancementProfile inferredProfile =
      backend_->inferProfile(report.inputFeatures, config_.userProfile);
  if (!hasSmoothedProfile_) {
    smoothedProfile_ = inferredProfile;
    hasSmoothedProfile_ = true;
  } else {
    smoothedProfile_ = smoothProfile(smoothedProfile_, inferredProfile, 0.30F);
  }
  report.appliedProfile = smoothedProfile_;

  if (frame.empty() || report.inputFeatures.silent) {
    report.outputPeakDb = report.inputPeakDb;
    return realtimeReport;
  }

  const float desiredGainDb =
      clamp(report.appliedProfile.targetLoudnessDb - report.inputFeatures.rmsDb,
            -18.0F, report.appliedProfile.maxPositiveGainDb);
  if (!hasSmoothedLoudness_) {
    smoothedLoudnessGainDb_ = desiredGainDb;
    hasSmoothedLoudness_ = true;
  } else {
    smoothedLoudnessGainDb_ = smooth(smoothedLoudnessGainDb_, desiredGainDb, 0.20F);
  }

  report.loudnessGainDb = smoothedLoudnessGainDb_;
  const float loudnessGain = dbToLinear(smoothedLoudnessGainDb_);
  const float lowAlpha =
      onePoleCoefficient(180.0F, static_cast<float>(config_.sampleRate));
  const float highAlpha =
      onePoleCoefficient(3600.0F, static_cast<float>(config_.sampleRate));
  const float bassAmount =
      clamp(report.appliedProfile.bassEnhancement, -0.45F, 0.45F);
  const float clarityAmount =
      clamp(report.appliedProfile.clarityEnhancement, -0.45F, 0.45F);
  const float sideGain = clamp(report.appliedProfile.stereoWidth, 0.65F, 1.35F);
  const float limiterCeiling = dbToLinear(report.appliedProfile.limiterCeilingDb);
  const float attack = std::exp(-1.0F /
                                (0.005F * static_cast<float>(config_.sampleRate)));
  const float release =
      std::exp(-1.0F / (0.080F * static_cast<float>(config_.sampleRate)));

  for (std::size_t frameIndex = 0; frameIndex < frame.frameCount();
       ++frameIndex) {
    for (std::uint16_t channel = 0; channel < frame.channels(); ++channel) {
      const float sample = frame.sample(frameIndex, channel) * loudnessGain;
      lowState_[channel] =
          (1.0F - lowAlpha) * sample + lowAlpha * lowState_[channel];
      highLowpassState_[channel] =
          (1.0F - highAlpha) * sample + highAlpha * highLowpassState_[channel];
      const float low = lowState_[channel];
      const float high = sample - highLowpassState_[channel];
      frame.setSample(frameIndex, channel,
                      sample + (bassAmount * low) + (clarityAmount * high));
    }

    if (frame.channels() >= 2) {
      const float left = frame.sample(frameIndex, 0);
      const float right = frame.sample(frameIndex, 1);
      const float mid = 0.5F * (left + right);
      const float side = 0.5F * (left - right) * sideGain;
      frame.setSample(frameIndex, 0, mid + side);
      frame.setSample(frameIndex, 1, mid - side);
    }

    float framePeak = 0.0F;
    for (std::uint16_t channel = 0; channel < frame.channels(); ++channel) {
      framePeak = std::max(framePeak, std::abs(frame.sample(frameIndex, channel)));
    }

    const float envelopeCoefficient =
        framePeak > compressorEnvelope_ ? attack : release;
    compressorEnvelope_ = envelopeCoefficient * compressorEnvelope_ +
                          (1.0F - envelopeCoefficient) * framePeak;

    const float envelopeDb = linearToDb(compressorEnvelope_);
    float compressorGain = 1.0F;
    if (envelopeDb > report.appliedProfile.compressorThresholdDb) {
      const float compressedDb =
          report.appliedProfile.compressorThresholdDb +
          ((envelopeDb - report.appliedProfile.compressorThresholdDb) /
           std::max(1.0F, report.appliedProfile.compressorRatio));
      compressorGain = dbToLinear(compressedDb - envelopeDb);
    }

    framePeak = 0.0F;
    for (std::uint16_t channel = 0; channel < frame.channels(); ++channel) {
      const float compressed = frame.sample(frameIndex, channel) * compressorGain;
      frame.setSample(frameIndex, channel, compressed);
      framePeak = std::max(framePeak, std::abs(compressed));
    }

    const float limiterGain =
        framePeak > limiterCeiling ? limiterCeiling / framePeak : 1.0F;
    for (std::uint16_t channel = 0; channel < frame.channels(); ++channel) {
      const float limited =
          clamp(frame.sample(frameIndex, channel) * limiterGain, -limiterCeiling,
                limiterCeiling);
      frame.setSample(frameIndex, channel, limited);
    }
  }

  report.outputPeakDb = analyzeAudio(frame).peakDb;
  return realtimeReport;
}

} // namespace npu_audio
