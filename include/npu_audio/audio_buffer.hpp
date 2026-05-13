#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace npu_audio {

class AudioBuffer {
public:
  AudioBuffer() = default;

  AudioBuffer(std::uint32_t sampleRate, std::uint16_t channels,
              std::vector<float> interleavedSamples)
      : sampleRate_(sampleRate), channels_(channels),
        samples_(std::move(interleavedSamples)) {
    if (sampleRate_ == 0) {
      throw std::invalid_argument("sample rate must be non-zero");
    }
    if (channels_ == 0) {
      throw std::invalid_argument("channel count must be non-zero");
    }
    if (samples_.size() % channels_ != 0) {
      throw std::invalid_argument("interleaved sample count must align to channels");
    }
  }

  [[nodiscard]] std::uint32_t sampleRate() const noexcept { return sampleRate_; }
  [[nodiscard]] std::uint16_t channels() const noexcept { return channels_; }
  [[nodiscard]] std::size_t sampleCount() const noexcept { return samples_.size(); }
  [[nodiscard]] std::size_t frameCount() const noexcept {
    return channels_ == 0 ? 0 : samples_.size() / channels_;
  }

  [[nodiscard]] bool empty() const noexcept { return samples_.empty(); }

  [[nodiscard]] const std::vector<float> &samples() const noexcept {
    return samples_;
  }

  [[nodiscard]] std::vector<float> &samples() noexcept { return samples_; }

  [[nodiscard]] float sample(std::size_t frame, std::uint16_t channel) const {
    return samples_.at(frame * channels_ + channel);
  }

  void setSample(std::size_t frame, std::uint16_t channel, float value) {
    samples_.at(frame * channels_ + channel) = value;
  }

private:
  std::uint32_t sampleRate_ = 48000;
  std::uint16_t channels_ = 2;
  std::vector<float> samples_;
};

} // namespace npu_audio
