#include "dsp/audio_frame.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace snae::dsp {

AudioFrame::AudioFrame(int sample_rate, std::vector<StereoSample> samples)
    : sample_rate_(sample_rate), samples_(std::move(samples)) {
    if (sample_rate_ <= 0) {
        throw std::invalid_argument("sample rate must be positive");
    }
}

AudioFrame::AudioFrame(std::size_t samples_per_channel, int sample_rate)
    : AudioFrame(sample_rate, std::vector<StereoSample>(samples_per_channel)) {}

int AudioFrame::sample_rate() const noexcept {
    return sample_rate_;
}

std::size_t AudioFrame::size() const noexcept {
    return samples_.size();
}

bool AudioFrame::empty() const noexcept {
    return samples_.empty();
}

const std::vector<StereoSample>& AudioFrame::samples() const noexcept {
    return samples_;
}

std::vector<StereoSample>& AudioFrame::samples() noexcept {
    return samples_;
}

void AudioFrame::apply_gain(float gain) {
    for (auto& sample : samples_) {
        sample.left *= gain;
        sample.right *= gain;
    }
}

float measurePeak(const AudioFrame& frame) {
    float peak = 0.0F;
    for (const auto& sample : frame.samples()) {
        peak = std::max({peak, std::abs(sample.left), std::abs(sample.right)});
    }
    return peak;
}

float measureRms(const AudioFrame& frame) {
    if (frame.empty()) {
        return 0.0F;
    }

    double sum_squares = 0.0;
    for (const auto& sample : frame.samples()) {
        sum_squares += static_cast<double>(sample.left) * sample.left;
        sum_squares += static_cast<double>(sample.right) * sample.right;
    }
    return static_cast<float>(std::sqrt(sum_squares / static_cast<double>(frame.size() * 2U)));
}

}  // namespace snae::dsp
