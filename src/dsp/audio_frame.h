#pragma once

#include <cstddef>
#include <vector>

namespace snae::dsp {

struct StereoSample {
    float left = 0.0F;
    float right = 0.0F;
};

class AudioFrame {
public:
    static constexpr int kDefaultSampleRate = 48000;

    AudioFrame(int sample_rate, std::vector<StereoSample> samples);
    AudioFrame(std::size_t samples_per_channel, int sample_rate = kDefaultSampleRate);

    [[nodiscard]] int sample_rate() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] const std::vector<StereoSample>& samples() const noexcept;
    [[nodiscard]] std::vector<StereoSample>& samples() noexcept;

    void apply_gain(float gain);

private:
    int sample_rate_ = kDefaultSampleRate;
    std::vector<StereoSample> samples_;
};

float measurePeak(const AudioFrame& frame);
float measureRms(const AudioFrame& frame);

}  // namespace snae::dsp
