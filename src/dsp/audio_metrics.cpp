#include "dsp/audio_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace snae::dsp {
namespace {
constexpr float kLogFloor = 1.0e-12F;
}

AudioMetrics MetricsAnalyzer::analyze(const AudioFrame& frame) const {
    AudioMetrics metrics{};
    if (frame.empty()) {
        return metrics;
    }

    double sum_squares = 0.0;
    double left_squares = 0.0;
    double right_squares = 0.0;
    double cross = 0.0;
    double low_energy = 0.0;
    double vocal_energy = 0.0;
    float peak = 0.0F;
    bool clipped = false;

    StereoSample previous{};
    bool has_previous = false;
    for (const auto& sample : frame.samples()) {
        peak = std::max({peak, std::abs(sample.left), std::abs(sample.right)});
        clipped = clipped || std::abs(sample.left) >= 0.999F || std::abs(sample.right) >= 0.999F;

        sum_squares += static_cast<double>(sample.left) * sample.left;
        sum_squares += static_cast<double>(sample.right) * sample.right;
        left_squares += static_cast<double>(sample.left) * sample.left;
        right_squares += static_cast<double>(sample.right) * sample.right;
        cross += static_cast<double>(sample.left) * sample.right;

        if (has_previous) {
            const float low_left = 0.92F * previous.left + 0.08F * sample.left;
            const float low_right = 0.92F * previous.right + 0.08F * sample.right;
            low_energy += static_cast<double>(low_left) * low_left + static_cast<double>(low_right) * low_right;

            const float vocal_left = sample.left - low_left;
            const float vocal_right = sample.right - low_right;
            vocal_energy += static_cast<double>(vocal_left) * vocal_left + static_cast<double>(vocal_right) * vocal_right;
        }
        previous = sample;
        has_previous = true;
    }

    const auto count = static_cast<double>(frame.size() * 2U);
    const float rms = static_cast<float>(std::sqrt(sum_squares / count));
    metrics.rms_dbfs = linearToDb(rms);
    metrics.peak_dbfs = linearToDb(peak);
    metrics.crest_factor_db = metrics.peak_dbfs - metrics.rms_dbfs;
    metrics.low_band_energy = static_cast<float>(low_energy / count);
    metrics.vocal_band_energy = static_cast<float>(vocal_energy / count);
    metrics.clipping_detected = clipped;

    const double denominator = std::sqrt(left_squares * right_squares);
    metrics.stereo_correlation = denominator <= std::numeric_limits<double>::epsilon()
                                      ? 0.0F
                                      : static_cast<float>(std::clamp(cross / denominator, -1.0, 1.0));
    return metrics;
}

float linearToDb(float value) {
    return 20.0F * std::log10(std::max(value, kLogFloor));
}

float dbToLinear(float db) {
    return std::pow(10.0F, db / 20.0F);
}

}  // namespace snae::dsp
