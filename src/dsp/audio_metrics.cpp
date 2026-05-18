#include "dsp/audio_metrics.h"

#include <algorithm>
#include <cmath>

#include "dsp/audio_math.h"

namespace sxnae::dsp {

float blockRmsLinear(const AudioBlock& block) {
    if (block.empty()) {
        return 0.0F;
    }

    double sum_squares = 0.0;
    for (const StereoFrame& frame : block) {
        sum_squares += static_cast<double>(frame.left) * frame.left;
        sum_squares += static_cast<double>(frame.right) * frame.right;
    }

    return static_cast<float>(std::sqrt(sum_squares / (block.size() * 2.0)));
}

float blockPeakLinear(const AudioBlock& block) {
    float peak = 0.0F;
    for (const StereoFrame& frame : block) {
        peak = std::max(peak, std::fabs(frame.left));
        peak = std::max(peak, std::fabs(frame.right));
    }
    return peak;
}

AudioMetrics analyzeBlock(const AudioBlock& block, float clip_threshold) {
    AudioMetrics metrics;
    const float rms = blockRmsLinear(block);
    const float peak = blockPeakLinear(block);

    metrics.rms_dbfs = linearToDb(rms);
    metrics.peak_dbfs = linearToDb(peak);
    metrics.crest_factor_db = metrics.peak_dbfs - metrics.rms_dbfs;

    if (block.empty()) {
        return metrics;
    }

    double diff_energy = 0.0;
    double signal_energy = 0.0;
    double left_right = 0.0;
    double left_energy = 0.0;
    double right_energy = 0.0;
    StereoFrame previous = block.front();

    for (const StereoFrame& frame : block) {
        if (std::fabs(frame.left) >= clip_threshold) {
            ++metrics.clipping_samples;
        }
        if (std::fabs(frame.right) >= clip_threshold) {
            ++metrics.clipping_samples;
        }

        const float left_diff = frame.left - previous.left;
        const float right_diff = frame.right - previous.right;
        diff_energy += static_cast<double>(left_diff) * left_diff;
        diff_energy += static_cast<double>(right_diff) * right_diff;
        signal_energy += static_cast<double>(frame.left) * frame.left;
        signal_energy += static_cast<double>(frame.right) * frame.right;
        left_right += static_cast<double>(frame.left) * frame.right;
        left_energy += static_cast<double>(frame.left) * frame.left;
        right_energy += static_cast<double>(frame.right) * frame.right;
        previous = frame;
    }

    if (signal_energy > 0.000000001) {
        metrics.transient_density = static_cast<float>(
            std::sqrt(diff_energy / signal_energy));
    }

    const double correlation_denominator = std::sqrt(left_energy * right_energy);
    if (correlation_denominator > 0.000000001) {
        metrics.stereo_correlation = static_cast<float>(
            left_right / correlation_denominator);
    }

    return metrics;
}

}  // namespace sxnae::dsp
