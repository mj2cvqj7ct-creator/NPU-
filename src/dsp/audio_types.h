#pragma once

#include <algorithm>
#include <vector>

namespace sxnae::dsp {

struct StereoFrame {
    float left = 0.0F;
    float right = 0.0F;
};

using AudioBlock = std::vector<StereoFrame>;

struct AudioFormat {
    int sample_rate_hz = 48000;
    int channels = 2;
};

inline float clampSample(float value, float ceiling = 1.0F) {
    return std::max(-ceiling, std::min(ceiling, value));
}

inline bool isStereoFloat48k(const AudioFormat& format) {
    return format.sample_rate_hz == 48000 && format.channels == 2;
}

}  // namespace sxnae::dsp
