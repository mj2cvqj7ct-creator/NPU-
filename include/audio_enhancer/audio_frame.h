#pragma once

#include <cstddef>
#include <vector>

namespace audio_enhancer {

struct AudioBuffer {
    int sample_rate = 48000;
    int channels = 2;
    std::vector<float> samples;

    [[nodiscard]] std::size_t frame_count() const {
        if (channels <= 0) {
            return 0;
        }
        return samples.size() / static_cast<std::size_t>(channels);
    }
};

}  // namespace audio_enhancer
