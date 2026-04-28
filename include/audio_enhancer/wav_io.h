#pragma once

#include "audio_enhancer/audio_frame.h"

#include <cstdint>
#include <filesystem>

namespace audio_enhancer {

struct WavFile {
    AudioBuffer buffer;
    std::uint32_t sample_rate = 0;
    std::uint16_t channels = 0;
};

WavFile read_wav(const std::filesystem::path& path);
void write_wav(const std::filesystem::path& path, const WavFile& wav);

}  // namespace audio_enhancer
