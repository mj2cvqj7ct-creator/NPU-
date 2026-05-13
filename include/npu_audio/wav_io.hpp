#pragma once

#include "npu_audio/audio_buffer.hpp"

#include <filesystem>

namespace npu_audio {

[[nodiscard]] AudioBuffer readWavFile(const std::filesystem::path &path);
void writeWavFile16(const std::filesystem::path &path,
                    const AudioBuffer &buffer);

} // namespace npu_audio
