#include "audio_enhancer/wav_io.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace audio_enhancer {
namespace {

std::uint16_t read_u16(std::istream& input) {
    std::array<unsigned char, 2> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
}

std::uint32_t read_u32(std::istream& input) {
    std::array<unsigned char, 4> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    return static_cast<std::uint32_t>(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) |
                                      (bytes[3] << 24));
}

void write_u16(std::ostream& output, std::uint16_t value) {
    const std::array<unsigned char, 2> bytes{
        static_cast<unsigned char>(value & 0xFFU),
        static_cast<unsigned char>((value >> 8U) & 0xFFU),
    };
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void write_u32(std::ostream& output, std::uint32_t value) {
    const std::array<unsigned char, 4> bytes{
        static_cast<unsigned char>(value & 0xFFU),
        static_cast<unsigned char>((value >> 8U) & 0xFFU),
        static_cast<unsigned char>((value >> 16U) & 0xFFU),
        static_cast<unsigned char>((value >> 24U) & 0xFFU),
    };
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

bool read_tag(std::istream& input, const char* tag) {
    std::array<char, 4> bytes{};
    input.read(bytes.data(), bytes.size());
    return input && std::memcmp(bytes.data(), tag, bytes.size()) == 0;
}

void write_tag(std::ostream& output, const char* tag) {
    output.write(tag, 4);
}

float int16_to_float(std::int16_t sample) {
    return std::max(-1.0F, static_cast<float>(sample) / 32768.0F);
}

std::int16_t float_to_int16(float sample) {
    const float clipped = std::clamp(sample, -1.0F, 1.0F);
    return static_cast<std::int16_t>(std::lrint(clipped * 32767.0F));
}

}  // namespace

WavFile read_wav(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open WAV file: " + path.string());
    }

    if (!read_tag(input, "RIFF")) {
        throw std::runtime_error("WAV file is missing RIFF header");
    }
    (void)read_u32(input);
    if (!read_tag(input, "WAVE")) {
        throw std::runtime_error("WAV file is missing WAVE header");
    }

    std::uint16_t audio_format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits_per_sample = 0;
    std::vector<char> data;

    while (input && (!channels || data.empty())) {
        std::array<char, 4> chunk_id{};
        input.read(chunk_id.data(), chunk_id.size());
        if (!input) {
            break;
        }

        const std::uint32_t chunk_size = read_u32(input);
        const std::string chunk(chunk_id.data(), chunk_id.size());
        if (chunk == "fmt ") {
            audio_format = read_u16(input);
            channels = read_u16(input);
            sample_rate = read_u32(input);
            (void)read_u32(input);
            (void)read_u16(input);
            bits_per_sample = read_u16(input);

            if (chunk_size > 16) {
                input.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }
        } else if (chunk == "data") {
            data.resize(chunk_size);
            input.read(data.data(), static_cast<std::streamsize>(data.size()));
        } else {
            input.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }

        if ((chunk_size % 2U) == 1U) {
            input.seekg(1, std::ios::cur);
        }
    }

    if (audio_format != 1 || bits_per_sample != 16) {
        throw std::runtime_error("only PCM 16-bit WAV files are supported by this prototype");
    }
    if (channels == 0 || sample_rate == 0 || data.empty()) {
        throw std::runtime_error("WAV file is missing required format or data chunk");
    }

    WavFile wav;
    wav.sample_rate = sample_rate;
    wav.channels = channels;
    wav.buffer.sample_rate = static_cast<int>(sample_rate);
    wav.buffer.channels = static_cast<int>(channels);
    wav.buffer.samples.reserve(data.size() / sizeof(std::int16_t));

    for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
        const auto lo = static_cast<unsigned char>(data[i]);
        const auto hi = static_cast<unsigned char>(data[i + 1]);
        const auto raw = static_cast<std::int16_t>(lo | (hi << 8));
        wav.buffer.samples.push_back(int16_to_float(raw));
    }

    return wav;
}

void write_wav(const std::filesystem::path& path, const WavFile& wav) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to create WAV file: " + path.string());
    }

    const auto sample_count = static_cast<std::uint32_t>(wav.buffer.samples.size());
    const std::uint32_t data_size = sample_count * sizeof(std::int16_t);
    const std::uint32_t riff_size = 36U + data_size;
    const std::uint16_t channels = wav.channels != 0 ? wav.channels
                                                     : static_cast<std::uint16_t>(wav.buffer.channels);
    const std::uint32_t sample_rate = wav.sample_rate != 0 ? wav.sample_rate
                                                           : static_cast<std::uint32_t>(wav.buffer.sample_rate);
    const std::uint16_t block_align = static_cast<std::uint16_t>(channels * sizeof(std::int16_t));
    const std::uint32_t byte_rate = sample_rate * block_align;

    write_tag(output, "RIFF");
    write_u32(output, riff_size);
    write_tag(output, "WAVE");
    write_tag(output, "fmt ");
    write_u32(output, 16);
    write_u16(output, 1);
    write_u16(output, channels);
    write_u32(output, sample_rate);
    write_u32(output, byte_rate);
    write_u16(output, block_align);
    write_u16(output, 16);
    write_tag(output, "data");
    write_u32(output, data_size);

    for (float sample : wav.buffer.samples) {
        write_u16(output, static_cast<std::uint16_t>(float_to_int16(sample)));
    }
}

}  // namespace audio_enhancer
