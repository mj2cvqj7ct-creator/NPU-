#include "npu_audio/wav_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npu_audio {
namespace {

struct WavFormat {
  std::uint16_t audioFormat = 0;
  std::uint16_t channels = 0;
  std::uint32_t sampleRate = 0;
  std::uint16_t bitsPerSample = 0;
};

[[nodiscard]] std::uint16_t readU16(std::istream &stream) {
  std::array<unsigned char, 2> bytes{};
  stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!stream) {
    throw std::runtime_error("unexpected end of WAV file");
  }
  return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8U));
}

[[nodiscard]] std::uint32_t readU32(std::istream &stream) {
  std::array<unsigned char, 4> bytes{};
  stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!stream) {
    throw std::runtime_error("unexpected end of WAV file");
  }
  return static_cast<std::uint32_t>(bytes[0] | (bytes[1] << 8U) |
                                    (bytes[2] << 16U) | (bytes[3] << 24U));
}

void writeU16(std::ostream &stream, std::uint16_t value) {
  const std::array<unsigned char, 2> bytes{
      static_cast<unsigned char>(value & 0xFFU),
      static_cast<unsigned char>((value >> 8U) & 0xFFU)};
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

void writeU32(std::ostream &stream, std::uint32_t value) {
  const std::array<unsigned char, 4> bytes{
      static_cast<unsigned char>(value & 0xFFU),
      static_cast<unsigned char>((value >> 8U) & 0xFFU),
      static_cast<unsigned char>((value >> 16U) & 0xFFU),
      static_cast<unsigned char>((value >> 24U) & 0xFFU)};
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

[[nodiscard]] bool chunkIdEquals(const std::array<char, 4> &id,
                                 const char *expected) {
  return std::memcmp(id.data(), expected, id.size()) == 0;
}

[[nodiscard]] std::uint32_t byteAt(const std::vector<unsigned char> &data,
                                   std::size_t offset) noexcept {
  return static_cast<std::uint32_t>(data[offset]);
}

[[nodiscard]] float decodePcmSample(const std::vector<unsigned char> &data,
                                    std::size_t offset,
                                    std::uint16_t bitsPerSample) {
  if (bitsPerSample == 16) {
    const auto raw = static_cast<std::int16_t>(
        byteAt(data, offset) | (byteAt(data, offset + 1) << 8U));
    return std::max(-1.0F, static_cast<float>(raw) / 32768.0F);
  }

  if (bitsPerSample == 24) {
    std::int32_t raw = static_cast<std::int32_t>(
        byteAt(data, offset) | (byteAt(data, offset + 1) << 8U) |
        (byteAt(data, offset + 2) << 16U));
    if ((raw & 0x00800000) != 0) {
      raw |= static_cast<std::int32_t>(0xFF000000);
    }
    return std::max(-1.0F, static_cast<float>(raw) / 8388608.0F);
  }

  if (bitsPerSample == 32) {
    const auto raw = static_cast<std::int32_t>(
        byteAt(data, offset) | (byteAt(data, offset + 1) << 8U) |
        (byteAt(data, offset + 2) << 16U) |
        (byteAt(data, offset + 3) << 24U));
    return std::max(-1.0F, static_cast<float>(raw) / 2147483648.0F);
  }

  throw std::runtime_error("unsupported PCM bit depth");
}

[[nodiscard]] float decodeFloatSample(const std::vector<unsigned char> &data,
                                      std::size_t offset) {
  float value = 0.0F;
  std::uint32_t raw = static_cast<std::uint32_t>(
      byteAt(data, offset) | (byteAt(data, offset + 1) << 8U) |
      (byteAt(data, offset + 2) << 16U) |
      (byteAt(data, offset + 3) << 24U));
  std::memcpy(&value, &raw, sizeof(value));
  return std::clamp(value, -4.0F, 4.0F);
}

} // namespace

AudioBuffer readWavFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open WAV file: " + path.string());
  }

  std::array<char, 4> riff{};
  stream.read(riff.data(), riff.size());
  const std::uint32_t riffSize = readU32(stream);
  (void)riffSize;

  std::array<char, 4> wave{};
  stream.read(wave.data(), wave.size());
  if (!stream || !chunkIdEquals(riff, "RIFF") || !chunkIdEquals(wave, "WAVE")) {
    throw std::runtime_error("not a RIFF/WAVE file: " + path.string());
  }

  WavFormat format{};
  std::vector<unsigned char> audioData;

  while (stream) {
    std::array<char, 4> chunkId{};
    stream.read(chunkId.data(), chunkId.size());
    if (!stream) {
      break;
    }
    const std::uint32_t chunkSize = readU32(stream);

    if (chunkIdEquals(chunkId, "fmt ")) {
      format.audioFormat = readU16(stream);
      format.channels = readU16(stream);
      format.sampleRate = readU32(stream);
      const std::uint32_t byteRate = readU32(stream);
      const std::uint16_t blockAlign = readU16(stream);
      (void)byteRate;
      (void)blockAlign;
      format.bitsPerSample = readU16(stream);

      const std::streamoff remaining =
          static_cast<std::streamoff>(chunkSize) - 16;
      if (remaining > 0) {
        stream.seekg(remaining, std::ios::cur);
      }
    } else if (chunkIdEquals(chunkId, "data")) {
      audioData.resize(chunkSize);
      stream.read(reinterpret_cast<char *>(audioData.data()),
                  static_cast<std::streamsize>(audioData.size()));
      if (!stream) {
        throw std::runtime_error("truncated WAV data chunk");
      }
    } else {
      stream.seekg(chunkSize, std::ios::cur);
    }

    if ((chunkSize % 2U) != 0U) {
      stream.seekg(1, std::ios::cur);
    }
  }

  if (format.channels == 0 || format.sampleRate == 0 || audioData.empty()) {
    throw std::runtime_error("WAV file is missing format or audio data");
  }

  const std::uint16_t bytesPerSample =
      static_cast<std::uint16_t>(format.bitsPerSample / 8U);
  if (bytesPerSample == 0 || (audioData.size() % bytesPerSample) != 0U) {
    throw std::runtime_error("invalid WAV sample layout");
  }

  const std::size_t sampleCount = audioData.size() / bytesPerSample;
  std::vector<float> samples;
  samples.reserve(sampleCount);
  for (std::size_t index = 0; index < sampleCount; ++index) {
    const std::size_t offset = index * bytesPerSample;
    if (format.audioFormat == 1) {
      samples.push_back(
          decodePcmSample(audioData, offset, format.bitsPerSample));
    } else if (format.audioFormat == 3 && format.bitsPerSample == 32) {
      samples.push_back(decodeFloatSample(audioData, offset));
    } else {
      throw std::runtime_error("unsupported WAV encoding");
    }
  }

  return AudioBuffer(format.sampleRate, format.channels, std::move(samples));
}

void writeWavFile16(const std::filesystem::path &path,
                    const AudioBuffer &buffer) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to create WAV file: " + path.string());
  }

  constexpr std::uint16_t bitsPerSample = 16;
  const std::uint16_t blockAlign =
      static_cast<std::uint16_t>(buffer.channels() * (bitsPerSample / 8U));
  const std::uint32_t byteRate = buffer.sampleRate() * blockAlign;
  const std::uint32_t dataSize =
      static_cast<std::uint32_t>(buffer.sampleCount() * sizeof(std::int16_t));
  const std::uint32_t riffSize = 36U + dataSize;

  stream.write("RIFF", 4);
  writeU32(stream, riffSize);
  stream.write("WAVE", 4);

  stream.write("fmt ", 4);
  writeU32(stream, 16);
  writeU16(stream, 1);
  writeU16(stream, buffer.channels());
  writeU32(stream, buffer.sampleRate());
  writeU32(stream, byteRate);
  writeU16(stream, blockAlign);
  writeU16(stream, bitsPerSample);

  stream.write("data", 4);
  writeU32(stream, dataSize);

  for (const float sample : buffer.samples()) {
    const float clipped = std::clamp(sample, -1.0F, 1.0F);
    const auto quantized = static_cast<std::int16_t>(
        std::lrint(clipped * (clipped < 0.0F ? 32768.0F : 32767.0F)));
    writeU16(stream, static_cast<std::uint16_t>(quantized));
  }
}

} // namespace npu_audio
