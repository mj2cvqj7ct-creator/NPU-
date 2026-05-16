#include "npu_audio/enhancer.hpp"
#include "npu_audio/wav_io.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " <input.wav> <output.wav> [--service spotify|apple-music|youtube-music]"
               " [--target-db -18] [--bass 0.0] [--clarity 0.0]"
               " [--width 1.0]\n";
}

float parseFloatOption(const std::string &option, const char *text) {
  try {
    std::size_t parsed = 0;
    const std::string value(text);
    const float result = std::stof(value, &parsed);
    if (parsed != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return result;
  } catch (const std::exception &) {
    throw std::invalid_argument("Invalid value for " + option + ": " + text);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 2;
  }

  if (((argc - 3) % 2) != 0) {
    std::cerr << "Options must be provided as flag/value pairs.\n";
    printUsage(argv[0]);
    return 2;
  }

  try {
    npu_audio::EnhancementProfile profile{};
    for (int index = 3; index + 1 < argc; index += 2) {
      const std::string option = argv[index];
      const char *rawValue = argv[index + 1];
      if (option == "--service") {
        profile = npu_audio::profileForService(rawValue);
      } else if (option == "--target-db") {
        profile.targetLoudnessDb = parseFloatOption(option, rawValue);
      } else if (option == "--bass") {
        profile.bassEnhancement = parseFloatOption(option, rawValue);
      } else if (option == "--clarity") {
        profile.clarityEnhancement = parseFloatOption(option, rawValue);
      } else if (option == "--width") {
        profile.stereoWidth = parseFloatOption(option, rawValue);
      } else {
        std::cerr << "Unknown option: " << option << '\n';
        printUsage(argv[0]);
        return 2;
      }
    }

    npu_audio::AudioBuffer buffer =
        npu_audio::readWavFile(std::filesystem::path(argv[1]));
    npu_audio::AudioEnhancer enhancer;
    const npu_audio::EnhancementReport report = enhancer.process(buffer, profile);
    npu_audio::writeWavFile16(std::filesystem::path(argv[2]), buffer);

    std::cout << "backend=" << report.backendName
              << " accelerated=" << (report.acceleratedBackend ? "yes" : "no")
              << " input_peak_db=" << report.inputPeakDb
              << " output_peak_db=" << report.outputPeakDb
              << " loudness_gain_db=" << report.loudnessGainDb << '\n';
  } catch (const std::exception &error) {
    std::cerr << "npu_audio_enhance failed: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
