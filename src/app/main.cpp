#include "audio_enhancer/audio_pipeline.h"
#include "audio_enhancer/npu_backend.h"
#include "audio_enhancer/wav_io.h"

#include <exception>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* executable) {
    std::cerr << "Usage: " << executable << " <input.wav> <output.wav> [--target-lufs value]\n"
              << "Environment:\n"
              << "  AUDIO_ENHANCER_ENABLE_QNN_NPU=1   prefer Snapdragon/QNN NPU shim\n"
              << "  AUDIO_ENHANCER_ENABLE_DIRECTML=1  prefer DirectML shim when QNN is disabled\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        audio_enhancer::PipelineConfig config;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--target-lufs" && i + 1 < argc) {
                config.target_lufs = std::stof(argv[++i]);
            } else {
                print_usage(argv[0]);
                return 2;
            }
        }

        audio_enhancer::WavFile wav = audio_enhancer::read_wav(argv[1]);
        audio_enhancer::AudioPipeline pipeline(config);
        const audio_enhancer::PipelineStats stats = pipeline.Process(wav.buffer);
        audio_enhancer::write_wav(argv[2], wav);

        std::cout << "backend=" << audio_enhancer::ToString(stats.selected_backend)
                  << " status=\"" << stats.backend_status << "\""
                  << " input_rms=" << stats.input_rms
                  << " output_rms=" << stats.output_rms
                  << " gain_db=" << stats.applied_gain_db
                  << " peak=" << stats.peak << '\n';
    } catch (const std::exception& error) {
        std::cerr << "audio_enhancer_cli: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
