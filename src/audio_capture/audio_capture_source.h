#pragma once

#include <string>

#include "dsp/audio_types.h"

namespace sxnae::audio_capture {

struct CaptureDeviceInfo {
    std::string id;
    std::string display_name;
    dsp::AudioFormat mix_format;
};

class IAudioCaptureSource {
public:
    virtual ~IAudioCaptureSource() = default;

    virtual CaptureDeviceInfo deviceInfo() const = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual bool readBlock(dsp::AudioBlock& block) = 0;
};

}  // namespace sxnae::audio_capture
