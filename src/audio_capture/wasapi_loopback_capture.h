#pragma once

#include "audio_capture/audio_capture_source.h"

namespace sxnae::audio_capture {

// Windows-only implementation seam. The production class will wrap
// IAudioClient/IAudioCaptureClient loopback capture and output 48 kHz stereo
// float blocks for the DSP chain.
class WasapiLoopbackCapture final : public IAudioCaptureSource {
public:
    CaptureDeviceInfo deviceInfo() const override;
    bool start() override;
    void stop() override;
    bool readBlock(dsp::AudioBlock& block) override;
};

}  // namespace sxnae::audio_capture
