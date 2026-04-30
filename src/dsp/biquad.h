#pragma once

#include <array>

namespace snae::dsp {

enum class BiquadType {
    LowShelf,
    Peaking,
    HighShelf,
};

class StereoBiquad {
public:
    StereoBiquad() = default;

    void configure(BiquadType type, float sample_rate_hz, float frequency_hz, float q, float gain_db);
    void reset();
    void process(float& left, float& right);

private:
    struct ChannelState {
        float x1 = 0.0F;
        float x2 = 0.0F;
        float y1 = 0.0F;
        float y2 = 0.0F;
    };

    std::array<float, 5> coefficients_{1.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::array<ChannelState, 2> state_{};

    [[nodiscard]] float process_channel(float sample, ChannelState& state) const;
};

}  // namespace snae::dsp
