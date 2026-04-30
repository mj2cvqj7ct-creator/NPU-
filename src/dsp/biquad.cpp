#include "dsp/biquad.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace snae::dsp {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr float kMinGainDb = -12.0F;
constexpr float kMaxGainDb = 12.0F;
}

void StereoBiquad::configure(BiquadType type, float sample_rate_hz, float frequency_hz, float q, float gain_db) {
    if (sample_rate_hz <= 0.0F || frequency_hz <= 0.0F || frequency_hz >= sample_rate_hz * 0.5F || q <= 0.0F) {
        throw std::invalid_argument("invalid biquad parameters");
    }

    gain_db = std::clamp(gain_db, kMinGainDb, kMaxGainDb);
    const double a = std::pow(10.0, gain_db / 40.0);
    const double omega = 2.0 * kPi * frequency_hz / sample_rate_hz;
    const double sin_omega = std::sin(omega);
    const double cos_omega = std::cos(omega);
    const double alpha = sin_omega / (2.0 * q);

    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a0 = 1.0;
    double a1 = 0.0;
    double a2 = 0.0;

    if (type == BiquadType::Peaking) {
        b0 = 1.0 + alpha * a;
        b1 = -2.0 * cos_omega;
        b2 = 1.0 - alpha * a;
        a0 = 1.0 + alpha / a;
        a1 = -2.0 * cos_omega;
        a2 = 1.0 - alpha / a;
    } else {
        const double shelf_alpha = sin_omega / 2.0 * std::sqrt((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0);
        const double beta = 2.0 * std::sqrt(a) * shelf_alpha;
        if (type == BiquadType::LowShelf) {
            b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + beta);
            b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
            b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - beta);
            a0 = (a + 1.0) + (a - 1.0) * cos_omega + beta;
            a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
            a2 = (a + 1.0) + (a - 1.0) * cos_omega - beta;
        } else {
            b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + beta);
            b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
            b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - beta);
            a0 = (a + 1.0) - (a - 1.0) * cos_omega + beta;
            a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
            a2 = (a + 1.0) - (a - 1.0) * cos_omega - beta;
        }
    }

    coefficients_ = {static_cast<float>(b0 / a0), static_cast<float>(b1 / a0),
                     static_cast<float>(b2 / a0), static_cast<float>(a1 / a0),
                     static_cast<float>(a2 / a0)};
}

void StereoBiquad::reset() {
    state_ = {};
}

void StereoBiquad::process(float& left, float& right) {
    left = process_channel(left, state_[0]);
    right = process_channel(right, state_[1]);
}

float StereoBiquad::process_channel(float sample, ChannelState& state) const {
    const float output = coefficients_[0] * sample + coefficients_[1] * state.x1 +
                         coefficients_[2] * state.x2 - coefficients_[3] * state.y1 -
                         coefficients_[4] * state.y2;
    state.x2 = state.x1;
    state.x1 = sample;
    state.y2 = state.y1;
    state.y1 = output;
    return output;
}

}  // namespace snae::dsp
