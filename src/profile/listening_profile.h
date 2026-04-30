#pragma once

namespace snae::profile {

struct ListeningProfile {
    float clarity_preference = 0.45F;
    float warmth_preference = 0.35F;
    float stereo_preference = 0.20F;
    float low_volume_enhancement = 0.35F;
    float late_night_mode = 0.0F;
    float enhancement_amount = 0.85F;
};

ListeningProfile clampProfile(const ListeningProfile& profile);

}  // namespace snae::profile
