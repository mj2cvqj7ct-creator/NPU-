#include "profile/listening_profile.h"

#include <algorithm>

namespace snae::profile {

ListeningProfile clampProfile(const ListeningProfile& profile) {
    ListeningProfile clamped = profile;
    clamped.clarity_preference = std::clamp(clamped.clarity_preference, -1.0F, 1.0F);
    clamped.warmth_preference = std::clamp(clamped.warmth_preference, -1.0F, 1.0F);
    clamped.stereo_preference = std::clamp(clamped.stereo_preference, -1.0F, 1.0F);
    clamped.low_volume_enhancement = std::clamp(clamped.low_volume_enhancement, 0.0F, 1.0F);
    clamped.late_night_mode = std::clamp(clamped.late_night_mode, 0.0F, 1.0F);
    clamped.enhancement_amount = std::clamp(clamped.enhancement_amount, 0.0F, 1.0F);
    return clamped;
}

}  // namespace snae::profile
