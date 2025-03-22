//
// Created by Joe Brijs on 3/21/25.
//

#pragma once
#include "../models/SpotifyFrame.h"
#include <vector>

namespace SpotifyGenreRevealParty {

    // Function declarations
    std::vector<float> extractFeatures(const SpotifyFrame& frame);
    void minMaxScale(std::vector<std::vector<float>>& data);
    std::vector<std::vector<float>> prepareDataForKMeans(const std::vector<SpotifyFrame>& frames);

}  // namespace SpotifyGenreRevealParty