//
// Created by Joe Brijs on 3/21/25.
//

#pragma once
#include "../models/SpotifyFrame.h"
#include <vector>

#include "../models/Point.h"

namespace SpotifyGenreRevealParty {

    // Function declarations
    std::vector<float> extractFeatures(const SpotifyFrame& frame);
    void minMaxScale(std::vector<Point>& data);
    std::vector<Point> prepareDataForKMeans(const std::vector<SpotifyFrame>& frames);

}  // namespace SpotifyGenreRevealParty