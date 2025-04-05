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
    void writePointsToBinary(const std::string& filename, const std::vector<Point>& points);
    std::vector<Point> readPointsFromBinary(const std::string& filename);
    std::vector<Point> getOrLoadPoints(const std::string& csvFile, const std::string& binaryCache);

}  // namespace SpotifyGenreRevealParty