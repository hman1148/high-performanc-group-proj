//
// Created by Joe Brijs on 4/2/25.
//

// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include "../models/Point.h"

namespace utils {
    void writePointsAndCentroidsToFile(const std::vector<SpotifyGenreRevealParty::Point>& points, const std::vector<SpotifyGenreRevealParty::Point>& centroids, const std::string& filename);
}

#endif // UTILS_H