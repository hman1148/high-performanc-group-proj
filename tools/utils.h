//
// Created by Joe Brijs on 4/2/25.
//

// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include "../models/Point.h"

namespace SpotifyGenreRevealParty {
    void writePointsAndCentroidsToFile(const std::vector<Point>& points, const std::vector<Point>& centroids, const std::string& filename);
}

#endif // UTILS_H