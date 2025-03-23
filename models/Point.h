//
// Created by Joe Brijs on 3/22/25.
//


#include <vector>
#include <cmath>

namespace SpotifyGenreRevealParty {

#ifndef POINT_H
#define POINT_H

struct Point {
    std::vector<float> dimensions;
    int clusterId;

    // Constructor to initialize dimensions and set default clusterId (-1 means no cluster)
    Point(std::vector<float> dims) : dimensions(dims), clusterId(-1) {}

    // Method to calculate Euclidean distance from this point to another point
    float calculateDistance(const Point& other) const {
        float sum = 0.0f;
        for (size_t i = 0; i < dimensions.size(); i++) {
            sum += std::pow(dimensions[i] - other.dimensions[i], 2);
        }
        return std::sqrt(sum);
    }
};

#endif // POINT_H


}
