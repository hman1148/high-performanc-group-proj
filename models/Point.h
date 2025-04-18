//
// Created by Joe Brijs on 3/22/25.
//
#ifndef POINT_H
#define POINT_H

#include <utility>
#include <vector>
#include <cmath>

namespace SpotifyGenreRevealParty
{

    struct Point
    {
        std::vector<float> features;
        int clusterId;
        double minDist; // default infinite dist to nearest cluster

        Point() : clusterId(-1), minDist(__DBL_MAX__) {}

        // Constructor to initialize dimensions and set default clusterId (-1 means no cluster)
        explicit Point(std::vector<float> features) : features(std::move(features)), clusterId(-1), minDist(__DBL_MAX__) {}

        // Method to calculate Euclidean distance from this point to another point
        float calculateDistance(const Point &other) const
        {
            float sum = 0.0f;
            for (size_t i = 0; i < features.size(); i++)
            {
                sum += std::pow(features[i] - other.features[i], 2);
            }
            return std::sqrt(sum);
        }
    };
}
#endif // POINT_H
