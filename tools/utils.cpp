//
// Created by Joe Brijs on 3/22/25.
//

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "../models/Point.h"
#include "utils.h"

using namespace SpotifyGenreRevealParty;

namespace utils
{
    double euclideanDistance(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Vectors must be of the same dimension");
        }

        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            const double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    void writePointsAndCentroidsToFile(const std::vector<Point> &points, const std::vector<Point> &centroids, const std::string &filename)
    {
        std::ofstream file(filename, std::ios::trunc);
        if (!file)
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // Write header based on feature order from extractFeatures
        file << "index,danceability,energy,tempo,loudness,speechiness,acousticness,instrumentalness,liveness,valence,cluster_id,is_centroid\n";

        size_t index = 0;

        // Write data points
        for (const auto &point : points)
        {
            file << index++;
            for (const auto &feature : point.features)
            {
                file << "," << feature;
            }
            file << "," << point.clusterId << ",false\n";
        }

        // Write centroids
        for (const auto &centroid : centroids)
        {
            file << index++;
            for (const auto &feature : centroid.features)
            {
                file << "," << feature;
            }
            file << "," << centroid.clusterId << ",true\n";
        }

        file.close();
        std::cout << "Output CSV written to: " << filename << std::endl;
    }
}
