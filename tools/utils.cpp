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


namespace utils {
    double euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension");
        }

        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    void writePointsAndCentroidsToFile(const std::vector<SpotifyGenreRevealParty::Point>& points, const std::vector<SpotifyGenreRevealParty::Point>& centroids, const std::string& filename) {
        std::ofstream file(filename, std::ios::trunc); // Open in truncate mode to overwrite if exists
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // Write points
        file << "# Points" << std::endl;
        for (const auto& point : points) {
            for (const auto& feature : point.features) {
                file << feature << " ";
            }
            file << "| Cluster: " << point.clusterId << std::endl;
        }

        // Write centroids
        file << "\n# Centroids" << std::endl;
        for (const auto& centroid : centroids) {
            for (const auto& feature : centroid.features) {
                file << feature << " ";
            }
            file << "| Centroid" << std::endl;
        }

        file.close();
        std::cout << "Output file written to:  " << filename << std::endl;
    }

}
