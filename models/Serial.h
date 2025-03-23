//
// Created by Joe Brijs on 3/21/25.
//

#ifndef SERIAL_H
#define SERIAL_H

#include "IAlgorithm.h"
#include <iostream>
#include <vector>
#include <memory>
#include <random>  // Use for better random number generation
#include <ctime>
#include "../models/Point.h"

class Serial : public IAlgorithm {
public:
    void run(std::vector<SpotifyGenreRevealParty::Point> data, int k, size_t dimensions) override {
        std::cout << "Running Serial implementation." << std::endl;
        serial(k, dimensions); // Call the serial function that generates centroids
    }

private:
    // The serial function that generates and prints centroids
    void serial(std::vector<SpotifyGenreRevealParty::Point>& points, int k, int dimensions) {
        auto centroids = generateCentroids(k, dimensions);

        // Assign points to cluster
        for (size_t i = 0; i < k; ++i) {  // Iterate over centroids
            auto& centroid = centroids[i];  // Get the current centroid
            int clusterId = i;  // Cluster ID is the index of the centroid

            for (auto& point : points) {  // Iterate over the points
                double dist = centroid->calculateDistance(point);

                // If the point is closer to this centroid, update its distance and cluster ID
                if (dist < point.minDist) {
                    point.minDist = dist;
                    point.clusterId = clusterId;
                }
            }
        }
    }


    // Generate k centroids with random features
    std::vector<std::unique_ptr<SpotifyGenreRevealParty::Point>> generateCentroids(int k, int numFeatures) {
        std::vector<std::unique_ptr<SpotifyGenreRevealParty::Point>> centroids;

        // Random number generator setup
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Random float between 0 and 1

        // Generate k centroids
        for (int i = 0; i < k; i++) {
            std::vector<float> feats;

            // Assign random float values to each feature
            for (int j = 0; j < numFeatures; j++) {
                feats.push_back(dis(gen));  // Generate random value between 0 and 1
            }

            // Create a new Point using the features vector and use unique_ptr for memory management
            auto centroid = std::make_unique<SpotifyGenreRevealParty::Point>(feats);

            // Add the Point to the centroids vector
            centroids.push_back(std::move(centroid));
        }

        return centroids;
    }
};

#endif // SERIAL_H
