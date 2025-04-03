//
// Created by Joe Brijs on 3/21/25.
//

#ifndef SERIAL_H
#define SERIAL_H

#include "IAlgorithm.h"
#include <iostream>
#include <vector>
#include <memory>
#include <random> // For better random number generation
#include <ctime>
#include "../models/Point.h"
# include "../tools/utils.h"

class Serial final : public IAlgorithm
{
public:
    void run(std::vector<SpotifyGenreRevealParty::Point> &data, const int k, const size_t dimensions, const int maxIterations = 100, const double tolerance = 1e-4) override
    {
        std::cout << "Running Serial implementation." << std::endl;

        serial(data, k, dimensions, maxIterations, tolerance); // Pass the tolerance for early stopping
    }

private:
    static void serial(std::vector<SpotifyGenreRevealParty::Point> &points, const int k, const int dimensions, const int maxIterations, const double tolerance)
    {
        auto centroids = generateCentroids(k, dimensions);

        for (int iter = 0; iter < maxIterations; ++iter)
        { // Iterate for maxIterations times
            std::cout << "Iteration " << iter + 1 << std::endl;

            // Move centroids to prevCentroids for convergence check
            auto prevCentroids = centroids;

            // Assign points to clusters
            assignPointsToClusters(points, centroids, k);

            // Compute the new centroids based on the points
            computeCentroids(points, centroids, k);

            // Check for convergence: compare the old and new centroids
            if (hasConverged(prevCentroids, centroids, tolerance))
            {
                std::cout << "Convergence reached after " << iter + 1 << " iterations." << std::endl;
                utils::writePointsAndCentroidsToFile(points, centroids, "../output/serial_results.txt");
                break; // Exit early if the centroids have converged
            }
        }
        std::cout << "Convergence was not reached after " << maxIterations << " iterations." << std::endl;
        utils::writePointsAndCentroidsToFile(points, centroids, "../output/serial_results.txt");
    }

    // Function to assign points to the closest centroid
    static void assignPointsToClusters(std::vector<SpotifyGenreRevealParty::Point> &points,
                                       const std::vector<SpotifyGenreRevealParty::Point> &centroids, const int k)
    {
        for (auto &point : points)
        {
            double minDist = __DBL_MAX__;
            int clusterId = -1;

            // Find the nearest centroid to the point
            for (int i = 0; i < k; ++i)
            {
                double dist = centroids[i].calculateDistance(point);
                if (dist < minDist)
                {
                    minDist = dist;
                    clusterId = i;
                }
            }

            // Update the point's clusterId and its minimum distance
            point.minDist = minDist;
            point.clusterId = clusterId;
        }
    }

    // Function to compute new centroids based on the points
    static void computeCentroids(std::vector<SpotifyGenreRevealParty::Point> &points,
                                 std::vector<SpotifyGenreRevealParty::Point> &centroids, int k)
    {
        std::vector<int> nPoints(k, 0);          // Number of points in each cluster
        std::vector<std::vector<double>> sum(k); // Sum of feature values for each cluster

        // Accumulate the sum of feature values for each cluster
        for (auto &point : points)
        {
            int clusterId = point.clusterId;
            nPoints[clusterId] += 1;

            // Accumulate feature values
            for (size_t featureIndex = 0; featureIndex < point.features.size(); ++featureIndex)
            {
                if (sum[clusterId].size() <= featureIndex)
                {
                    sum[clusterId].push_back(0.0); // Ensure space for all features
                }
                sum[clusterId][featureIndex] += point.features[featureIndex];
            }

            point.minDist = __DBL_MAX__; // Reset the distance for the point
        }

        // Compute the new centroids based on the feature sums
        for (size_t clusterId = 0; clusterId < k; ++clusterId)
        {
            if (nPoints[clusterId] > 0)
            {
                for (size_t featureIndex = 0; featureIndex < sum[clusterId].size(); ++featureIndex)
                {
                    centroids[clusterId].features[featureIndex] = sum[clusterId][featureIndex] / nPoints[clusterId];
                }
            }
        }
    }

    // Function to compare the old centroids with the new centroids
    static bool hasConverged(const std::vector<SpotifyGenreRevealParty::Point> &prevCentroids,
                             const std::vector<SpotifyGenreRevealParty::Point> &centroids,
                             const double tolerance)
    {
        for (size_t i = 0; i < centroids.size(); ++i)
        {
            if (const double dist = prevCentroids[i].calculateDistance(centroids[i]); dist > tolerance)
            {
                return false; // If any centroid has moved more than the tolerance, return false (not converged)
            }
        }
        return true; // If all centroids have moved less than the tolerance, return true (converged)
    }

    // Generate k centroids with random feature values
    static std::vector<SpotifyGenreRevealParty::Point> generateCentroids(const int k, const int numFeatures)
    {
        std::vector<SpotifyGenreRevealParty::Point> centroids;

        std::default_random_engine generator(100);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Random float between 0 and 1

        for (int i = 0; i < k; i++)
        {
            std::vector<float> feats;

            feats.reserve(numFeatures);
            for (int j = 0; j < numFeatures; j++)
            {
                feats.push_back(dis(generator)); // Generate random value between 0 and 1
            }

            centroids.emplace_back(feats); // Create and push centroid
        }

        return centroids;
    }
};

#endif // SERIAL_H
