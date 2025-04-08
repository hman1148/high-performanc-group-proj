#pragma once

#ifndef DISTRIBUTEDCPU_H
#define DISTRIBUTEDCPU_H

#include "IAlgorithm.h"
#include <mpi.h>
#includ <point.h>
#include <vector>
#include <iostream>
#include "../models/Point.h"
#include "../tools/utils.h"

class DistributedCpu : public IAlgorithm {
public:
    void run(std::vector<SpotifyGenreRevealParty::Point>& data, int k, size_t dimensions, int maxIterations = 100, double tolerance = 1e-4) override
    {
        int rank, worldSize;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

        if (rank == 0)
            std::cout << "Running Distributed Memory CPU implementation with " << worldSize << " ranks." << std::endl;

        distributedKMeans(data, k, dimensions, maxIterations, tolerance, rank, worldSize);
    }

private:
    void distributedKMeans(std::vector<SpotifyGenreRevealParty::Point>& fullData, int k, size_t dimensions, int maxIterations, double tolerance, int rank, int worldSize)
    {
        using namespace SpotifyGenreRevealParty;

        // TODO 1: Divvy up the points to each rank
        int totalPoints = fullData.size();
        int localSize = totalPoints / worldSize;

        std::vector<float> flatData;

        std::vector<float> localData(localSize * dimensions);
        // Note: You’ll need to serialize/deserialize Point objects manually or switch to raw arrays of floats


        //flatten for MPI_Scatter
        if (rank == 0) {
            for (int i = 0; i < totalPoints; i++){
                flatData.insert(flatData.end(), fullData[i].features.begin(), fullData[i].features.end());
            }
        }

        MPI_Scatter(flatData.data(), localSize * dimensions, MPI_FLOAT, localData.data(), localSize * dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);

        //Begin reconstruction
        std::vector<Point> localPoints;

        for (int i = 0; i < localSize; i++) {
            std::vector<float> features(dimensions);
            for (size_t j = 0; j < dimensions; j++) {
                features[j] = localData[i * dimensions + j];
            }
            localPoints.emplace_back(features);
        }

        // if its rank 0 assign the centroids, and then broadcast, and reconcstruct
        std::vector<Point> centroids;
        std::vector<float> centroidsFlattened;
        if (rank == 0) {
            centroids = generateCentroids(k, dimensions);

            for (int i = 0; i < centroids.size(); i++){
                centroidsFlattened.insert(centroidsFlattened.end(), centroids[i].features.begin(), centroids[i].features.end());
            }

            
        }
        MPI_Bcast(centroidsFlattened, centroidsFlattened.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            for (int i = 0; i < k; ++i) {
                std::vector<float> features(dimensions);
                for (size_t d = 0; d < dimensions; d++) {
                    features[d] = centroidsFlattened[i * dimensions + d];
                }
                centroids.emplace_back(features);
            }
        }

        for (int iter = 0; iter < maxIterations; ++iter) {
            if (rank == 0)
                std::cout << "Iteration " << iter + 1 << std::endl;

            assignPointsToClusters(localPoints, centroids, k);

            // TODO 5: Compute partial sums and counts on each rank
            std::vector<std::vector<float>> localFeaturesSums(k, std::vector<double>(dimensions, 0.0));

            // TODO 6: Reduce all partial sums and counts

            // TODO 7: Update centroids (all ranks now have global info)
            // TODO 8: Check convergence on all ranks
        }

        // TODO 9: Optionally gather all labeled points back to rank 0 and write to file
    }

    void assignPointsToClusters(std::vector<SpotifyGenreRevealParty::Point>& points,
                                const std::vector<SpotifyGenreRevealParty::Point>& centroids,
                                int k)
    {
        for (auto& point : points) {
            double minDist = __DBL_MAX__;
            int clusterId = -1;
            for (int i = 0; i < k; ++i) {
                double dist = centroids[i].calculateDistance(point);
                if (dist < minDist) {
                    minDist = dist;
                    clusterId = i;
                }
            }
            point.clusterId = clusterId;
            point.minDist = minDist;
        }
    }

    bool hasConverged(const std::vector<SpotifyGenreRevealParty::Point>& prev,
                      const std::vector<SpotifyGenreRevealParty::Point>& curr,
                      double tolerance)
    {
        for (size_t i = 0; i < curr.size(); ++i) {
            if (prev[i].calculateDistance(curr[i]) > tolerance)
                return false;
        }
        return true;
    }

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

#endif // DISTRIBUTEDCPU_H
