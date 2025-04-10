#pragma once

#ifndef DISTRIBUTEDCPU_H
#define DISTRIBUTEDCPU_H

#include "IAlgorithm.h"
#include <mpi.h>
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
    void distributedKMeans(std::vector<SpotifyGenreRevealParty::Point>& fullData, 
                           int k, 
                           size_t dimensions, 
                           int maxIterations, 
                           double tolerance, 
                           int rank, 
                           int worldSize)
    {
        using namespace SpotifyGenreRevealParty;

        int totalPoints = fullData.size();

        std::vector<float> flatData;

        // Same scatter logic
        std::vector<int> sendCounts(worldSize, 0);
        std::vector<int> displs(worldSize, 0);

        int base = totalPoints / worldSize;
        int remainder = totalPoints % worldSize;

        for (int i = 0; i < worldSize; i++) {
            int pointsForRank = (i < remainder) ? (base + 1) : base;
            sendCounts[i] = pointsForRank * dimensions; 
        }

        displs[0] = 0;
        for (int i = 1; i < worldSize; i++) {
            displs[i] = displs[i - 1] + sendCounts[i - 1];
        }

        if (rank == 0) {
            flatData.reserve(totalPoints * dimensions);
            for (int i = 0; i < totalPoints; i++) {
                flatData.insert(flatData.end(),
                                fullData[i].features.begin(),
                                fullData[i].features.end());
            }
        }

        int localFloatCount = sendCounts[rank];
        std::vector<float> localData(localFloatCount);

        MPI_Scatterv(rank == 0 ? flatData.data() : nullptr, 
                     sendCounts.data(), displs.data(), MPI_FLOAT,
                     localData.data(), localFloatCount, MPI_FLOAT,
                     0, MPI_COMM_WORLD);

        // Reconstruct local points
        int localSize = localFloatCount / dimensions;
        std::vector<Point> localPoints;
        localPoints.reserve(localSize);
        for (int i = 0; i < localSize; i++) {
            std::vector<float> features(dimensions);
            for (size_t j = 0; j < dimensions; j++) {
                features[j] = localData[i * dimensions + j];
            }
            localPoints.emplace_back(features);
        }

        // Generate or broadcast centroids
        std::vector<Point> centroids;
        std::vector<float> centroidsFlattened;
        if (rank == 0) {
            centroids = generateCentroids(k, dimensions);
            centroidsFlattened.reserve(k * dimensions);
            for (int i = 0; i < k; i++) {
                centroidsFlattened.insert(centroidsFlattened.end(),
                                          centroids[i].features.begin(),
                                          centroids[i].features.end());
            }
        }
        centroidsFlattened.resize(k * dimensions);
        MPI_Bcast(centroidsFlattened.data(), centroidsFlattened.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            centroids.clear();
            centroids.reserve(k);
            for (int i = 0; i < k; ++i) {
                std::vector<float> feats(dimensions);
                for (size_t d = 0; d < dimensions; d++) {
                    feats[d] = centroidsFlattened[i * dimensions + d];
                }
                centroids.emplace_back(feats);
            }
        }

        // Main k-means loop
        for (int iter = 0; iter < maxIterations; ++iter) {
            if (rank == 0)
                std::cout << "Iteration " << iter + 1 << std::endl;

            assignPointsToClusters(localPoints, centroids, k);

            // Compute partial sums
            std::vector<std::vector<double>> localFeaturesSums(k, std::vector<double>(dimensions, 0.0));
            std::vector<int> localClustersSums(k, 0);

            for (auto &point : localPoints) {
                localClustersSums[point.clusterId]++;
                for (size_t j = 0; j < dimensions; j++) {
                    localFeaturesSums[point.clusterId][j] += point.features[j];
                }
            }

            // Allreduce partial sums
            std::vector<std::vector<double>> globalFeaturesSums(k, std::vector<double>(dimensions, 0.0));
            std::vector<int> globalClustersSums(k, 0);

            MPI_Allreduce(localClustersSums.data(), globalClustersSums.data(),
                          k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            for (int i = 0; i < k; i++) {
                MPI_Allreduce(localFeaturesSums[i].data(), globalFeaturesSums[i].data(),
                              dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }

            // Update centroids
            std::vector<Point> oldCentroids = centroids;
            for (int i = 0; i < k; ++i) {
                if (globalClustersSums[i] > 0) {
                    for (size_t j = 0; j < dimensions; ++j) {
                        centroids[i].features[j] =
                            globalFeaturesSums[i][j] / globalClustersSums[i];
                    }
                }
            }

            // Check convergence on rank 0
            int doneYet = 0;
            if (rank == 0) {
                if (hasConverged(oldCentroids, centroids, tolerance)) {
                    doneYet = 1;
                }
            }
            MPI_Bcast(&doneYet, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (doneYet) {
                if (rank == 0) {
                    std::cout << "Convergence reached after " << iter + 1 << " iterations." << std::endl;
                }
                break;
            }
        }

        // Build localFlatData
        int localNumPoints = localPoints.size();
        int localDataSize = localNumPoints * (dimensions + 1); 
        std::vector<float> localFlatData;
        localFlatData.reserve(localDataSize);

        for (size_t i = 0; i < localPoints.size(); i++) {
            localFlatData.insert(localFlatData.end(),
                                 localPoints[i].features.begin(),
                                 localPoints[i].features.end());
            localFlatData.push_back(static_cast<float>(localPoints[i].clusterId));
        }

        // -- Now we gather variable-sized data from each rank.

        // 1) Gather each rank's localDataSize into recvCounts
        std::vector<int> recvCounts(worldSize, 0);
        MPI_Gather(&localDataSize,               // what I'm sending
                   1, MPI_INT,                   // I'm sending 1 integer
                   rank == 0 ? recvCounts.data() : nullptr, 
                   1, MPI_INT,
                   0, MPI_COMM_WORLD);

        // 2) On rank 0, build recvDispls and allocate gatheredFlatData
        std::vector<int> recvDispls(worldSize);
        std::vector<float> gatheredFlatData;
        if (rank == 0) {
            recvDispls[0] = 0;
            for (int i = 1; i < worldSize; i++) {
                recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
            }
            int totalGatheredFloats = recvDispls[worldSize - 1] + recvCounts[worldSize - 1];
            gatheredFlatData.resize(totalGatheredFloats);
        }

        // 3) Gatherv the actual data
        MPI_Gatherv(localFlatData.data(), localDataSize, MPI_FLOAT,
                    rank == 0 ? gatheredFlatData.data() : nullptr,
                    rank == 0 ? recvCounts.data()        : nullptr,
                    rank == 0 ? recvDispls.data()        : nullptr,
                    MPI_FLOAT, 0, MPI_COMM_WORLD);

        // 4) Rank 0 reconstructs points from gatheredFlatData
        if (rank == 0) {
            std::vector<Point> allPoints;
            allPoints.reserve(totalPoints);

            int offset = 0;
            for (int r = 0; r < worldSize; r++) {
                int count = recvCounts[r];
                int numPointsFromRank = count / (dimensions + 1);
                
                for (int p = 0; p < numPointsFromRank; p++) {
                    std::vector<float> features(dimensions);
                    for (int d = 0; d < dimensions; d++) {
                        features[d] = gatheredFlatData[offset + p*(dimensions + 1) + d];
                    }
                    int clusterId = static_cast<int>(
                        gatheredFlatData[offset + p*(dimensions + 1) + dimensions]
                    );
                    Point tmpPoint(features);
                    tmpPoint.clusterId = clusterId;
                    allPoints.push_back(tmpPoint);
                }
                offset += count;
            }

            utils::writePointsAndCentroidsToFile(allPoints, centroids, "../output/distributed_cpu_results.csv");
        }
    }

    // ... (unchanged helper methods) ...

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
        std::uniform_real_distribution<float> dis(0.0f, 1.0f); 

        for (int i = 0; i < k; i++) {
            std::vector<float> feats;
            feats.reserve(numFeatures);
            for (int j = 0; j < numFeatures; j++) {
                feats.push_back(dis(generator));
            }
            centroids.emplace_back(feats);
        }
        return centroids;
    }
};

#endif // DISTRIBUTEDCPU_H
//AI disclaimer: I used ai to sweep the code and convert it from gather and scatter to gatherv and scatterv after realizing that it would 
//not work if the data disnt divide evenly into the ranks
