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
    void distributedKMeans(std::vector<SpotifyGenreRevealParty::Point>& fullData, int k, size_t dimensions, int maxIterations, double tolerance, int rank, int worldSize)
    {
        using namespace SpotifyGenreRevealParty;

        // Divvy up the points to each rank
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
        //CHATGPT LOOKE HERE
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
        MPI_Bcast(centroidsFlattened.data(), centroidsFlattened.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

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

            // Compute partial sums and counts on each rank
            std::vector<std::vector<float>> localFeaturesSums(k, std::vector<float>(dimensions, 0.0));
            std::vector<int> localClustersSums(k, 0);

            for (int i = 0; i< localPoints.size(); i++){
                const Point& point = localPoints[i]; //Faster retreval optimizing it since we access it so much
                localClustersSums[point.clusterId]++;
                for(int j = 0; j < dimensions; j++){
                    localFeaturesSums[point.clusterId][j] = localFeaturesSums[point.clusterId][j] + point.features[j];
                }
            }

            //Reduce all partial sums and counts
            std::vector<std::vector<float>> globalFeaturesSums(k, std::vector<float>(dimensions, 0.0));
            std::vector<int> globalClustersSums(k, 0);

            MPI_Allreduce(localClustersSums.data(), globalClustersSums.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            for(int i= 0; i < k; i++){
                MPI_Allreduce(localFeaturesSums[i].data(), globalFeaturesSums[i].data(), dimensions, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            
            // Update centroids 
            std::vector<Point> oldCentroids = centroids;

            for (int i = 0; i < k; ++i) { //for every cluster
                if (globalClustersSums[i] > 0) { 
                    for (size_t j = 0; j < dimensions; ++j) { //for every feature
                        centroids[i].features[j] = globalFeaturesSums[i][j] / globalClustersSums[i];
                    }
                }
            }
            
            //Check convergence rank 0 and tell all ranks to quit if convergence reached
            int doneYet = 0;
            if (rank == 0) {
                if (hasConverged(oldCentroids, centroids, tolerance)){
                    doneYet = 1;
                }
            }
            MPI_Bcast(&doneYet, 1, MPI_INT, 0, MPI_COMM_WORLD);

            

            if (doneYet){
                if (rank == 0){
                   std::cout << "Convergence reached after " << iter + 1 << " iterations." << std::endl;
                }

                break;
            }
    
        }

        //  Gather all labeled points back to rank 0 and write to file
        // Flatten localPoints into a float buffer for gathering (features + clusterId)
        int localNumPoints = localPoints.size();
        int localDataSize = localNumPoints * (dimensions + 1);  // +1 for clusterId

        std::vector<float> localFlatData;
        for (const auto& point : localPoints) {
            localFlatData.insert(localFlatData.end(), point.features.begin(), point.features.end());
            localFlatData.push_back(static_cast<float>(point.clusterId));  // add clusterId as last element
        }

        // Rank 0 prepares buffer to receive all points
        std::vector<float> gatheredFlatData;
        if (rank == 0) {
            gatheredFlatData.resize(worldSize * localDataSize);
        }

        // All ranks participate in the gather
        MPI_Gather(
            localFlatData.data(), localDataSize, MPI_FLOAT,
            rank == 0 ? gatheredFlatData.data() : nullptr, localDataSize, MPI_FLOAT,
            0, MPI_COMM_WORLD
        );

        // Rank 0 reconstructs all points and writes to file
        if (rank == 0) {
            std::vector<Point> allPoints;
            int totalPoints = worldSize * localNumPoints;

            for (int i = 0; i < totalPoints; ++i) {
                std::vector<float> features(dimensions);
                for (int j = 0; j < dimensions; ++j) {
                    features[j] = gatheredFlatData[i * (dimensions + 1) + j];
                }
                int clusterId = static_cast<int>(gatheredFlatData[i * (dimensions + 1) + dimensions]);

                Point p(features);
                p.clusterId = clusterId;
                allPoints.push_back(p);
            }

            utils::writePointsAndCentroidsToFile(allPoints, centroids, "../output/distributed_cpu_results.txt");
            std::cout << "Final centroids written to output." << std::endl;
        }
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
