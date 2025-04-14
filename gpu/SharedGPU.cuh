//
// Created by Hunter Peart on 3/25/2025.
//

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

#include "../models/IAlgorithm.h"

struct cudaDeviceProp;
typedef struct CUstream_st *cudaStream_t;

namespace SpotifyGenreRevealParty
{
    class SharedGPU : public IAlgorithm
    {
    public:
        SharedGPU(int clusters, int maxIterations);
        ~SharedGPU();

        // Initialize GPU Resources
        void initialize(const std::vector<std::vector<float>> &data);

        // Run KMeans on GPU
        void runKMeans();

        // Get the cluster assignments
        std::vector<int> &getClusterAssignments();

        // Get centroids of clusters
        std::vector<std::vector<float>> &getClusterCentroids();

        void run(std::vector<SpotifyGenreRevealParty::Point> &data,
                 int k,
                 size_t dimensions,
                 int maxIterations,
                 double tolerance) override;

    private:
        // number of clusters
        int m_number_of_clusters;

        // number of iterations
        int m_max_iterations;

        // number of data points
        int m_number_of_data_points;

        // number of dimensions
        int m_number_of_dimensions;

        // device pointers
        float *m_device_data;              // Input Data points
        float *m_device_centroids;         // Centroids
        int *m_device_cluster_assignments; // Cluster Assignments
        int *m_device_cluster_counts;      // Cluster Counts

        // Host copies
        std::vector<std::vector<float>> m_host_centroids;
        std::vector<int> m_host_cluster_assignments;

        // CUDA helper functions
        void allocateMemory();
        void freeMemory();
        void copyDataToDevice(const std::vector<std::vector<float>> &data);
        void copyResultsToHost();
    };
}
