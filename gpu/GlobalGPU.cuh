// Architecture
// [Node 1] --- [GPU 1]        [Node 2] --- [GPU 2]
//     |                           |
//     |                           |
//     +---- MPI Network  ---------+
//               |
// [Node 3] --- [GPU 3]        [Node 4] --- [GPU 4]

#pragma once

#include <vector>
#include <memory>
#include <string>

#include "../models/IAlgorithm.h"

namespace SpotifyGenreRevealParty
{
    class GlobalGPU : public IAlgorithm
    {
    public:
        GlobalGPU(int clusters, int maxIterations);
        ~GlobalGPU();

        void run(std::vector<SpotifyGenreRevealParty::Point> &data, int k, std::size_t dimensions, int maxIterations, double tolerance) override;
        void saveResultsToCSV(const std::string &filename, const std::vector<std::string> &songIds);

    private:
        // Basic params
        int m_number_of_clusters;
        int m_max_iterations;
        int m_number_of_dimensions;

        // MPI Params
        int m_rank;
        int m_num_processs;
        int m_num_points_per_process;
        bool m_cuda_aware_mpi;

        float *m_device_data;
        float *m_device_centroids;
        float *m_device_temp_centroids;
        int *m_device_cluster_assignments;
        int *m_device_cluster_count;

        // Host mem
        std::vector<float> m_host_flat_data;
        std::vector<float> m_host_centroids;
        std::vector<int> m_host_cluster_assignments;
        std::vector<int> m_global_cluster_assignments;

        // methods
        void initializeMPI();
        bool isMpiCudaAware();
        void distributeData(const std::vector<SpotifyGenreRevealParty::Point> &data);
        void initializeCentroids(int k, int dimensions);
        void runDistributedKMeans(double tolerance);
        void gatherResults(std::vector<SpotifyGenreRevealParty::Point> &data);

        // Memory management
        void allocateGPUMemory();
        void freeGPUMemory();
    };
}