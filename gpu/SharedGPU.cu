//
// Created by Hunter Peart on 3/25/2025.
//

// CUDA includes at global scope
#include <cuda_runtime.h>

// Then include our header
#include "SharedGPU.cuh"

// Define error checking macro
#define CHECK_CUDA_ERROR(call)                                                 \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// CUDA Kernel for assigning points to clusters
__global__ void assignClustersKernel(float *data, float *centroids, int *assignments, int numPoints, int numDimensions, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        float minDistance = __FLT_MAX__;
        int nearestCentroid = 0;

        // Find the nearest centroid for this data point
        for (int i = 0; i < k; ++i)
        {
            float distance = 0.0f;
            for (int j = 0; j < numDimensions; ++j)
            {
                // calculate the difference between a data point's coordinate and a centroid's coordinate along a specific dimension.
                float diff = data[idx * numDimensions + j] - centroids[i * numDimensions + j];
                distance += diff * diff;
            }

            if (distance < minDistance)
            {
                minDistance = distance;
                nearestCentroid = i;
            }
        }
        assignments[idx] = nearestCentroid;
    }
}

// CUDA Kernel for updating centroids (accumulate points)
__global__ void updateCentroidsKernel(float *data, float *centroids, int *assignments, int *clusterCounts, int numPoints, int numDimensions, int k)
{
    // Get block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        int clusterId = assignments[idx];

        // Atomic operations to safely update shared centroids and sizes
        for (int j = 0; j < numDimensions; ++j)
        {
            atomicAdd(&centroids[clusterId * numDimensions + j], data[idx * numDimensions + j]);
        }
        atomicAdd(&clusterCounts[clusterId], 1);
    }
}

// CUDA Kernel for updating centroids (divide by cluster sizes)
__global__ void normalizeCentroidsKernel(float *centroids, int *clusterCounts, int numDimensions, int k)
{
    int clusterId = blockIdx.x * blockDim.x + threadIdx.x;

    if (clusterId < k && clusterCounts[clusterId] > 0)
    {
        for (int j = 0; j < numDimensions; ++j)
        {
            centroids[clusterId * numDimensions + j] /= clusterCounts[clusterId];
        }
    }
}

namespace SpotifyGenreRevealParty
{
    // Constructor
    SharedGPU::SharedGPU(int clusters, int maxIterations)
        : m_number_of_clusters(clusters), m_max_iterations(maxIterations),
          m_number_of_data_points(0), m_number_of_dimensions(0),
          m_device_data(nullptr), m_device_centroids(nullptr),
          m_device_cluster_assignments(nullptr), m_device_cluster_counts(nullptr)
    {
    }

    // Destructor
    SharedGPU::~SharedGPU()
    {
        freeMemory();
    }

    void SharedGPU::run(std::vector<Point> &data,
                        int k,
                        size_t dimensions,
                        int maxIterations,
                        double tolerance)
    {
        // Set the number of clusters and iterations
        m_number_of_clusters = k;
        m_max_iterations = maxIterations;

        std::vector<std::vector<float>> featureData;
        featureData.reserve(data.size());

        // Convert data to the format algorithm expects
        for (auto &&point : data)
        {
            featureData.push_back(point.features);
        }

        // Initialize GPU resources
        initialize(featureData);

        // Run KMeans on GPU
        runKMeans();

        // Update cluster assignments in the original data
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            data[i].clusterId = this->m_host_cluster_assignments[i];
        }

        // Print out some info about clustering results
        std::cout << "KMeans complete with " << this->m_number_of_clusters << " clusters and " << m_max_iterations << " iterations" << std::endl;
        std::cout << "Processed " << this->m_number_of_data_points << " data points with " << m_number_of_dimensions << " dimensions" << std::endl;

        // Print cluster sizes
        std::vector<int> clusterSizes(this->m_number_of_clusters, 0);
        for (auto &&assignment : this->m_host_cluster_assignments)
        {
            clusterSizes[assignment]++;
        }

        std::cout << "Cluster sizes:" << std::endl;
        for (int i = 0; i < this->m_number_of_clusters; i++)
        {
            std::cout << "  Cluster " << i << ": " << clusterSizes[i] << " songs" << std::endl;
        }

        // Print cluster centroids
        std::cout << "Cluster centroids:" << std::endl;
        for (int i = 0; i < m_number_of_clusters; i++)
        {
            std::cout << "  Cluster " << i << ": ";
            for (int j = 0; j < m_number_of_dimensions; j++)
            {
                std::cout << m_host_centroids[i][j] << " ";
            }
            std::cout << std::endl;
        }

        // Free memory
        freeMemory();
    }

    // Initialize GPU Resources
    void SharedGPU::initialize(const std::vector<std::vector<float>> &data)
    {
        if (data.empty())
        {
            throw std::runtime_error("Empty Dataset provided");
        }

        m_number_of_data_points = data.size();
        m_number_of_dimensions = data[0].size();

        // Allocate memory on the GPU
        allocateMemory();

        // Copy data to GPU
        copyDataToDevice(data);

        // Initialize centroids (randomly select k data points as centroids)
        std::vector<int> indices(m_number_of_data_points);
        for (int i = 0; i < m_number_of_data_points; ++i)
        {
            indices[i] = i;
        }

        // Initialize random lib
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        m_host_centroids.resize(m_number_of_clusters);
        for (int i = 0; i < m_number_of_clusters; ++i)
        {
            m_host_centroids[i].resize(m_number_of_dimensions);
            for (int j = 0; j < m_number_of_dimensions; ++j)
            {
                m_host_centroids[i][j] = data[indices[i]][j];
            }
        }

        // Copy centroids to GPU
        float *device_centroids = static_cast<float *>(m_device_centroids);
        for (int i = 0; i < m_number_of_clusters; ++i)
        {
            for (int j = 0; j < m_number_of_dimensions; ++j)
            {
                CHECK_CUDA_ERROR(cudaMemcpy(&device_centroids[i * m_number_of_dimensions + j],
                                            &m_host_centroids[i][j],
                                            sizeof(float),
                                            cudaMemcpyHostToDevice));
            }
        }
    }

    void SharedGPU::runKMeans()
    {
        const int threadPerBlock = 256;
        const int blocksForPoints = (m_number_of_data_points + threadPerBlock - 1) / threadPerBlock;
        const int blocksForCentroids = (m_number_of_clusters + threadPerBlock - 1) / threadPerBlock;

        // Cast to proper types for kernel calls
        float *device_data = static_cast<float *>(m_device_data);
        float *device_centroids = static_cast<float *>(m_device_centroids);
        int *device_cluster_assignments = static_cast<int *>(m_device_cluster_assignments);
        int *device_cluster_counts = static_cast<int *>(m_device_cluster_counts);

        // Temporary buffer for centroids
        float *d_new_centroids;
        CHECK_CUDA_ERROR(cudaMalloc(&d_new_centroids, m_number_of_clusters * m_number_of_dimensions * sizeof(float)));

        // Temporary buffer for cluster counts
        for (int iter = 0; iter < m_max_iterations; ++iter)
        {
            // Step 1, assign points to clusters
            assignClustersKernel<<<blocksForPoints, threadPerBlock>>>(device_data, device_centroids, device_cluster_assignments,
                                                                      m_number_of_data_points, m_number_of_dimensions, m_number_of_clusters);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // Step 2, reset new centroids and cluster sizes
            CHECK_CUDA_ERROR(cudaMemset(d_new_centroids, 0, m_number_of_clusters * m_number_of_dimensions * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(device_cluster_counts, 0, m_number_of_clusters * sizeof(int)));

            // Step 3, accumulate points for each cluster
            updateCentroidsKernel<<<blocksForPoints, threadPerBlock>>>(device_data, d_new_centroids, device_cluster_assignments,
                                                                       device_cluster_counts, m_number_of_data_points,
                                                                       m_number_of_dimensions, m_number_of_clusters);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // Step 4. Normalize centroids
            normalizeCentroidsKernel<<<blocksForCentroids, threadPerBlock>>>(d_new_centroids, device_cluster_counts,
                                                                             m_number_of_dimensions, m_number_of_clusters);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // Check for convergence
            bool converged = true;
            std::vector<float> old_centroid(m_number_of_dimensions);
            std::vector<float> new_centroid(m_number_of_dimensions);

            for (int i = 0; i < m_number_of_clusters; ++i)
            {
                // Copy current centroids
                CHECK_CUDA_ERROR(cudaMemcpy(old_centroid.data(),
                                            &device_centroids[i * m_number_of_dimensions],
                                            m_number_of_dimensions * sizeof(float),
                                            cudaMemcpyDeviceToHost));

                // Copy new centroids
                CHECK_CUDA_ERROR(cudaMemcpy(new_centroid.data(),
                                            &d_new_centroids[i * m_number_of_dimensions],
                                            m_number_of_dimensions * sizeof(float),
                                            cudaMemcpyDeviceToHost));

                // Check if there's significant change
                for (int j = 0; j < m_number_of_dimensions; ++j)
                {
                    if (std::abs(old_centroid[j] - new_centroid[j]) > 1e-6)
                    {
                        converged = false;
                        break;
                    }
                }

                if (!converged)
                    break;
            }

            if (converged)
            {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }

            // copy new centroids to centroids
            CHECK_CUDA_ERROR(cudaMemcpy(device_centroids, d_new_centroids,
                                        m_number_of_clusters * m_number_of_dimensions * sizeof(float),
                                        cudaMemcpyDeviceToDevice));
        }

        // Free mem buffer
        CHECK_CUDA_ERROR(cudaFree(d_new_centroids));

        // Copy results back to host device
        copyResultsToHost();
    }

    std::vector<int> &SharedGPU::getClusterAssignments()
    {
        return m_host_cluster_assignments;
    }

    std::vector<std::vector<float>> &SharedGPU::getClusterCentroids()
    {
        return m_host_centroids;
    }

    // Save results to the csv
    void SharedGPU::saveResultsToCSV(const std::string &filename, const std::vector<std::string> &songIds)
    {
        // Open file
        std::ofstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Write header
        file << "songId,cluster" << std::endl;

        // Write data
        for (std::size_t index = 0; index < songIds.size() && index < m_host_cluster_assignments.size(); ++index)
        {
            file << songIds[index] << "," << m_host_cluster_assignments[index] << std::endl;
        }

        file.close();
    }

    // Allocate memory on GPU
    void SharedGPU::allocateMemory()
    {
        // Allocate memory on the GPU
        float *device_data;
        float *device_centroids;
        int *device_cluster_assignments;
        int *device_cluster_counts;

        CHECK_CUDA_ERROR(cudaMalloc(&device_data, m_number_of_data_points * m_number_of_dimensions * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&device_centroids, m_number_of_clusters * m_number_of_dimensions * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&device_cluster_assignments, m_number_of_data_points * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&device_cluster_counts, m_number_of_clusters * sizeof(int)));

        // Store as void* in the class
        m_device_data = device_data;
        m_device_centroids = device_centroids;
        m_device_cluster_assignments = device_cluster_assignments;
        m_device_cluster_counts = device_cluster_counts;

        // Resize host copies
        m_host_cluster_assignments.resize(m_number_of_data_points, 0);
    }

    // Free memory on GPU
    void SharedGPU::freeMemory()
    {
        // Free memory on the GPU
        if (m_device_data)
            CHECK_CUDA_ERROR(cudaFree(m_device_data));
        if (m_device_centroids)
            CHECK_CUDA_ERROR(cudaFree(m_device_centroids));
        if (m_device_cluster_assignments)
            CHECK_CUDA_ERROR(cudaFree(m_device_cluster_assignments));
        if (m_device_cluster_counts)
            CHECK_CUDA_ERROR(cudaFree(m_device_cluster_counts));

        // Clear host copies
        m_device_data = nullptr;
        m_device_centroids = nullptr;
        m_device_cluster_assignments = nullptr;
        m_device_cluster_counts = nullptr;
    }

    // Copy data to device
    void SharedGPU::copyDataToDevice(const std::vector<std::vector<float>> &data)
    {
        // Flatten data for more efficient copying
        std::vector<float> flatData(m_number_of_data_points * m_number_of_dimensions);
        for (int i = 0; i < m_number_of_data_points; ++i)
        {
            for (int j = 0; j < m_number_of_dimensions; ++j)
            {
                flatData[i * m_number_of_dimensions + j] = data[i][j];
            }
        }

        // Copy data to GPU
        float *device_data = static_cast<float *>(m_device_data);
        CHECK_CUDA_ERROR(cudaMemcpy(device_data, flatData.data(),
                                    m_number_of_data_points * m_number_of_dimensions * sizeof(float),
                                    cudaMemcpyHostToDevice));
    }

    void SharedGPU::copyResultsToHost()
    {
        // Copy cluster assignments
        int *device_cluster_assignments = static_cast<int *>(m_device_cluster_assignments);
        CHECK_CUDA_ERROR(cudaMemcpy(m_host_cluster_assignments.data(), device_cluster_assignments,
                                    m_number_of_data_points * sizeof(int),
                                    cudaMemcpyDeviceToHost));

        // Prepare host centroids
        m_host_centroids.resize(m_number_of_clusters, std::vector<float>(m_number_of_dimensions, 0.0f));

        // Copy centroids
        float *device_centroids = static_cast<float *>(m_device_centroids);

        // Copy centroids from device to host
        for (int i = 0; i < m_number_of_clusters; ++i)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(m_host_centroids[i].data(),
                                        &device_centroids[i * m_number_of_dimensions],
                                        m_number_of_dimensions * sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
    }
}