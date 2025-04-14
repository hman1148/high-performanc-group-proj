#include <cuda_runtime.h>
#include <mpi.h>

#include "../tools/utils.h"
#include "GlobalGPU.cuh"

// Cuda Error checking macro
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

__global__ void assignPointsToGlobalCentroids(float *data, float *centroids, int *assignments, int numPoints, int numDimensions, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        float minDist = __FLT_MAX__;
        int nearestCentroid = 0;

        for (int i = 0; i < k; ++i)
        {
            float distance = 0.0f;
            for (int j = 0; j < numDimensions; ++j)
            {
                float diff = data[idx * numDimensions + j] - centroids[i * numDimensions + j];
                distance += diff * diff;
            }

            if (distance < minDist)
            {
                minDist = distance;
                nearestCentroid = i;
            }
        }

        assignments[idx] = nearestCentroid;
    }
}

// CUDA Kernel for accumulating points for centroid updates
__global__ void updateCentroidsGlobalKernel(float *data, float *centroids, int *assignments, int *clusterCounts, int numPoints, int numDimensions, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints)
    {
        int clusterId = assignments[idx];

        // Atomic operation to accumulate the sum of points for each centroid
        for (int j = 0; j < numDimensions; ++j)
        {
            atomicAdd(&centroids[clusterId * numDimensions + j], data[idx * numDimensions + j]);
        }

        atomicAdd(&clusterCounts[clusterId], 1);
    }
}

// CUDA Kernel for normalzing centroids by cluster size
__global__ void normalizeCentroidsGlobalKernel(float *centroids, int *clusterCounts, int k, int numDimensions)
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

SpotifyGenreRevealParty::GlobalGPU::GlobalGPU(int clusters, int maxIterations)
    : m_number_of_clusters(clusters),
      m_max_iterations(maxIterations),
      m_rank(0),
      m_num_processes(0),
      m_num_points_per_process(0),
      m_cuda_aware_mpi(false),
      m_device_data(nullptr),
      m_device_centroids(nullptr),
      m_device_temp_centroids(nullptr),
      m_device_cluster_assignments(nullptr),
      m_device_cluster_count(nullptr)
{
    initializeMPI();
}

SpotifyGenreRevealParty::GlobalGPU::~GlobalGPU()
{
    this->freeGPUMemory();
}

void SpotifyGenreRevealParty::GlobalGPU::initializeMPI()
{
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);

    if (!initialized)
    {
        // Only print this warning from process 0
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            std::cerr << "WARNING: MPI was not initialized by main(). "
                      << "This may cause issues with proper MPI cleanup." << std::endl;
        }

        // Don't try to initialize MPI here - it should be done in main
        // instead, throw an error
        throw std::runtime_error("MPI should be initialized in main before creating algorithms");
    }

    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &this->m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->m_num_processes);

    // select gpu based on local rank
    char *local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = (local_rank_str != nullptr) ? atoi(local_rank_str) : this->m_rank;

    // Get number of available gpus
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount > 0)
    {
        int device = local_rank % deviceCount;
        CHECK_CUDA_ERROR(cudaSetDevice(device));

        if (this->m_rank == 0)
        {
            std::cout << "Using GPU device " << device << " on rank " << this->m_rank << std::endl;
        }
    }
    else
    {
        throw std::runtime_error("No GPU devices available");
    }

    this->m_cuda_aware_mpi = this->isMpiCudaAware();

    if (this->m_rank == 0)
    {
        std::cout << "CUDA-aware MPI: " << (this->m_cuda_aware_mpi ? "Enabled" : "Disabled") << std::endl;
    }
}

void SpotifyGenreRevealParty::GlobalGPU::freeGPUMemory()
{
    // Free all device memory
    if (this->m_device_data)
        CHECK_CUDA_ERROR(cudaFree(this->m_device_data));
    if (this->m_device_centroids)
        CHECK_CUDA_ERROR(cudaFree(this->m_device_centroids));
    if (this->m_device_temp_centroids)
        CHECK_CUDA_ERROR(cudaFree(this->m_device_temp_centroids));
    if (this->m_device_cluster_assignments)
        CHECK_CUDA_ERROR(cudaFree(this->m_device_cluster_assignments));
    if (this->m_device_cluster_count)
        CHECK_CUDA_ERROR(cudaFree(this->m_device_cluster_count));

    this->m_device_data = nullptr;
    this->m_device_centroids = nullptr;
    this->m_device_temp_centroids = nullptr;
    this->m_device_cluster_assignments = nullptr;
    this->m_device_cluster_count = nullptr;
}

bool SpotifyGenreRevealParty::GlobalGPU::isMpiCudaAware()
{
#ifdef MPIX_CUDA_AWARE_SUPPORT
    if (MPIX_CUDA_AWARE_SUPPORT)
    {
        return true;
    }
    else
    {
        return false;
    }
#elif defined(MVAPICH2_NUMVERSION) && (MVAPICH2_NUMVERSION >= 20000000)
    // MVAPICH2 2.0+ supports CUDA
    return true;
#elif defined(OPEN_MPI) && (OPEN_MPI >= 1005004)
    // OpenMPI 1.5.4+ might support CUDA
    char *env_var = getenv("OMPI_MCA_mpi_cuda_support");
    if (env_var && strcmp(env_var, "1") == 0)
    {
        return true;
    }
    return false;
#else
    // Default to false to be safe
    if (m_rank == 0)
    {
        std::cout << "CUDA-aware MPI detection not available, defaulting to CPU buffers" << std::endl;
    }
    return false;
#endif
}

void SpotifyGenreRevealParty::GlobalGPU::allocateGPUMemory()
{
    // Calculate memory requirements
    std::size_t data_size = this->m_num_points_per_process * this->m_number_of_dimensions * sizeof(float);
    std::size_t centroid_size = this->m_number_of_clusters * this->m_number_of_dimensions * sizeof(float);
    std::size_t assignment_size = this->m_num_points_per_process * sizeof(int);
    std::size_t count_size = this->m_number_of_clusters * sizeof(int);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&this->m_device_data, data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&this->m_device_centroids, centroid_size));
    CHECK_CUDA_ERROR(cudaMalloc(&this->m_device_temp_centroids, centroid_size));
    CHECK_CUDA_ERROR(cudaMalloc(&this->m_device_cluster_assignments, assignment_size));
    CHECK_CUDA_ERROR(cudaMalloc(&this->m_device_cluster_count, count_size));

    // Initialize assignments and counts to zero
    CHECK_CUDA_ERROR(cudaMemset(this->m_device_cluster_assignments, 0, assignment_size));
    CHECK_CUDA_ERROR(cudaMemset(this->m_device_cluster_count, 0, count_size));

    // Alocate host buffers
    this->m_host_cluster_assignments.resize(this->m_num_points_per_process, 0);

    if (this->m_rank == 0)
    {
        std::cout << "Allocated GPU memory: "
                  << (data_size + 2 * centroid_size + assignment_size + count_size) / (1024 * 1024)
                  << "MB on rank " << m_rank << std::endl;
    }
}

void SpotifyGenreRevealParty::GlobalGPU::distributeData(const std::vector<SpotifyGenreRevealParty::Point> &data)
{
    int total_points = static_cast<int>(data.size());

    // Calculate number of points per process
    this->m_num_points_per_process = total_points / this->m_num_processes;
    int remainder = total_points % this->m_num_processes;

    // Last process gets any remainder poitns
    if (this->m_rank == this->m_num_processes - 1)
    {
        this->m_num_points_per_process += remainder;
    }

    // Calculate start index for this process' data
    int start_idx = this->m_rank * (total_points / this->m_num_processes);

    this->m_host_flat_data.resize(this->m_num_points_per_process * this->m_number_of_dimensions);

    if (this->m_rank == 0)
    {
        // process 0 has the original data, extract its portion
        for (int i = 0; i < this->m_num_points_per_process; ++i)
        {
            for (int j = 0; j < this->m_number_of_dimensions; ++j)
            {
                this->m_host_flat_data[i * this->m_number_of_dimensions + j] = data[start_idx + i].features[j];
            }
        }

        // Send data to other processes
        for (int proc = 1; proc < this->m_num_processes; ++proc)
        {
            int proc_start_idx = proc * (total_points / this->m_num_processes);
            int proc_num_points = (proc == this->m_num_processes - 1) ? (total_points / this->m_num_processes) + remainder : (total_points / this->m_num_processes);

            // Last process gets any remainder points
            std::vector<float> proc_data(proc_num_points * this->m_number_of_dimensions);

            // Pack data for the process
            for (int i = 0; i < proc_num_points; ++i)
            {
                for (int j = 0; j < this->m_number_of_dimensions; ++j)
                {
                    proc_data[i * this->m_number_of_dimensions + j] = data[proc_start_idx + i].features[j];
                }
            }

            // Send data to the process
            MPI_Send(proc_data.data(), proc_num_points * this->m_number_of_dimensions, MPI_FLOAT, proc, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Receive data from process 0
        MPI_Recv(this->m_host_flat_data.data(), this->m_num_points_per_process * this->m_number_of_dimensions, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Copy data to gpu
    CHECK_CUDA_ERROR(cudaMemcpy(this->m_device_data, this->m_host_flat_data.data(), this->m_num_points_per_process * this->m_number_of_dimensions * sizeof(float), cudaMemcpyHostToDevice));
}

void SpotifyGenreRevealParty::GlobalGPU::initializeCentroids(int k, int dimensions)
{
    // Resize host centroids vector
    this->m_host_centroids.resize(k * dimensions);

    // process 0 initializes centroids
    if (this->m_rank == 0)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0, this->m_num_points_per_process - 1);

        // Randomly select k unique points from the data
        for (int i = 0; i < k; ++i)
        {
            int point_idx = dist(gen);
            for (int j = 0; j < dimensions; ++j)
            {
                this->m_host_centroids[i * dimensions + j] = this->m_host_flat_data[point_idx * dimensions + j];
            }
        }
    }

    // Broadcast the centroids to all processes
    MPI_Bcast(this->m_host_centroids.data(), k * dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Copy centroids to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(this->m_device_centroids, this->m_host_centroids.data(), k * dimensions * sizeof(float), cudaMemcpyHostToDevice));
}

void SpotifyGenreRevealParty::GlobalGPU::runDistributedKMeans(double tolerance)
{
    const int threadsPerBlock = 256;
    const int blocksForPoints = (this->m_num_points_per_process + threadsPerBlock - 1) / threadsPerBlock;
    const int blocksForCentroids = (this->m_number_of_clusters + threadsPerBlock - 1) / threadsPerBlock;

    // For convergence check
    std::vector<float> prev_centroids(this->m_number_of_clusters * this->m_number_of_dimensions);

    // Temporary buffers for reduction
    std::vector<float> local_centroids_sums(this->m_number_of_clusters * this->m_number_of_dimensions, 0.0f);
    std::vector<int> local_centroid_counts(this->m_number_of_clusters, 0);
    std::vector<float> global_centroids_sums(this->m_number_of_clusters * this->m_number_of_dimensions, 0.0f);
    std::vector<int> global_centroid_counts(this->m_number_of_clusters, 0);

    // Main KMeans loop
    for (int iter = 0; iter < this->m_max_iterations; ++iter)
    {
        // save current centroids for convergence check
        std::copy(this->m_host_centroids.begin(), this->m_host_centroids.end(), prev_centroids.begin());

        // Reset temporary buffers
        CHECK_CUDA_ERROR(cudaMemset(this->m_device_temp_centroids, 0, this->m_number_of_clusters * this->m_number_of_dimensions * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(this->m_device_cluster_count, 0, this->m_number_of_clusters * sizeof(int)));

        // Step 1: assign points to nearest centroid
        assignPointsToGlobalCentroids<<<blocksForPoints, threadsPerBlock>>>(this->m_device_data, this->m_device_centroids, this->m_device_cluster_assignments, this->m_num_points_per_process, this->m_number_of_dimensions, this->m_number_of_clusters);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Step 2. Update centroids (accumulate sum)
        updateCentroidsGlobalKernel<<<blocksForPoints, threadsPerBlock>>>(this->m_device_data, this->m_device_temp_centroids, this->m_device_cluster_assignments, this->m_device_cluster_count, this->m_num_points_per_process, this->m_number_of_dimensions, this->m_number_of_clusters);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Step 3. Global Reduction
        if (this->m_cuda_aware_mpi)
        {
            // CUDA aware MPI: use GPU memory directly
            MPI_Allreduce(MPI_IN_PLACE, this->m_device_temp_centroids, this->m_number_of_clusters * this->m_number_of_dimensions, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, this->m_device_cluster_count, this->m_number_of_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            // normalize centroids
            normalizeCentroidsGlobalKernel<<<blocksForCentroids, threadsPerBlock>>>(this->m_device_temp_centroids, this->m_device_cluster_count, this->m_number_of_clusters, this->m_number_of_dimensions);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // Copy to main centroids buffer
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_device_centroids, this->m_device_temp_centroids, this->m_number_of_clusters * this->m_number_of_dimensions * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        else
        {
            // Traditional MPI Approach - copy through host
            CHECK_CUDA_ERROR(cudaMemcpy(local_centroids_sums.data(), this->m_device_temp_centroids,
                                        this->m_number_of_clusters * this->m_number_of_dimensions * sizeof(float),
                                        cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaMemcpy(local_centroid_counts.data(), this->m_device_cluster_count,
                                        this->m_number_of_clusters * sizeof(int),
                                        cudaMemcpyDeviceToHost));

            // Reduce across processes
            MPI_Allreduce(local_centroids_sums.data(), global_centroids_sums.data(),
                          this->m_number_of_clusters * this->m_number_of_dimensions,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(local_centroid_counts.data(), global_centroid_counts.data(),
                          this->m_number_of_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            // Normalize centroids on CPU
            for (int i = 0; i < this->m_number_of_clusters; ++i)
            {
                if (global_centroid_counts[i] > 0)
                {
                    for (int j = 0; j < this->m_number_of_dimensions; ++j)
                    {
                        this->m_host_centroids[i * this->m_number_of_dimensions + j] =
                            global_centroids_sums[i * this->m_number_of_dimensions + j] / global_centroid_counts[i];
                    }
                }
            }

            // Copy updated centroids back to GPU
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_device_centroids, this->m_host_centroids.data(),
                                        this->m_number_of_clusters * this->m_number_of_dimensions * sizeof(float),
                                        cudaMemcpyHostToDevice));
        }

        // Step 4. check for convergence
        bool local_converged = true;

        // copy centroids back to host if using CUDA aware MPI
        if (this->m_cuda_aware_mpi)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_host_centroids.data(), this->m_device_centroids, this->m_number_of_clusters * this->m_number_of_dimensions * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // check if centorids have changed significantly
        for (int i = 0; i < this->m_number_of_clusters * this->m_number_of_dimensions; i++)
        {
            if (std::abs(prev_centroids[i] - this->m_host_centroids[i]) > tolerance)
            {
                local_converged = false;
                break;
            }
        }

        // Check if all proceses agree on convergence
        bool global_converged = false;
        MPI_Allreduce(&local_converged, &global_converged, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (global_converged)
        {
            if (this->m_rank == 0)
            {
                std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            }
            break;
        }

        // status update every 10 iters
        if (this->m_rank == 0 && (iter + 1) % 10 == 0)
        {
            std::cout << "Iteration " << iter + 1 << " completed." << std::endl;
        }
    }

    // Copy final cluster Assignments to host
    CHECK_CUDA_ERROR(cudaMemcpy(this->m_host_cluster_assignments.data(), this->m_device_cluster_assignments, this->m_num_points_per_process * sizeof(int), cudaMemcpyDeviceToHost));
}

void SpotifyGenreRevealParty::GlobalGPU::gatherResults(std::vector<SpotifyGenreRevealParty::Point> &data)
{
    // Gather all cluster assignments from all processes
    int total_points = static_cast<int>(data.size());

    // First gather the counts from each process
    std::vector<int> recv_counts(this->m_num_processes);
    std::vector<int> displacements(this->m_num_processes);

    MPI_Gather(&this->m_num_points_per_process, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements for each process
    if (this->m_rank == 0)
    {
        this->m_global_cluster_assignments.resize(total_points);
        int displacement = 0;

        for (int i = 0; i < this->m_num_processes; ++i)
        {
            displacements[i] = displacement;
            displacement += recv_counts[i];
        }
    }

    // Gather all assignments to rank 0
    MPI_Gatherv(this->m_host_cluster_assignments.data(), this->m_num_points_per_process, MPI_INT, this->m_rank == 0 ? this->m_global_cluster_assignments.data() : nullptr, recv_counts.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // Update original dat with cluster assignments (only on rank 0)
    if (this->m_rank == 0)
    {
        for (int i = 0; i < total_points; ++i)
        {
            data[i].clusterId = this->m_global_cluster_assignments[i];
        }

        // prlint cluster stats
        std::vector<int> cluster_sizes(this->m_number_of_clusters, 0);
        for (int i = 0; i < total_points; ++i)
        {
            cluster_sizes[this->m_global_cluster_assignments[i]]++;
        }

        std::cout << "\nClustering results: " << std::endl;
        std::cout << "Total points: " << total_points << std::endl;
        std::cout << "Clusters: " << this->m_number_of_clusters << std::endl;
        std::cout << "Cluster sizes: " << std::endl;
        for (int i = 0; i < this->m_number_of_clusters; ++i)
        {
            std::cout << "Cluster " << i << ": " << cluster_sizes[i] << " points" << std::endl;
        }
        std::cout << "----------------------------------------" << std::endl;
    }
}

void SpotifyGenreRevealParty::GlobalGPU::run(std::vector<SpotifyGenreRevealParty::Point> &data, int k, std::size_t dimensions, int maxIterations, double tolerance)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Rank " << this->m_rank << ": Starting GlobalGPU run " << std::endl;
    this->m_number_of_dimensions = static_cast<int>(dimensions);
    this->m_number_of_clusters = k;
    this->m_max_iterations = maxIterations;

    // Step 1. Calculate data distribution
    this->calculateDataDistribution(data.size());

    // Step 2: NOW allocate GPU memory
    this->allocateGPUMemory();

    // Step 3: Distribute the actual data
    this->distributeData(data);

    // Step 3: Initialize centroids
    this->initializeCentroids(k, this->m_number_of_dimensions);

    if (this->m_rank == 0)
    {
        std::cout << "Running KMeans with " << this->m_number_of_clusters
                  << " clusters and " << this->m_max_iterations
                  << " max iterations." << std::endl;
    }

    // Step 4: Run distributed KMeans
    this->runDistributedKMeans(tolerance);

    // Step 5: Gather results
    this->gatherResults(data);

    // Step 6: Save results to CSV on rank 0
    if (this->m_rank == 0)
    {
        std::cout << "Saving results to CSV..." << std::endl;
        std::vector<std::string> songIds;
        songIds.reserve(data.size());
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            songIds.push_back(std::to_string(i));
        }

        // Create output directory and save
        try
        {
            std::string output_dir = "output";
            int result = system(("mkdir -p " + output_dir).c_str());

            if (result != 0)
            {
                std::cerr << "Warning: Could not create output directory" << std::endl;
            }

            // Save clustering results
            std::vector<SpotifyGenreRevealParty::Point> centroids(this->m_number_of_clusters);
            for (int i = 0; i < this->m_number_of_clusters; ++i)
            {
                centroids[i].clusterId = i;
                centroids[i].features.resize(this->m_number_of_dimensions);

                for (int j = 0; j < this->m_number_of_dimensions; ++j)
                {
                    centroids[i].features[j] = this->m_host_centroids[i * this->m_number_of_dimensions + j];
                }
            }

            std::string detailedFile = "output/global_gpu_results.csv";
            utils::writePointsAndCentroidsToFile(data, centroids, detailedFile);
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Error creating output directory: " << ex.what() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "TIMING: GlobalGPU completed in " << elapsed_time.count() << " seconds" << std::endl;

    // Free GPU memory
    this->freeGPUMemory();
    std::cout << "Rank " << m_rank << ": GlobalGPU run complete" << std::endl;
}

void SpotifyGenreRevealParty::GlobalGPU::calculateDataDistribution(const size_t &total_points)
{
    this->m_num_points_per_process = total_points / this->m_num_processes;
    int remainder = total_points % this->m_num_points_per_process;

    // Last process gets any remainder points
    if (this->m_rank == this->m_num_processes - 1)
    {
        this->m_num_points_per_process += remainder;
    }

    if (this->m_rank == 0)
    {
        std::cout << "Data distribution: " << total_points << " total points, ";
        std::cout << "each process gets ~" << this->m_num_points_per_process << " points" << std::endl;
    }
}