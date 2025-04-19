# Serial Implementation (`Serial.h`)

## Description

The serial version performs K-Means clustering on a single thread without any form of parallelism. The algorithm:
- Initializes `k` random centroids
- Iteratively assigns each point to the nearest centroid
- Computes new centroids as the mean of all points in each cluster
- Terminates early if centroids converge based on a tolerance value

Output is written to a CSV file after the algorithm completes.

## Pseudocode
```
function run_kmeans_serial(points, k, dimensions, maxIterations, tolerance):
    centroids = generate_random_centroids(k, dimensions)
    for iter in 1 to maxIterations:
        prevCentroids = centroids
        for point in points:
            point.clusterId = index of closest centroid
        for each cluster:
            update centroid as mean of assigned points
        if centroids are close to prevCentroids within tolerance:
            break
```
---

# Shared Memory CPU Implementation (`SharedCpu.h`)

## Description

This version is parallelized for multi-core CPUs using OpenMP. It accelerates the serial K-Means by:
- Parallelizing point-to-cluster assignment using `#pragma omp parallel for`
- Parallelizing centroid computation using local thread accumulations and `#pragma omp critical`
- Parallel reduction for convergence checking

This implementation keeps all data in shared memory and is suitable for running on a single machine with multiple cores.

## Pseudocode
```
function run_kmeans_shared_cpu(points, k, dimensions, maxIterations, tolerance):
    centroids = generate_random_centroids(k, dimensions)
    for iter in 1 to maxIterations:
        prevCentroids = centroids
        parallel for each point:
            point.clusterId = index of closest centroid
        parallel:
            compute local centroid sums and counts
            critical section:
                accumulate into global sums
        parallel for each cluster:
            update centroid = sum / count
        parallel reduction:
            check convergence
        if converged:
            break
```
---

# Parallel CUDA GPU Implementation (`SharedGPU.cu`)

## Description

This GPU version runs the K-Means algorithm using CUDA, without explicit use of CUDA shared memory. It performs:
- Parallel point assignment using a CUDA kernel (`assignClustersKernel`)
- Accumulation of feature sums using `atomicAdd` in `updateCentroidsKernel`
- Centroid normalization using `normalizeCentroidsKernel`
- Host manages convergence check by copying centroids from device to host

Each thread handles one point; operations are memory-bound and use global memory for all data transfers.

## Pseudocode
```
function run_kmeans_gpu_global(data, k, dimensions, maxIterations):
    allocate GPU memory for data, centroids, assignments
    initialize centroids randomly
    for iter in 1 to maxIterations:
        assignClustersKernel<<<blocks, threads>>>(data, centroids, assignments)
        cudaMemset(newCentroids, 0)
        cudaMemset(clusterCounts, 0)
        updateCentroidsKernel<<<blocks, threads>>>(data, newCentroids, assignments, clusterCounts)
        normalizeCentroidsKernel<<<blocks, threads>>>(newCentroids, clusterCounts)
        if centroids converged:
            break
        copy newCentroids to centroids
    copy results back to host
```
---

# Distributed Memory CPU Implementation (`DistributedCpu.h`)

## Description

This implementation distributes the dataset across multiple processes using MPI. Each process:
- Receives a chunk of the dataset via `MPI_Scatterv`
- Computes local assignments and partial sums
- Participates in a global reduction using `MPI_Allreduce`
- Updates centroids globally and checks convergence on rank 0
- Rank 0 gathers and saves final results using `MPI_Gatherv`

This approach allows for scalable processing across multiple nodes or cores in a cluster.

## Pseudocode
```
function run_kmeans_distributed_cpu(fullData, k, dimensions, maxIterations, tolerance):
    if rank == 0:
        flatten fullData and broadcast totalPoints
    scatter flatData across all ranks
    reconstruct localPoints from flat data
    if rank == 0:
        centroids = generate_random_centroids()
    broadcast centroids to all ranks
    for iter in 1 to maxIterations:
        assign each localPoint to nearest centroid
        compute local centroid sums and counts
        allreduce to get global sums and counts
        update centroids using global sums
        if rank == 0:
            check convergence
        broadcast doneYet
        if doneYet: break
    gather results back to rank 0 and write to CSV
```
---

# Distributed Memory GPU Implementation (`GlobalGPU.cu`)

## Description

This hybrid implementation combines MPI with CUDA to distribute K-Means across multiple GPUs. It:
- Distributes data to MPI ranks, each backed by a GPU
- Performs point assignments and centroid updates on each GPU
- Uses `MPI_Allreduce` to combine centroid sums across ranks
  - Can use CUDA-aware MPI (GPU buffers directly)
  - Or fallback to host memory transfers for reduction
- Handles convergence check either on host or via device buffer comparison
- Rank 0 writes output to CSV after gathering results

It's the most scalable implementation, targeting clusters with multiple GPUs.

## Pseudocode
```
function run_kmeans_distributed_gpu(data, k, dimensions, maxIterations):
    initialize MPI and GPU context
    distribute data using MPI
    allocate device memory for data, centroids, assignments
    initialize centroids on rank 0, broadcast
    for iter in 1 to maxIterations:
        assignPointsToGlobalCentroids<<<blocks>>>(data, centroids, assignments)
        cudaMemset(tempCentroids, 0)
        cudaMemset(clusterCounts, 0)
        updateCentroidsGlobalKernel<<<blocks>>>(...)
        if CUDA-aware MPI:
            MPI_Allreduce on device memory
            normalizeCentroidsGlobalKernel<<<blocks>>>
        else:
            cudaMemcpy to host
            MPI_Allreduce on host memory
            normalize on host, copy back to device
        check convergence
        if converged:
            break
    gather final cluster assignments and write CSV on rank 0
```