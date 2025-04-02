#include "SharedCpu.hpp"

void SharedCpu::run(std::vector<SpotifyGenreRevealParty::Point> &points, int k, size_t dimensions, int maxIterations, double tolerance)
{
    std::cout << "Running Shared CPU implementation with OpenMP." << std::endl;
    this->sharedMemoryParallelCpu(points, k, dimensions, maxIterations, tolerance);
}

void SharedCpu::sharedMemoryParallelCpu(std::vector<SpotifyGenreRevealParty::Point> &points, const int k, const int dimensions, const int maxIterations, const double tolerance)
{
    auto centroids = generateCentroids(k, dimensions);

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        std::cout << "Iteration " << iter + 1 << std::endl;

        auto prevCentroids = centroids;

        // Parallelized points assignment using OpenMP
        assignPointsToClusters(points, centroids, k);

        // Parallelized centroid computation using OpenMP
        computeCentroids(points, centroids, k);
    }
}

void SharedCpu::assignPointsToClusters(std::vector<SpotifyGenreRevealParty::Point> &points,
                                       const std::vector<SpotifyGenreRevealParty::Point> &centroids, const int k)
{
#pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i)
    {
        double minDist = __DBL_MAX__;
        int clusterId = -1;

        for (int j = 0; j < k; ++j)
        {
            double dist = centroids[j].calculateDistance(points[i]);
            if (dist < minDist)
            {
                minDist = dist;
                clusterId = j;
            }
        }

        points[i].minDist = minDist;
        points[i].clusterId = clusterId;
    }
}

void SharedCpu::computeCentroids(std::vector<SpotifyGenreRevealParty::Point> &points,
                                 std::vector<SpotifyGenreRevealParty::Point> &centroids, int k)
{
    // This will hold the number of points in each cluster
    std::vector<int> nPoints(k, 0);

    // This will hold the sum of features for each cluster
    std::vector<std::vector<double>> sum(k, std::vector<double>(centroids[0].features.size(), 0.0));

// Parallelized sum computation using OpenMP
#pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i)
    {
        int clusterId = points[i].clusterId;

// Update the number of points in the cluster (thread-private)
#pragma omp atomic
        nPoints[clusterId]++;

        // Update the sum of the features (atomic per feature for each cluster)
        for (size_t featureIndex = 0; featureIndex < points[i].features.size(); ++featureIndex)
        {
#pragma omp atomic
            sum[clusterId][featureIndex] += points[i].features[featureIndex];
        }

        // Min distance reset
        points[i].minDist = __DBL_MAX__;
    }

    // After parallel loop, compute centroids
    for (int i = 0; i < k; ++i)
    {
        if (nPoints[i] > 0)
        {
            for (size_t featureIndex = 0; featureIndex < centroids[0].features.size(); ++featureIndex)
            {
                centroids[i].features[featureIndex] = sum[i][featureIndex] / nPoints[i];
            }
        }
    }
}

std::vector<SpotifyGenreRevealParty::Point> SharedCpu::generateCentroids(const int k, const int numFeatures)
{
    std::vector<SpotifyGenreRevealParty::Point> centroids;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < k; i++)
    {
        std::vector<float> feats(numFeatures);
        for (int j = 0; j < numFeatures; j++)
        {
            feats[j] = dis(gen);
        }
        centroids.emplace_back(feats);
    }

    return centroids;
};
