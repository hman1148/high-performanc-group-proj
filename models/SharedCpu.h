#ifndef SHAREDCPU_H
#define SHAREDCPU_H

#include "IAlgorithm.h"
#include <iostream>
#include <omp.h>

class SharedCpu : public IAlgorithm {
public:
    void run(std::vector<SpotifyGenreRevealParty::Point> points, int k, size_t dimensions, int maxIterations, double tolerance) override {
        std::cout << "Running Shared CPU implementation with OpenMP." << std::endl;
        sharedMemoryParallelCpu(points, k, dimensions, maxIterations, tolerance);
    }

private:
    static void sharedMemoryParallelCpu(std::vector<SpotifyGenreRevealParty::Point>& points, const int k, const int dimensions, const int maxIterations, const double tolerance) {
        auto centroids = generateCentroids(k, dimensions);

        for (int iter = 0; iter < maxIterations; ++iter) {
            std::cout << "Iteration " << iter + 1 << std::endl;

            auto prevCentroids = centroids;

            // Parallelized points assignment using OpenMP
            assignPointsToClusters(points, centroids, k);

            // Parallelized centroid computation using OpenMP
            computeCentroids(points, centroids, k);

            if (hasConverged(prevCentroids, centroids, tolerance)) {
                std::cout << "Convergence reached after " << iter + 1 << " iterations." << std::endl;
                break;
            }
        }
    }

    static void assignPointsToClusters(std::vector<SpotifyGenreRevealParty::Point>& points,
                                        const std::vector<SpotifyGenreRevealParty::Point>& centroids, const int k) {
        #pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i) {
            double minDist = __DBL_MAX__;
            int clusterId = -1;

            for (int j = 0; j < k; ++j) {
                double dist = centroids[j].calculateDistance(points[i]);
                if (dist < minDist) {
                    minDist = dist;
                    clusterId = j;
                }
            }

            points[i].minDist = minDist;
            points[i].clusterId = clusterId;
        }
    }

    static void computeCentroids(std::vector<SpotifyGenreRevealParty::Point>& points,
                                  std::vector<SpotifyGenreRevealParty::Point>& centroids, int k) {
        std::vector<int> nPoints(k, 0);
        std::vector<std::vector<double>> sum(k, std::vector<double>(centroids[0].features.size(), 0.0));

        // Parallelized sum computation using OpenMP
        #pragma omp parallel for reduction(+ : nPoints[:k])
        for (size_t i = 0; i < points.size(); ++i) {
            int clusterId = points[i].clusterId;
            nPoints[clusterId]++;

            for (size_t featureIndex = 0; featureIndex < points[i].features.size(); ++featureIndex) {
                #pragma omp atomic
                sum[clusterId][featureIndex] += points[i].features[featureIndex];
            }

            points[i].minDist = __DBL_MAX__;
        }

        // Compute new centroids
        #pragma omp parallel for
        for (int clusterId = 0; clusterId < k; ++clusterId) {
            if (nPoints[clusterId] > 0) {
                for (size_t featureIndex = 0; featureIndex < centroids[clusterId].features.size(); ++featureIndex) {
                    centroids[clusterId].features[featureIndex] = sum[clusterId][featureIndex] / nPoints[clusterId];
                }
            }
        }
    }

    static bool hasConverged(const std::vector<SpotifyGenreRevealParty::Point>& prevCentroids,
                               const std::vector<SpotifyGenreRevealParty::Point>& centroids,
                               const double tolerance) {
        bool converged = true;

        #pragma omp parallel for reduction(&& : converged)
        for (size_t i = 0; i < centroids.size(); ++i) {
            if (const double dist = prevCentroids[i].calculateDistance(centroids[i]); dist > tolerance) {
                converged = false;
            }
        }

        return converged;
    }

    static std::vector<SpotifyGenreRevealParty::Point> generateCentroids(const int k, const int numFeatures) {
        std::vector<SpotifyGenreRevealParty::Point> centroids;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < k; i++) {
            std::vector<float> feats(numFeatures);
            for (int j = 0; j < numFeatures; j++) {
                feats[j] = dis(gen);
            }
            centroids.emplace_back(feats);
        }

        return centroids;
    }
};

#endif // SHAREDCPU_H
