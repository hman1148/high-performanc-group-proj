#pragma once

#include "IAlgorithm.h"
#include <iostream>
#include "random"
#include <omp.h>

class SharedCpu : public IAlgorithm
{
public:
    void run(std::vector<SpotifyGenreRevealParty::Point> &points, int k, size_t dimensions, int maxIterations, double tolerance) override;

private:
    static void sharedMemoryParallelCpu(std::vector<SpotifyGenreRevealParty::Point> &points, const int k, const int dimensions, const int maxIterations, const double tolerance);

    static void assignPointsToClusters(std::vector<SpotifyGenreRevealParty::Point> &points,
                                       const std::vector<SpotifyGenreRevealParty::Point> &centroids, const int k);

    static void computeCentroids(std::vector<SpotifyGenreRevealParty::Point> &points,
                                 std::vector<SpotifyGenreRevealParty::Point> &centroids, int k);

    static std::vector<SpotifyGenreRevealParty::Point> generateCentroids(const int k, const int numFeatures);
};
