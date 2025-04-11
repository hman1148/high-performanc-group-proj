//
// Created by Joe Brijs on 3/21/25.
//
#include "../models/AlgorithmFactory.h"
#include "../models/Serial.h"
#include "../models/DistributedCpu.h"
#include "../models/DistributedGpu.h"
#include "../models/SharedCpu.h"

#include "../gpu/SharedGPU.cuh"
#include "../gpu/GlobalGPU.cuh"

#include <stdexcept>

std::unique_ptr<IAlgorithm> createAlgorithm(int choice, const int &k, const int &max_iterations)
{
    switch (choice)
    {
    case 1:
        return std::make_unique<Serial>();
    case 2:
        return std::make_unique<SharedCpu>();
    case 3:
        return std::make_unique<DistributedCpu>();
    case 4:
        return std::make_unique<SpotifyGenreRevealParty::SharedGPU>(k, max_iterations);
    case 5:
        return std::make_unique<SpotifyGenreRevealParty::GlobalGPU>(k, max_iterations);
    default:
        throw std::invalid_argument("Invalid choice. Please enter a number between 1 and 5.");
    }
}