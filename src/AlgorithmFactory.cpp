//
// Created by Joe Brijs on 3/21/25.
//
#include "../models/AlgorithmFactory.h"
#include "../models/Serial.h"
#include "../models/SharedCpu.hpp"
#include "../models/DistributedCpu.h"
#include "../models/DistributedGpu.h"

#include "../gpu/SharedGPU.cuh"

#include <memory>
#include <stdexcept>
#include <cstdint>
#include "../tools/utils.h"

std::unique_ptr<IAlgorithm> createAlgorithm(int choice)
{
    switch (choice)
    {
    case 1:
        return std::make_unique<Serial>();
    case 2:
        return std::make_unique<SharedCpu>();
    case 3:
        return std::make_unique<SpotifyGenreRevealParty::SharedGPU>(10, 100);
    case 4:
        return std::make_unique<DistributedCpu>();
    case 5:
        return std::make_unique<DistributedGpu>();
    default:
        throw std::invalid_argument("Invalid choice. Please enter a number between 1 and 5.");
    }
}