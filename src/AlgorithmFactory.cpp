//
// Created by Joe Brijs on 3/21/25.
//
#include "../models/AlgorithmFactory.h"
#include "../models/Serial.h"
#include "../models/SharedCpu.h"
#include "../models/SharedGpu.h"
#include "../models/DistributedCpu.h"
#include "../models/DistributedGpu.h"
#include <memory>
#include <stdexcept>
#include <cstdint>

std::unique_ptr<IAlgorithm> createAlgorithm(int choice) {
    switch (choice) {
        case 1: return std::make_unique<Serial>();
        case 2: return std::make_unique<SharedCpu>();
        case 3: return std::make_unique<SharedGpu>();
        case 4: return std::make_unique<DistributedCpu>();
        case 5: return std::make_unique<DistributedGpu>();
        default: throw std::invalid_argument("Invalid choice. Please enter a number between 1 and 5.");
    }
}