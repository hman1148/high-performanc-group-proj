//
// Created by Joe Brijs on 3/21/25.
//

#ifndef DISTRIBUTEDGPU_CPP_H
#define DISTRIBUTEDGPU_CPP_H

#include "IAlgorithm.h"
#include <iostream>

class DistributedGpu : public IAlgorithm {
public:
    void run(std::vector<SpotifyGenreRevealParty::Point>, int, size_t) override {
        std::cout << "Running Disributed GPU implementation." << std::endl;
    }
};

#endif //DISTRIBUTEDGPU_CPP_H
