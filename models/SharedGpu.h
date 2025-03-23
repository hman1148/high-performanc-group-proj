//
// Created by Joe Brijs on 3/21/25.
//

#ifndef SHAREDGPU_H
#define SHAREDGPU_H

#include "IAlgorithm.h"
#include <iostream>

class SharedGpu : public IAlgorithm {
public:
    void run(std::vector<SpotifyGenreRevealParty::Point>, int, size_t) override {
        std::cout << "Running Shared CPU implementation." << std::endl;
    }
};

#endif //SHAREDGPU_H
