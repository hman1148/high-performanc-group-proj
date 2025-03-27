//
// Created by Joe Brijs on 3/21/25.
//

#pragma once

#include "IAlgorithm.h"
#include <iostream>

class DistributedGpu : public IAlgorithm
{
public:
    void run(std::vector<SpotifyGenreRevealParty::Point> &data, int, size_t, int, double) override
    {
        std::cout << "Running Disributed GPU implementation." << std::endl;
    }
};
