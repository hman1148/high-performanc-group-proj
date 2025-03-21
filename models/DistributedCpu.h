//
// Created by Joe Brijs on 3/21/25.
//

#ifndef DISTRIBUTEDCPU_H
#define DISTRIBUTEDCPU_H

#include "IAlgorithm.h"
#include <iostream>

class DistributedCpu : public IAlgorithm {
public:
    void run(std::vector<std::vector<float>>) override {
        std::cout << "Running Distributed CPU implementation." << std::endl;
    }
};

#endif // DISTRIBUTEDCPU_H