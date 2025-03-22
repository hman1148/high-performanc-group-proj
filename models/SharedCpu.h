//
// Created by Joe Brijs on 3/21/25.
//

#ifndef SHAREDCPU_H
#define SHAREDCPU_H

#include "IAlgorithm.h"
#include <iostream>

class SharedCpu : public IAlgorithm {
public:
    void run(std::vector<std::vector<float>>) override {
        std::cout << "Running Shared CPU implementation." << std::endl;
    }
};

#endif // SHAREDCPU_H