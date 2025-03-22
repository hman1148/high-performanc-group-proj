//
// Created by Joe Brijs on 3/21/25.
//

#ifndef SERIAL_H
#define SERIAL_H

#include "IAlgorithm.h"
#include <iostream>

class Serial : public IAlgorithm {
public:
    void run() override {
        std::cout << "Running Serial implementation." << std::endl;
    }
};

#endif // SERIAL_H