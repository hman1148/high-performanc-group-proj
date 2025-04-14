//
// Created by Joe Brijs on 3/21/25.
//

#ifndef ALGORITHMFACTORY_H
#define ALGORITHMFACTORY_H

#include "IAlgorithm.h"
#include <memory>
#include <stdexcept>

std::unique_ptr<IAlgorithm> createAlgorithm(int choice, const int &k, const int &max_iterations);

#endif // ALGORITHMFACTORY_H
