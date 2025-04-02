//
// Created by Joe Brijs on 3/21/25.
//

#ifndef ALGORITHMFACTORY_H
#define ALGORITHMFACTORY_H

#include "IAlgorithm.h"
#include <memory>

std::unique_ptr<IAlgorithm> createAlgorithm(int choice);

#endif // ALGORITHMFACTORY_H
