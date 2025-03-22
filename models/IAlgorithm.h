//
// Created by Joe Brijs on 3/21/25.
//

#ifndef IALGORITHM_H
#define IALGORITHM_H

#include <vector>

class IAlgorithm {
  public:
    virtual void run(std::vector<std::vector<float>>) = 0;
    virtual ~IAlgorithm() = default;
};

#endif //IALGORITHM_H
