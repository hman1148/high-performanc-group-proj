//
// Created by Joe Brijs on 3/21/25.
//

#ifndef IALGORITHM_H
#define IALGORITHM_H

class IAlgorithm {
  public:
    virtual void run() = 0;
    virtual ~IAlgorithm() = default;
};

#endif //IALGORITHM_H
