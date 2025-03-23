//
// Created by Joe Brijs on 3/21/25.
//

#ifndef IALGORITHM_H
#define IALGORITHM_H

#include <vector>

#include "Point.h"

class IAlgorithm {
  public:
    virtual void run(std::vector<SpotifyGenreRevealParty::Point>, int, size_t) = 0;
    virtual ~IAlgorithm() = default;
};

#endif //IALGORITHM_H
