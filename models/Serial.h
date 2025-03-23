//
// Created by Joe Brijs on 3/21/25.
//

#ifndef SERIAL_H
#define SERIAL_H

#include "IAlgorithm.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>

class Serial : public IAlgorithm {
public:
    void run(std::vector<SpotifyGenreRevealParty::Point> data, int k, size_t dimensions) override {
        std::cout << "Running Serial implementation." << std::endl;
        serial(k, dimensions); // Call the serial function that generates centroids
    }

private:
    // The serial function that generates and prints centroids
    void serial(int k, int dimensions) {
        auto centroids = generateCentroids(k, dimensions);

        // Print centroids
        for (int i = 0; i < k; i++) {
            std::cout << "Centroid " << i + 1 << ": ";
            for (float value : *centroids[i]) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

    // Function to generate centroids
    std::vector<std::unique_ptr<std::vector<float>>> generateCentroids(int k, int dimensions) {
        std::vector<std::unique_ptr<std::vector<float>>> centroids;

        std::srand(static_cast<unsigned int>(std::time(0)));

        for (int i = 0; i < k; i++) {
            auto centroid = std::make_unique<std::vector<float>>();
            for (int j = 0; j < dimensions; j++) {
                centroid->push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            centroids.push_back(std::move(centroid));
        }
        return centroids;
    }
};

#endif // SERIAL_H
