#include <iostream>
#include "../models/AlgorithmFactory.h"
#include <string>
#include <vector>

#include "../tools/SpotifyFrameReader.h"
#include "../tools/SpotifyFrameUtils.h"  // Assuming the utils are in this file

// Assuming the `SpotifyFrame` and `SpotifyFrameReader` classes are already defined in `SpotifyGenreRevealParty` namespace

int main() {
    // Path to your CSV file (relative to working directory)
    const std::string fileName = "../data/tracks_features.csv";

    // Read frames from the CSV
    const auto frames = SpotifyGenreRevealParty::SpotifyFrameReader::readCSV(fileName);

    std::cout << "Number of frames: " << frames.size() << std::endl;

    // Prepare the data for K-means
    std::vector<std::vector<float>> data;

    // Extract features from each shared_ptr<SpotifyFrame>
    for (const auto& framePtr : frames) {
        if (framePtr) {
            // Dereference the shared_ptr to access the actual object
            data.push_back(SpotifyGenreRevealParty::extractFeatures(*framePtr));
        }
    }

    SpotifyGenreRevealParty::minMaxScale(data);

    // Output a few of the scaled data points to ensure the data is ready
    std::cout << "Scaled data (first 3 points):" << std::endl;
    for (size_t i = 0; i < std::min(data.size(), static_cast<size_t>(3)); ++i) {
        for (float j : data[i]) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    // You can loop through different i values for the different implementations
    for (int i = 1; i <= 5; i++) {
        try {
            // Create the algorithm and pass in the prepared data
            const std::unique_ptr<IAlgorithm> algorithm = createAlgorithm(i);

            algorithm->run(data);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}