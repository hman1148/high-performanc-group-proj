#include <iostream>
#include "../models/AlgorithmFactory.h"
#include <string>
#include <vector>

#include "../tools/SpotifyFrameReader.h"
#include "../tools/SpotifyFrameUtils.h"

int main(int argc, char* argv[]) {

    if (argc < 2 || argc > 4) {
        std::cerr << "Error: Please provide 1 to 3 arguments for k, max_iterations, and tolerance" << std::endl;
        return 1;
    }

    int k = 0;
    try {
        k = std::stoi(argv[1]);
        if (k <= 0) {
            std::cerr << "Error: k must be a positive integer." << std::endl;
            return 1;
        }
        std::cout << "You entered k = " << k << std::endl;
    } catch (const std::invalid_argument&) {
        std::cerr << "Error: Invalid input. Please provide a valid integer for k." << std::endl;
        return 1;
    } catch (const std::out_of_range&) {
        std::cerr << "Error: Input out of range for an integer." << std::endl;
        return 1;
    }

    // Default values
    int max_iterations = 100; // Default max_iterations
    double tolerance = 0.0001; // Default tolerance

    // Parse max_iterations if provided (argc == 3 or 4)
    if (argc >= 3) {
        try {
            max_iterations = std::stoi(argv[2]);
            if (max_iterations <= 0) {
                std::cerr << "Error: max_iterations must be a positive integer." << std::endl;
                return 1;
            }
            std::cout << "You entered max_iterations = " << max_iterations << std::endl;
        } catch (const std::invalid_argument&) {
            std::cerr << "Error: Invalid input. Please provide a valid integer for max_iterations." << std::endl;
            return 1;
        } catch (const std::out_of_range&) {
            std::cerr << "Error: Input out of range for an integer." << std::endl;
            return 1;
        }
    }

    // Parse tolerance if provided (argc == 4)
    if (argc == 4) {
        try {
            tolerance = std::stod(argv[3]);
            if (tolerance <= 0) {
                std::cerr << "Error: tolerance must be a positive number." << std::endl;
                return 1;
            }
            std::cout << "You entered tolerance = " << tolerance << std::endl;
        } catch (const std::invalid_argument&) {
            std::cerr << "Error: Invalid input. Please provide a valid number for tolerance." << std::endl;
            return 1;
        } catch (const std::out_of_range&) {
            std::cerr << "Error: Input out of range for a double." << std::endl;
            return 1;
        }
    }

    // Path to your CSV file (relative to working directory)
    const std::string fileName = "../data/tracks_features.csv";

    // Read frames from the CSV
    const auto frames = SpotifyGenreRevealParty::SpotifyFrameReader::readCSV(fileName);

    std::cout << "Number of frames: " << frames.size() << std::endl;

    // Prepare the data for K-means as a vector of Point objects
    std::vector<SpotifyGenreRevealParty::Point> points;

    // Extract features from each shared_ptr<SpotifyFrame> and store as Point objects
    for (const auto& framePtr : frames) {
        if (framePtr) {
            // Dereference the shared_ptr to access the actual object
            auto features = SpotifyGenreRevealParty::extractFeatures(*framePtr);
            points.emplace_back(features);  // Create a Point with the feature vector
        }
    }

    SpotifyGenreRevealParty::minMaxScale(points);  // Min-max scale based on Points

    // Output a few of the scaled data points to ensure the data is ready
    std::cout << "Scaled data (first 3 points):" << std::endl;
    for (size_t i = 0; i < std::min(points.size(), static_cast<size_t>(3)); ++i) {
        for (float j : points[i].features) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    size_t dimension = points.empty() ? 0 : points[0].features.size();

    if (dimension == 0) {
        std::cerr << "Error: No data available to determine the vector dimension." << std::endl;
        return 1;
    }

    std::cout << "Dimension of feature vectors: " << dimension << std::endl;

    // Loop through different i values for the different implementations
    for (int i = 1; i <= 5; i++) {
        try {
            // Create the algorithm and pass in the prepared data (now as Points)
            const std::unique_ptr<IAlgorithm> algorithm = createAlgorithm(i);

            algorithm->run(points, k, dimension, max_iterations, tolerance);  // Pass Points instead of raw data
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}
