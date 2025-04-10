#include <iostream>
#include "../models/AlgorithmFactory.h"
#include <string>
#include <vector>
#include <mpi.h>

#include "../tools/SpotifyFrameReader.h"
#include "../tools/SpotifyFrameUtils.h"
using namespace SpotifyGenreRevealParty;

void printUsage() {
    std::cerr << "Usage: <executable> <k> <max_iterations> <tolerance> <dataset_percentage> <algorithm_id>" <<
            std::endl;
    std::cerr << "  k: Number of clusters (positive integer)" << std::endl;
    std::cerr << "  max_iterations: Maximum number of iterations (positive integer)" << std::endl;
    std::cerr << "  tolerance: Convergence tolerance (positive float)" << std::endl;
    std::cerr << "  dataset_percentage: What percentage of the data set to use (1 to 100)" << std::endl;
    std::cerr << "  algorithm_id: ID of the algorithm to run (1 to 5)" << std::endl;
    std::cerr << "\tSerial = 1\n"
            "\tShared memory parallel CPU = 2\n"
            "\tDistributed memory parallel CPU = 3\n"
            "\tShared Memory Parallel GPU = 4\n"
            "\tDistributed Memory Parallel GPU = 5"
            << std::endl;
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_initialized = 0;
    int rank = 0;

    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    // Check if the number of arguments is correct (5 total: executable + k + max_iterations + tolerance + algorithm_id)
    if (argc != 6) {
        printUsage();
        return 1;
    }

    // Parse arguments
    int k = std::stoi(argv[1]);
    int max_iterations = std::stoi(argv[2]);
    double tolerance = std::stod(argv[3]);
    int size = std::stoi(argv[4]);
    int algorithm_id = std::stoi(argv[5]);

    if (rank == 0) {
        std::cout << "Validating arguments..." << std::endl;
    }
    // Validate k, max_iterations, tolerance, and algorithm_id
    if (k <= 0 || max_iterations <= 0 || tolerance <= 0 || size < 1 || size > 100 || algorithm_id < 1 || algorithm_id >
        5) {
        printUsage();
        return 1;
    }

    const std::string csvFile = "../data/tracks_features.csv";
    const std::string binaryCache = "../data/points_cache.bin";

    std::vector<Point> points;

    if (rank == 0) {
        try {
            points = getOrLoadPoints(csvFile, binaryCache);
        } catch (const std::exception &e) {
            std::cerr << "Failed to load data: " << e.what() << std::endl;
            return 1;
        }
    }


    const size_t totalSize = points.size();
    const auto newSize = static_cast<size_t>((size / 100.0) * totalSize); // cast to ensure decimal math
    std::vector<Point> subset(points.begin(), points.begin() + newSize);
    points = std::move(subset); // replace original with subset

    if (rank == 0) {
        std::cout << "Scaling data ..." << std::endl;
    }
    minMaxScale(points); // Min-max scale based on Points


    size_t dimension = points.empty() ? 0 : points[0].features.size();

    if (dimension == 0 && rank == 0) {
        std::cerr << "Error: No data available to determine the vector dimension." << std::endl;
        return 1;
    }

    // Create the algorithm and run it with the specified parameters
    try {
        const std::unique_ptr<IAlgorithm> algorithm = createAlgorithm(algorithm_id, k, max_iterations);
        algorithm->run(points, k, dimension, max_iterations, tolerance); // Pass Points instead of raw data
    } catch (const std::exception &e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        return 1;
    }

    // Kill MPI
    MPI_Finalize();
    return 0;
}
