#include <iostream>
#include "../models/AlgorithmFactory.h"
#include <string>

#include "../tools/SpotifyFrameReader.h"

int main()
{
    // Path to your CSV file (relative to working directory)
    std::string fileName = "../data/tracks_features.csv";

    // Read frames from the CSV
    auto frames = SpotifyGenreRevealParty::SpotifyFrameReader::readCSV(fileName);

    std:: cout << "Number of frames: " << frames.size() << std::endl;

    for (int i = 1; i <= 5; i++) {
        try {
            std::unique_ptr<IAlgorithm> algorithm = createAlgorithm(i);
            algorithm->run();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    return 0;
}
