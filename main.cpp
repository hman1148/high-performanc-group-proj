#include <iostream>
#include <string>

#include "tools/SpotifyFrameReader.h"

int main()
{
    // Path to your CSV file (relative to working directory)
    std::string fileName = "../data/tracks_features.csv";

    // Read frames from the CSV
    auto frames = SpotifyGenreRevealParty::SpotifyFrameReader::readCSV(fileName);

    std:: cout << "Number of frames: " << frames.size() << std::endl;

    return 0;
}
