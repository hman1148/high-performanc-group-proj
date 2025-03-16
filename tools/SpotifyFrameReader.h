//
// Created by Hunter Peart on 3/15/2025.
//
#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include "../models/SpotifyFrame.h"


namespace SpotifyGenreRevealParty
{
class SpotifyFrameReader
{
public:
    static std::vector<std::shared_ptr<SpotifyGenreRevealParty::SpotifyFrame>> readCSV(const std::string& fileName);

private:
    static std::shared_ptr<SpotifyGenreRevealParty::SpotifyFrame> parseCSVLine(const std::string& line);
};
}


