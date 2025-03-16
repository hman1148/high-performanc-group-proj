//
// Created by Hunter Peart on 3/15/2025.
//

#include "SpotifyFrameReader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>

std::vector<std::shared_ptr<SpotifyGenreRevealParty::SpotifyFrame>>
SpotifyGenreRevealParty::SpotifyFrameReader::readCSV(const std::string &fileName)
{
    std::vector<std::shared_ptr<SpotifyGenreRevealParty::SpotifyFrame>> frames;
    std::ifstream file(fileName);
    std::string row;

    if (file.is_open())
    {
        std::getline(file, row); // Skip the header line

        std::vector<std::string> lines;
        while (std::getline(file, row))
        {
            lines.push_back(row);
        }
        file.close();

#pragma omp parallel for shared(frames) private(row)
        for (int i = 0; i < lines.size(); ++i)
        {
            auto frame = parseCSVLine(lines[i]);
#pragma omp critical
            frames.push_back(frame);
        }
    }
    else
    {
        throw std::runtime_error("Could not open file: " + fileName);
    }
    return frames;
}

// --- NEW: Proper CSV line parser that handles quotes ---
std::vector<std::string> parseCSV(const std::string& line) {
    std::vector<std::string> result;
    std::string token;
    bool in_quotes = false;

    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            result.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    result.push_back(token); // Add last field
    return result;
}

std::shared_ptr<SpotifyGenreRevealParty::SpotifyFrame>
SpotifyGenreRevealParty::SpotifyFrameReader::parseCSVLine(const std::string &line)
{
    auto cleanField = [](const std::string &s) -> std::string {
        std::string result = s;
        result.erase(std::remove(result.begin(), result.end(), '['), result.end());
        result.erase(std::remove(result.begin(), result.end(), ']'), result.end());
        result.erase(std::remove(result.begin(), result.end(), '\''), result.end());
        result.erase(std::remove(result.begin(), result.end(), '\"'), result.end());
        return result;
    };

    auto split = [](const std::string &s, char delimiter) -> std::vector<std::string> {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    };

    auto safe_stoi = [](const std::string &str) -> std::uint16_t {
        try {
            return static_cast<std::uint16_t>(std::stoi(str));
        } catch (...) {
            std::cerr << "Invalid argument for stoi: " << str << std::endl;
            return 0;
        }
    };

    auto safe_stof = [](const std::string &str) -> float {
        try {
            return std::stof(str);
        } catch (...) {
            std::cerr << "Invalid argument for stof: " << str << std::endl;
            return 0.0f;
        }
    };

    std::vector<std::string> fields = parseCSV(line);

    if (fields.size() < 24) {
        std::cerr << "Skipping line with insufficient fields: " << line << std::endl;
        return nullptr;
    }

    std::string id = fields[0];
    std::string name = fields[1];
    std::string album = fields[2];
    std::string album_id = fields[3];

    std::string artists_str = cleanField(fields[4]);
    std::string artist_ids_str = cleanField(fields[5]);

    std::vector<std::string> artists = split(artists_str, ';');
    std::vector<std::string> artist_ids = split(artist_ids_str, ';');

    std::uint16_t track_number = safe_stoi(fields[6]);
    std::uint16_t disc_number = safe_stoi(fields[7]);

    std::string explicit_raw = fields[8];
    std::transform(explicit_raw.begin(), explicit_raw.end(), explicit_raw.begin(), ::tolower);
    bool is_explicit = (explicit_raw == "true");

    float danceability = safe_stof(fields[9]);
    float energy = safe_stof(fields[10]);
    float key = safe_stof(fields[11]);
    float loudness = safe_stof(fields[12]);
    float mode = safe_stof(fields[13]);
    float speechiness = safe_stof(fields[14]);
    float acousticness = safe_stof(fields[15]);
    float instrumentalness = safe_stof(fields[16]);
    float liveness = safe_stof(fields[17]);
    float valence = safe_stof(fields[18]);
    float tempo = safe_stof(fields[19]);
    float duration_ms = safe_stof(fields[20]);
    float time_signature = safe_stof(fields[21]);
    std::uint16_t year = safe_stoi(fields[22]);
    std::string release_date = fields[23];

    return std::make_shared<SpotifyGenreRevealParty::SpotifyFrame>(
            id, name, album, album_id, artists, artist_ids, track_number, disc_number, is_explicit,
            danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness,
            liveness, valence, tempo, duration_ms, time_signature, year, release_date
    );
}
