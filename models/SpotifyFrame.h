//
// Created by Hunter Peart on 3/15/2025.
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace SpotifyGenreRevealParty
{
class SpotifyFrame
{
public:
    const std::string& id;
    const std::string& name;
    const std::string& album;
    const std::string& album_id;
    const std::vector<std::string>& artists;
    const std::vector<std::string>& artist_ids;
    std::uint16_t track_number;
    std::uint16_t disc_number;
    bool is_explicit;
    float danceability;
    float energy;
    float key;
    float loudness;
    float mode;
    float speechiness;
    float acousticness;
    float instrumentalness;
    float liveness;
    float valence;
    float tempo;
    float duration_ms;
    float time_signature;
    std::uint16_t year;
    const std::string& release_date;

    SpotifyFrame(const std::string& id, const std::string& name, const std::string& album, const std::string& album_id,
                 const std::vector<std::string>& artists, const std::vector<std::string>& artist_ids, std::uint16_t track_number,
                 std::uint16_t disc_number, bool is_explicit,  float danceability, float energy, float key, float loudness, float mode, float speechiness,
                 float acousticness, float instrumentalness, float liveness, float valence, float tempo, float duration_ms,
                 float time_signature, std::uint16_t year, const std::string& release_date) : id(id), name(name), album(album),
          album_id(album_id), artists(artists), artist_ids(artist_ids), track_number(track_number), disc_number(disc_number),
          is_explicit(is_explicit), danceability(danceability), energy(energy), key(key), loudness(loudness), mode(mode), speechiness(speechiness),
          acousticness(acousticness), instrumentalness(instrumentalness), liveness(liveness), valence(valence), tempo(tempo),
          duration_ms(duration_ms), time_signature(time_signature), year(year), release_date(release_date) {}
};
}


