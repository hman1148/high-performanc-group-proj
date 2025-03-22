//
// Created by Joe Brijs on 3/21/25.
//
#include "../tools/SpotifyFrameUtils.h"
#include <limits>
#include <algorithm>
# include <vector>

namespace SpotifyGenreRevealParty {

    // Function definitions
    std::vector<float> extractFeatures(const SpotifyFrame& frame) {
        return {
            frame.danceability,
            frame.energy,
            frame.tempo,
            frame.loudness,
            frame.speechiness,
            frame.acousticness,
            frame.instrumentalness,
            frame.liveness,
            frame.valence
        };
    }

    void minMaxScale(std::vector<std::vector<float>>& data) {
        size_t numFeatures = data[0].size();
        for (size_t j = 0; j < numFeatures; ++j) {
            // Find the min and max values for the current feature (column)
            float minVal = std::numeric_limits<float>::infinity();
            float maxVal = -std::numeric_limits<float>::infinity();

            // Loop through all rows to find the min and max for the feature
            for (const auto& row : data) {
                minVal = std::min(minVal, row[j]);
                maxVal = std::max(maxVal, row[j]);
            }

            // Check if min == max, and if so, handle the scaling for this feature
            if (minVal != maxVal) {
                // Scale the data for the current feature
                for (auto& row : data) {
                    row[j] = (row[j] - minVal) / (maxVal - minVal);
                }
            } else {
                // If all values in this column are the same set them to 0
                for (auto& row : data) {
                    row[j] = 0.0f;  // or row[j] = 1.0f; if you prefer
                }
            }
        }
    }

    std::vector<std::vector<float>> prepareDataForKMeans(const std::vector<SpotifyFrame>& frames) {
        std::vector<std::vector<float>> data;

        // Extract features from the frames
        for (const auto& frame : frames) {
            data.push_back(extractFeatures(frame));
        }

        // Apply min-max scaling to the data
        minMaxScale(data);

        return data;
    }

}  // namespace SpotifyGenreRevealParty