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

    // Updated minMaxScale function to work with a vector of Point objects
    void minMaxScale(std::vector<Point>& points) {
        if (points.empty()) {
            return;  // No points to scale
        }

        size_t numDimensions = points[0].dimensions.size();

        // Find the min and max values for each dimension
        std::vector<float> minValues(numDimensions, std::numeric_limits<float>::infinity());
        std::vector<float> maxValues(numDimensions, -std::numeric_limits<float>::infinity());

        // Find the min and max values for each dimension
        for (const auto& point : points) {
            for (size_t i = 0; i < numDimensions; ++i) {
                if (point.dimensions[i] < minValues[i]) {
                    minValues[i] = point.dimensions[i];
                }
                if (point.dimensions[i] > maxValues[i]) {
                    maxValues[i] = point.dimensions[i];
                }
            }
        }

        // Scale the dimensions of each point
        for (auto& point : points) {
            for (size_t i = 0; i < numDimensions; ++i) {
                // Avoid division by zero
                if (maxValues[i] != minValues[i]) {
                    point.dimensions[i] = (point.dimensions[i] - minValues[i]) / (maxValues[i] - minValues[i]);
                } else {
                    point.dimensions[i] = 0.0f;  // If min == max, set to 0 (or handle as a special case if needed)
                }
            }
        }
    }


    std::vector<Point> prepareDataForKMeans(const std::vector<SpotifyFrame>& frames) {
        std::vector<Point> data;

        // Extract features from the frames
        for (const auto& frame : frames) {
            data.push_back(extractFeatures(frame));
        }

        // Apply min-max scaling to the data
        minMaxScale(data);

        return data;
    }

}  // namespace SpotifyGenreRevealParty