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

        size_t numDimensions = points[0].features.size();

        // Find the min and max values for each dimension
        std::vector<float> minValues(numDimensions, std::numeric_limits<float>::infinity());
        std::vector<float> maxValues(numDimensions, -std::numeric_limits<float>::infinity());

        // Find the min and max values for each dimension
        for (const auto& point : points) {
            for (size_t i = 0; i < numDimensions; ++i) {
                if (point.features[i] < minValues[i]) {
                    minValues[i] = point.features[i];
                }
                if (point.features[i] > maxValues[i]) {
                    maxValues[i] = point.features[i];
                }
            }
        }

        // Scale the features of each point
        for (auto& point : points) {
            for (size_t i = 0; i < numDimensions; ++i) {
                // Avoid division by zero
                if (maxValues[i] != minValues[i]) {
                    point.features[i] = (point.features[i] - minValues[i]) / (maxValues[i] - minValues[i]);
                } else {
                    point.features[i] = 0.0f;  // If min == max, set to 0 (or handle as a special case if needed)
                }
            }
        }
    }


    std::vector<Point> prepareDataForKMeans(const std::vector<SpotifyFrame>& frames) {
        std::vector<Point> data;

        // Extract features from the frames and wrap them in Point objects
        for (const auto& frame : frames) {
            std::vector<float> features = extractFeatures(frame);
            data.push_back(Point(features));  // Create Point and push it
        }

        // Apply min-max scaling to the data
        minMaxScale(data);

        return data;
    }


}  // namespace SpotifyGenreRevealParty