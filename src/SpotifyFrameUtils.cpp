//
// Created by Joe Brijs on 3/21/25.
//
#include "../tools/SpotifyFrameUtils.h"
#include <limits>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "../tools/SpotifyFrameReader.h"

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

    void writePointsToBinary(const std::string& filename, const std::vector<Point>& points) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for binary write: " + filename);

        size_t numPoints = points.size();
        size_t dim = points.empty() ? 0 : points[0].features.size();

        out.write(reinterpret_cast<const char*>(&numPoints), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));

        for (const Point& p : points) {
            out.write(reinterpret_cast<const char*>(p.features.data()), dim * sizeof(float));
        }
    }

    std::vector<Point> readPointsFromBinary(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for binary read: " + filename);

        size_t numPoints, dim;
        in.read(reinterpret_cast<char*>(&numPoints), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&dim), sizeof(size_t));

        std::vector<Point> points(numPoints, Point(std::vector<float>(dim)));
        for (Point& p : points) {
            in.read(reinterpret_cast<char*>(p.features.data()), dim * sizeof(float));
        }

        return points;
    }

    std::vector<Point> getOrLoadPoints(const std::string& csvFile, const std::string& binaryCache) {
        std::vector<Point> points;

        if (std::ifstream binCheck(binaryCache); binCheck) {
            std::cout << "Reading points from binary cache..." << std::endl;
            try {
                points = readPointsFromBinary(binaryCache);
                return points;
            } catch (const std::exception& e) {
                std::cerr << "Failed to read binary cache: " << e.what() << "\nFalling back to CSV..." << std::endl;
            }
        }

            std::cout << "Reading frames from CSV... " << csvFile << std::endl;
            if (const std::ifstream fileCheck(csvFile); !fileCheck) {
                throw std::runtime_error("CSV file not found or cannot be opened: " + csvFile);

        }

        const auto frames = SpotifyFrameReader::readCSV(csvFile);

        for (const auto& framePtr : frames) {
            if (framePtr) {
                auto features = extractFeatures(*framePtr);
                points.emplace_back(features);
            }
        }

        try {
            writePointsToBinary(binaryCache, points);
                std::cout << "Cached scaled points to binary file." << std::endl;

        } catch (const std::exception& e) {
                std::cerr << "Failed to write binary cache: " << e.what() << std::endl;

        }

        return points;
    }


}  // namespace SpotifyGenreRevealParty