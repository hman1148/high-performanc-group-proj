cmake_minimum_required(VERSION 3.28)
project(class_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(class_project src/main.cpp
        models/SpotifyFrame.h
        src/AlgorithmFactory.cpp
        src/Serial.cpp
        src/SharedCpu.cpp
        src/SharedGpu.cpp
        src/DistributedCpu.cpp
        src/DistributedGpu.cpp
        tools/SpotifyFrameReader.cpp
        tools/SpotifyFrameReader.h)
