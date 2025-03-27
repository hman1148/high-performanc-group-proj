cmake_minimum_required(VERSION 3.28)
project(class_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(class_project src/main.cpp
        models/SpotifyFrame.h
        src/AlgorithmFactory.cpp
        src/SpotifyFrameUtils.cpp
        tools/utils.cpp
        tools/SpotifyFrameReader.cpp
        tools/SpotifyFrameReader.h
        algoirthm_impl/SharedGPU.cu
        algoirthm_impl/SharedGPU.cuh
        algoirthm_impl/SharedGPU.cuh
)
