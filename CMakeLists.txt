cmake_minimum_required(VERSION 3.28)
project(class_project)

set(CMAKE_CXX_STANDARD 17)

# Check for OpenMP support
find_package(OpenMP REQUIRED)

add_executable(class_project src/main.cpp
        models/SpotifyFrame.h
        src/AlgorithmFactory.cpp
        src/SpotifyFrameUtils.cpp
        tools/utils.cpp
        tools/SpotifyFrameReader.cpp
        tools/SpotifyFrameReader.h)

# Link OpenMP to your executable
if(OpenMP_CXX_FOUND)
    target_link_libraries(class_project PUBLIC OpenMP::OpenMP_CXX)
endif()