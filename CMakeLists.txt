cmake_minimum_required(VERSION 3.28)
project(class_project)

set(CMAKE_CXX_STANDARD 11)

add_executable(class_project main.cpp
        models/SpotifyFrame.h
        models/SpotifyFrame.h
        tools/SpotifyFrameReader.cpp
        tools/SpotifyFrameReader.h)
