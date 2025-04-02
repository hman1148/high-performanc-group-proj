cmake_minimum_required(VERSION 3.21)
project(class_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)  
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)

add_subdirectory(gpu)

find_package(OpenMP REQUIRED)

add_executable(class_project 
        src/main.cpp
        src/AlgorithmFactory.cpp
        src/SpotifyFrameUtils.cpp
        tools/utils.cpp
        tools/SpotifyFrameReader.cpp
)

target_include_directories(class_project PRIVATE
        ${CMAKE_SOURCE_DIR}
)

target_link_libraries(class_project PRIVATE 
        gpu 
        OpenMP::OpenMP_CXX
)
