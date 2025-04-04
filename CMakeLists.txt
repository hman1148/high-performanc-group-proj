cmake_minimum_required(VERSION 3.18)

project(class_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Turn off device code linking globally - this is crucial
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70")

# Add GPU subdirectory
add_subdirectory(gpu)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Main executable with no device linking
add_executable(class_project 
    src/main.cpp
    src/AlgorithmFactory.cpp
    src/SpotifyFrameUtils.cpp
    models/SharedCpu.cpp
    tools/utils.cpp
    tools/SpotifyFrameReader.cpp
)

# Include directories
target_include_directories(class_project PRIVATE
    ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tools
)

# Link libraries
target_link_libraries(class_project PRIVATE 
    gpu 
    OpenMP::OpenMP_CXX
)

# Disable device linking for the main executable
set_target_properties(class_project PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)