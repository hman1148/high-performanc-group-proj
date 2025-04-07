cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 61 75 86)

project(class_project LANGUAGES CXX CUDA)

# Check if MPI was found correctly
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add GPU subdirectory
add_subdirectory(gpu)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Main executable
add_executable(class_project 
    src/main.cpp
    src/AlgorithmFactory.cpp
    src/SpotifyFrameUtils.cpp
    tools/utils.cpp
    tools/SpotifyFrameReader.cpp
)

# Include MPI headers in the main executable too
target_include_directories(class_project PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
)

# Link libraries including MPI
target_link_libraries(class_project PRIVATE 
    gpu
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)

# Add any MPI-specific flags
target_compile_options(class_project PRIVATE
    ${MPI_CXX_COMPILE_OPTIONS}
)

target_compile_definitions(class_project PRIVATE
    ${MPI_CXX_COMPILE_DEFINITIONS}
    USE_MPI
)