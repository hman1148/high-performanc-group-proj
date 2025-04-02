cmake_minimum_required(VERSION 3.18)

# Enable CUDA
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add library without device linking
add_library(gpu STATIC
    SharedGPU.cu
    SharedGPU.cuh
)

# Set properties
set_target_properties(gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    # Disable device linking at library level
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Include directories
target_include_directories(gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries, but only the host interface
target_link_libraries(gpu PUBLIC
    CUDA::cudart_static
)