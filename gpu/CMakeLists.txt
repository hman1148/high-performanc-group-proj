cmake_minimum_required(VERSION 3.18)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add library without device linking
add_library(gpu STATIC
    SharedGPU.cu
    SharedGPU.cuh
)

# Set properties
set_target_properties(gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Include directories
target_include_directories(gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries correctly using shared libraries instead of static
target_link_libraries(gpu PUBLIC
    CUDA::cudart
)