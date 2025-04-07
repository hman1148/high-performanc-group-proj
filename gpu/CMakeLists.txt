cmake_minimum_required(VERSION 3.18)

# Enable CUDA language
enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Find MPI
find_package(MPI REQUIRED)

# Set CUDA architecture flags explicitly
set(CMAKE_CUDA_ARCHITECTURES 60 70 75)

# Add library
add_library(gpu STATIC
    SharedGPU.cu
    GlobalGPU.cu
)

# Specify CUDA as the language for source files
set_source_files_properties(
    SharedGPU.cu
    GlobalGPU.cu
    PROPERTIES
    LANGUAGE CUDA
)

# Set properties explicitly
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
    ${MPI_CXX_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(gpu PUBLIC
    CUDA::cudart
    MPI::MPI_CXX  
)

# Add compile definitions
target_compile_definitions(gpu PUBLIC
    USE_MPI 
)