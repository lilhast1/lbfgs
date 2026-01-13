cmake_minimum_required(VERSION 3.16)
project(LBFGS_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA


# CUDA architecture (adjust based on your GPU)
# Common values: 61 (GTX 1080), 75 (RTX 2080), 80 (A100), 86 (RTX 3090), 89 (RTX 4090)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/lbfgs
    ${CMAKE_SOURCE_DIR}/src/utils
)

# Source files
set(CUDA_SOURCES
    src/lbfgs/lbfgs.cu
    src/lbfgs/lbfgs_kernel.cu
    src/lbfgs/lbfgs_lnsrch.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/utils/functions.cpp
)

# Create executable
add_executable(lbfgs_cuda ${CUDA_SOURCES} ${CPP_SOURCES})

# Link CUDA libraries
target_link_libraries(lbfgs_cuda 
    CUDA::cudart
    CUDA::cublas
)

# Compiler flags
target_compile_options(lbfgs_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra
    >
)

# Set output directory
set_target_properties(lbfgs_cuda PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Optional: Enable separable compilation for CUDA
set_target_properties(lbfgs_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)