cmake_minimum_required(VERSION 3.18)
project(LBFGS_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Sources
set(SOURCES
    src/main.cpp
    src/lbfgs.cpp
    cuda/lbfgs_kernel.cu
)

# Create executable
add_executable(lbfgs_cuda ${SOURCES})

# Enable CUDA separable compilation (needed for device code linking)
set_target_properties(lbfgs_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: specify GPU architecture (adjust to your GPU)
set_target_properties(lbfgs_cuda PROPERTIES
    CUDA_ARCHITECTURES "61"
)

# Link libraries if needed (e.g., cuBLAS)
find_package(CUDA REQUIRED)
target_link_libraries(lbfgs_cuda PRIVATE ${CUDA_LIBRARIES})
