cmake_minimum_required(VERSION 3.16)
project(LBFGS_CUDA LANGUAGES CXX CUDA)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# Find CUDA (old-style)
# -----------------------------
find_package(CUDA REQUIRED)

# -----------------------------
# Set GPU architectures
# -----------------------------
# Adjust based on your GPU:
# T550 -> 5.2, GTX1060 -> 6.1
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61")

# Use shared runtime to avoid missing static libraries
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# -----------------------------
# Source files
# -----------------------------
set(SRC
    src/main.cpp
    cuda/lbfgs_kernel.cu
)

# -----------------------------
# Add executable
# -----------------------------
cuda_add_executable(lbfgs_cuda ${SRC})

# -----------------------------
# Include directories
# -----------------------------
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(cuda)  # Optional: for your lbfgs_kernel.cu headers

# -----------------------------
# Link libraries
# -----------------------------
target_link_libraries(lbfgs_cuda
    ${CUDA_CUBLAS_LIBRARIES}  # cuBLAS
    ${CUDA_CUDA_LIBRARY}       # CUDA runtime
)