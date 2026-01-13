#!/bin/bash

# Docker-based build script for CUDA L-BFGS project
# Uses the same NVIDIA CUDA container as your nvcc wrapper

HOST_PWD=$(pwd)

echo "Building L-BFGS CUDA project in Docker..."

docker run --rm --gpus all \
    -v "$HOST_PWD:$HOST_PWD" \
    -w "$HOST_PWD" \
    nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 \
    bash -c "
        # Install CMake if not present
        if ! command -v cmake &> /dev/null; then
            apt-get update && apt-get install -y cmake build-essential
        fi
        
        # Create build directory
        mkdir -p build
        cd build
        
        # Configure with CMake
        cmake ..
        
        # Build
        make -j\$(nproc)
        
        echo '======================================'
        echo 'Build complete!'
        echo 'Binary location: build/bin/lbfgs_cuda'
        echo '======================================'
    "

echo ""
echo "To run the program:"
echo "  ./docker-run.sh"
echo ""