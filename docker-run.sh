#!/bin/bash

# Docker-based run script for CUDA L-BFGS project

HOST_PWD=$(pwd)

# Check if binary exists
if [ ! -f "build/bin/lbfgs_cuda" ]; then
    echo "Error: Binary not found. Run ./docker-build.sh first"
    exit 1
fi

echo "Running L-BFGS CUDA program in Docker..."
echo ""

sudo docker run --rm --gpus all \
    -v "$HOST_PWD:$HOST_PWD" \
    -w "$HOST_PWD" \
    nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 \
    ./build/bin/lbfgs_cuda "$@"