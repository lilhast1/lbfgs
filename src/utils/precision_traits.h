#pragma once

#include <cuda_fp16.h>

template<typename T>
struct NumericTraits;

template<>
struct NumericTraits<double> {
    using Accum = double;
    __device__ __host__ static double zero() { return 0.0; }
};

template<>
struct NumericTraits<float> {
    using Accum = float;
    __device__ __host__ static float zero() { return 0.0f; }
};

template<>
struct NumericTraits<__half> {
    using Accum = float;   // IMPORTANT
    __device__ __host__ static __half zero() { return __float2half(0.0f); }
};