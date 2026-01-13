// #include <cuda_runtime.h>

#include <cstddef>
#include <iostream>

#include "lbfgs/lbfgs_kernel.h"
#include "utils/functions.h"
// include your full.cu header OR forward-declare LBFGS
// #include "lbfgs.cuh"

int main() {
    constexpr std::size_t N = 16;
    constexpr std::size_t M = 5;
    constexpr std::size_t MAX_ITR = 50;

    double x0[N];
    for (std::size_t i = 0; i < N; ++i) x0[i] = 10.0;  // start far from minimum

    Quadratic2D func;
    lbfgs::basic::lbfgs optimizer;

    optimizer(N, M, x0, MAX_ITR, func, 1e-8);

    std::cout << "Result x:\n";
    for (double v : x0) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
