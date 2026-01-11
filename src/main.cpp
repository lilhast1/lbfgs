#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>

#include "../cuda/lbfgs_kernel.cu"

// include your full.cu header OR forward-declare LBFGS
// #include "lbfgs.cuh"

struct Quadratic {
    // f(x) = 0.5 * ||x||^2
    double f(double* d_x, std::size_t n) {
        double result = 0.0;

        // compute on host for simplicity
        double* h_x = new double[n];
        cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

        for (std::size_t i = 0; i < n; ++i) result += 0.5 * h_x[i] * h_x[i];

        delete[] h_x;
        return result;
    }

    // df/dx = x
    void df(double* d_x, double* d_g, std::size_t n) {
        cudaMemcpy(d_g, d_x, n * sizeof(double), cudaMemcpyDeviceToDevice);
    }
};

int main() {
    constexpr std::size_t N = 16;
    constexpr std::size_t M = 5;
    constexpr std::size_t MAX_ITR = 50;

    double x0[N];
    for (std::size_t i = 0; i < N; ++i) x0[i] = 10.0;  // start far from minimum

    Quadratic func;
    lbfgs<Quadratic> optimizer;

    optimizer(N, M, x0, MAX_ITR, func, 1e-8);

    std::cout << "Result x:\n";
    for (double v : x0) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
