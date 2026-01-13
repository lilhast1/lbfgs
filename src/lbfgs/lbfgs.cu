#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>

#include "lbfgs.h"

// --- CUSTOM KERNELS ---
namespace lbfgs {
namespace ghjkm {
__global__ void copy_kernel(int n, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i];
}

__global__ void scal_kernel(int n, double alpha, double* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= alpha;
}

__global__ void axpy_kernel(int n, double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += alpha * x[i];
}

__global__ void dot_kernel(int n, const double* x, const double* y, double* res) {
    __shared__ double cache[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double temp = 0;
    if (i < n)
        temp = x[i] * y[i];
    cache[tid] = temp;
    __syncthreads();

    // Simple reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(res, cache[0]);
}

__global__ void nrm2_kernel(int n, const double* x, double* res) {
    __shared__ double cache[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double temp = 0;
    if (i < n)
        temp = x[i] * x[i];
    cache[tid] = temp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(res, cache[0]);
}

// --- EXISTING L-BFGS KERNELS ---

__global__ void update_x_kernel(int n, double* x, const double* d, double alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += alpha * d[i];
}

__global__ void compute_sy_kernel(int n, double* s, double* y, const double* x_new,
                                  const double* x_old, const double* g_new, const double* g_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = x_new[i] - x_old[i];
        y[i] = g_new[i] - g_old[i];
    }
}

__global__ void scale_vector_kernel(int n, double* r, const double* q, double H0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        r[i] = H0 * q[i];
}

// --- L-BFGS CLASS ---

template <typename Func>
class lbfgs {
    double *d_x, *d_x_old, *d_g, *d_g_old, *d_d, *d_q, *d_r, *d_S, *d_Y;
    double *rho, *a;
    double* d_res;  // Device memory to hold intermediate reduction results

    void init(std::size_t problem_size, std::size_t memory_size) {
        cudaMallocManaged(&d_x, problem_size * sizeof(double));
        cudaMallocManaged(&d_x_old, problem_size * sizeof(double));
        cudaMallocManaged(&d_g, problem_size * sizeof(double));
        cudaMallocManaged(&d_g_old, problem_size * sizeof(double));
        cudaMalloc(&d_d, problem_size * sizeof(double));
        cudaMalloc(&d_q, problem_size * sizeof(double));
        cudaMalloc(&d_r, problem_size * sizeof(double));
        cudaMalloc(&d_S, problem_size * memory_size * sizeof(double));
        cudaMalloc(&d_Y, problem_size * memory_size * sizeof(double));
        cudaMalloc(&d_res, sizeof(double));  // for dot and nrm2

        rho = new double[memory_size];
        a = new double[memory_size];
    }

    void destroy() {
        cudaFree(d_x);
        cudaFree(d_x_old);
        cudaFree(d_g);
        cudaFree(d_g_old);
        cudaFree(d_d);
        cudaFree(d_q);
        cudaFree(d_r);
        cudaFree(d_S);
        cudaFree(d_Y);
        cudaFree(d_res);
        delete[] rho;
        delete[] a;
    }

    // Helper to perform reduction and bring result to host
    double get_dot(int n, const double* x, const double* y) {
        double h_res = 0;
        cudaMemset(d_res, 0, sizeof(double));
        dot_kernel<<<(n + 255) / 256, 256>>>(n, x, y, d_res);
        cudaMemcpy(&h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
        return h_res;
    }

    double get_nrm2(int n, const double* x) {
        double h_res = 0;
        cudaMemset(d_res, 0, sizeof(double));
        nrm2_kernel<<<(n + 255) / 256, 256>>>(n, x, d_res);
        cudaMemcpy(&h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
        return std::sqrt(h_res);
    }

   public:
    void operator()(const std::size_t problem_size, const std::size_t memory_size, double* x0,
                    const std::size_t max_itr, Func func, const double eps = 1e-9) {
        init(problem_size, memory_size);
        int blocks = (problem_size + 255) / 256;
        int threads = 256;

        cudaMemcpy(d_x, x0, problem_size * sizeof(double), cudaMemcpyHostToDevice);
        double val = func.f(d_x, problem_size);
        func.df(d_x, d_g, problem_size);

        // Replace cublasDcopy + cublasDscal
        copy_kernel<<<blocks, threads>>>(problem_size, d_g, d_d);
        scal_kernel<<<blocks, threads>>>(problem_size, -1.0, d_d);

        double alpha = 1.0;
        cudaMemcpy(d_x_old, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_g_old, d_g, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

        update_x_kernel<<<blocks, threads>>>(problem_size, d_x, d_d, alpha);

        val = func.f(d_x, problem_size);
        func.df(d_x, d_g, problem_size);
        compute_sy_kernel<<<blocks, threads>>>(problem_size, d_S, d_Y, d_x, d_x_old, d_g, d_g_old);

        for (int itr = 0; itr < max_itr; itr++) {
            // Replace cublasDnrm2
            double g_norm = get_nrm2(problem_size, d_g);

            if (g_norm / (std::fabs(val) + 1.0) <= eps)
                break;

            int bound = itr > (int)memory_size ? (int)memory_size : itr;
            int curr = (itr - 1) % (int)memory_size;
            cudaMemcpy(d_q, d_g, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

            // Backward Loop
            for (int i = bound - 1; i >= 0; --i) {
                int idx = (itr - 1 - i) % (int)memory_size;
                // Replace cublasDdot
                double dot_sq = get_dot(problem_size, d_S + idx * problem_size, d_q);
                double dot_sy =
                    get_dot(problem_size, d_S + idx * problem_size, d_Y + idx * problem_size);

                rho[idx] = 1.0 / dot_sy;
                a[idx] = rho[idx] * dot_sq;
                // Replace cublasDaxpy
                axpy_kernel<<<blocks, threads>>>(problem_size, -a[idx], d_Y + idx * problem_size,
                                                 d_q);
            }

            // Central scaling
            double ys = get_dot(problem_size, d_Y + curr * problem_size, d_S + curr * problem_size);
            double yy = get_dot(problem_size, d_Y + curr * problem_size, d_Y + curr * problem_size);
            scale_vector_kernel<<<blocks, threads>>>(problem_size, d_r, d_q, ys / yy);

            // Forward Loop
            for (int i = 0; i < bound; ++i) {
                int idx = (itr - bound + i) % (int)memory_size;
                double b = get_dot(problem_size, d_Y + idx * problem_size, d_r);
                double factor = a[idx] - (rho[idx] * b);
                // Replace cublasDaxpy
                axpy_kernel<<<blocks, threads>>>(problem_size, factor, d_S + idx * problem_size,
                                                 d_r);
            }

            // Replace cublasDcopy + cublasDscal
            copy_kernel<<<blocks, threads>>>(problem_size, d_r, d_d);
            scal_kernel<<<blocks, threads>>>(problem_size, -1.0, d_d);

            cudaMemcpy(d_x_old, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_g_old, d_g, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

            update_x_kernel<<<blocks, threads>>>(problem_size, d_x, d_d, 1.0);

            val = func.f(d_x, problem_size);
            func.df(d_x, d_g, problem_size);

            int history_pos = (itr % (int)memory_size);
            compute_sy_kernel<<<blocks, threads>>>(problem_size, d_S + history_pos * problem_size,
                                                   d_Y + history_pos * problem_size, d_x, d_x_old,
                                                   d_g, d_g_old);
        }
        cudaMemcpy(x0, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToHost);
        destroy();
    }
};
}  // namespace ghjkm
}  // namespace lbfgs