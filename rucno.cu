#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/utils/functions.h"

#define BLOCK_SIZE 64

double gpu_sum(double* d_elements, int n);

/*----------------------------------------Datatypes---------------------------------------------------*/

struct GpuMatrix {
    int n, m;
    double* elems;
    GpuMatrix(int n, int m) : n(n), m(m) { cudaMalloc(&elems, n * m * sizeof(double)); }
    ~GpuMatrix() { cudaFree(elems); }
};

struct GpuVector {
    int n;
    double* elems;
    GpuVector(int n) : n(n) { cudaMalloc(&elems, n * sizeof(double)); }
    ~GpuVector() { cudaFree(elems); }
};

struct UnifiedVector {
    int n;
    double* elems;
    UnifiedVector(int n) : n(n) { cudaMallocManaged(&elems, n * sizeof(double)); }
    ~UnifiedVector() { cudaFree(elems); }
};

struct DotLookupTable {
    double* d_partial;
    double* h_partial;
    int max_grid_size;

    DotLookupTable(int n) {
        max_grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&d_partial, max_grid_size * sizeof(double));
        h_partial = (double*)malloc(max_grid_size * sizeof(double));
    }
    ~DotLookupTable() {
        cudaFree(d_partial);
        free(h_partial);
    }

    void clean() {
        cudaMemset(d_partial, 0, max_grid_size * sizeof(double));
        memset(h_partial, 0, max_grid_size * sizeof(double));
    }
};

struct KernelConfig {
    int blockSize;
    int gridSize;

    KernelConfig(int n) {
        blockSize = 256;
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        gridSize = 32 * numSMs;
        int maxBlocks = (n + blockSize - 1) / blockSize;
        if (gridSize > maxBlocks)
            gridSize = maxBlocks;
    }
};

/*----------------------------------------KERNELI---------------------------------------------------*/

// CUDA kernel. Each thread takes care of one element of c, each thread block preforms reduction
__global__ void dotProduct(const double* a, const double* b, double* c, int n);
__global__ void setVectorScalar(double* r, const double* q, double factor, int n);
__global__ void axpy(double alpha, const double* x, double* y, int n);
__global__ void mulVecScal(double alpha, const double* x, double* y, int n);
__global__ void sum_reduction_kernel(const double* input, double* output, int n);

/*----------------------------------------Testovi---------------------------------------------------*/

__global__ void quad_kernel(const double* x, double* f_vals, double* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double xi = x[i];
        if (f_vals) f_vals[i] = xi * xi;
        if (g)      g[i] = 2.0 * xi;
    }
}

struct QuadraticTest {
    double* d_temp_f;
    QuadraticTest(int n) { cudaMalloc(&d_temp_f, n * sizeof(double)); }
    ~QuadraticTest() { cudaFree(d_temp_f); }

    double f(double* d_x, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quad_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, nullptr, n);
        return gpu_sum(d_temp_f, n);
    }
    void df(double* d_x, double* d_g, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quad_kernel<<<blocks, BLOCK_SIZE>>>(d_x, nullptr, d_g, n);
    }
};


__global__ void rosen_kernel(const double* x, double* f_vals, double* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        double x_curr = x[i];
        double x_next = x[i + 1];
        double t1 = (x_next - x_curr * x_curr);
        double t2 = (1.0 - x_curr);

        if (f_vals)
            f_vals[i] = 100.0 * t1 * t1 + t2 * t2;

        // Gradient components (requires atomicAdd because indices overlap)
        if (g) {
            atomicAdd(&g[i], -400.0 * x_curr * t1 - 2.0 * t2);
            atomicAdd(&g[i + 1], 200.0 * t1);
        }
    }
}

struct RosenbrockTest {
    double* d_temp_f;
    RosenbrockTest(int n) { cudaMalloc(&d_temp_f, n * sizeof(double)); }
    ~RosenbrockTest() { cudaFree(d_temp_f); }

    double f(double* d_x, int n) {
        cudaMemset(d_temp_f, 0, n * sizeof(double));
        rosen_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_temp_f, nullptr, n);
        return gpu_sum(d_temp_f, n);
    }
    void df(double* d_x, double* d_g, int n) {
        cudaMemset(d_g, 0, n * sizeof(double));
        rosen_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, nullptr, d_g, n);
    }
};

__global__ void rastrigin_kernel(const double* x, double* f_vals, double* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double xi = x[i];
        if (f_vals)
            f_vals[i] = xi * xi - 10.0 * cos(2.0 * M_PI * xi);
        if (g)
            g[i] = 2.0 * xi + 20.0 * M_PI * sin(2.0 * M_PI * xi);
    }
}

struct RastriginTest {
    double* d_temp_f;
    RastriginTest(int n) { cudaMalloc(&d_temp_f, n * sizeof(double)); }
    ~RastriginTest() { cudaFree(d_temp_f); }

    double f(double* d_x, int n) {
        rastrigin_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_temp_f, nullptr,
                                                                            n);
        return 10.0 * n + gpu_sum(d_temp_f, n);
    }
    void df(double* d_x, double* d_g, int n) {
        rastrigin_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, nullptr, d_g, n);
    }
};

/*----------------------------------------Wrapperfje---------------------------------------------------*/

// neka su vektori a i b na gpu alocirani vraca a . b
double dot(const double* a, const double* b, int n, DotLookupTable* context = nullptr);

template <typename Func>
double lbfgs(int n, int m, double* x0, int max_itr, Func func, const double eps = 1e-9);

int main(int argc, char* argv[]) {
    int N = 1 << 16;  // Problem size
    int M = 10;       // History size
    double* x0 = new double[N];

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float ms = 0;

    // --- TEST 1: QUADRATIC ---
    for (int i = 0; i < N; i++) x0[i] = 5.0;  // Start far away
    
    QuadraticTest quad(N);
    printf("Starting Quadratic...\n");
    
    cudaEventRecord(startEvent);
    double final_f = lbfgs(N, M, x0, 1000, quad, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    printf("Quadratic Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << ms << " ms\n";

    // --- TEST 2: ROSENBROCK ---
    for (int i = 0; i < N; i++) x0[i] = -1.2;  // Standard starting point
    RosenbrockTest rosen(N);

    printf("Starting Rosenbrock...\n");
    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, rosen, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    printf("Rosenbrock Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << ms << " ms\n";

    // --- TEST 2: Rastrigin ---
    for (int i = 0; i < N; i++) x0[i] = -1.2;  // Standard starting point
    RastriginTest rastrigin(N);
    printf("Starting Rastrigin...\n");
    
    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, rastrigin, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    printf("Rastrigin Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << ms << " ms\n";

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    delete[] x0;

    return 0;
}

/*----------------------------------------IMPL---------------------------------------------------*/

/*----------------------------------------Datatypes---------------------------------------------------*/

/*----------------------------------------KERNELI---------------------------------------------------*/

__global__ void mulVecScal(double alpha, const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += alpha * x[i];
}

// y = y + alpha * x
__global__ void axpy(double alpha, const double* x, double* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] += alpha * x[i];
    }
}

// r = factor * q
__global__ void setVectorScalar(double* r, const double* q, double factor, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        r[i] = q[i] * factor;
    }
}

__global__ void dotProduct(const double* a, const double* b, double* c, int n) {
    // Get global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double buf[BLOCK_SIZE];

    // Initialize shared buffer
    buf[threadIdx.x] = 0;

    // Make sure we do not go out of bounds
    while (tid < n) {
        buf[threadIdx.x] += a[tid] * b[tid];
        tid += gridDim.x * blockDim.x;
    }

    __syncthreads();

    // Reduction:
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            // perform the addition:
            buf[threadIdx.x] += buf[threadIdx.x + i];
        }
        // wait at the barrier:
        __syncthreads();
        i = i / 2;
    }

    // only one thread in block writes the result:
    if (threadIdx.x == 0) {
        c[blockIdx.x] = buf[0];
    }
}

__global__ void sum_reduction_kernel(const double* input, double* output, int n) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = (i < n) ? input[i] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(output, sdata[0]);
}

/*----------------------------------------Wrapperfje---------------------------------------------------*/

// Helper to get sum of a device array
double gpu_sum(double* d_elements, int n) {
    static double* d_res = nullptr;
    if (!d_res)
        cudaMalloc(&d_res, sizeof(double));

    cudaMemset(d_res, 0, sizeof(double));
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_reduction_kernel<<<blocks, BLOCK_SIZE>>>(d_elements, d_res, n);

    double h_res;
    cudaMemcpy(&h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    return h_res;
}

double dot(const double* a, const double* b, int n, DotLookupTable* context) {
    dim3 blockSize = BLOCK_SIZE;
    dim3 gridSize = (n + blockSize.x - 1) / blockSize.x;

    auto datasize = sizeof(double) * n;
    auto partial_results_size = sizeof(double) * gridSize.x;

    double *d_c, *h_c;  // parcijalne sume

    if (!context) {
        h_c = (double*)malloc(partial_results_size);
        cudaMalloc(&d_c, partial_results_size);
        dotProduct<<<gridSize, blockSize>>>(a, b, d_c, n);
    } else {
        dotProduct<<<gridSize, blockSize>>>(a, b, context->d_partial, n);
    }

    // redukcija
    if (!context) {
        cudaMemcpy(h_c, d_c, partial_results_size, cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(context->h_partial, context->d_partial, partial_results_size,
                   cudaMemcpyDeviceToHost);
        h_c = context->h_partial;
    }

    double result = 0.0;
    for (int i = 0; i < gridSize.x; i++) result += h_c[i];

    if (!context) {
        cudaFree(d_c);
        free(h_c);
    } else {
        context->clean();
    }

    return result;
}

template <typename Func>
double lbfgs(int n, int m, double* x0, int max_itr, Func func, const double eps) {
    UnifiedVector x(n), g(n);
    GpuVector x_old(n), g_old(n), d(n), q(n), r(n);
    GpuMatrix S(n, m), Y(n, m);

    // For CPU evaluation, we need a host buffer
    double* host_x_trial = (double*)malloc(n * sizeof(double));

    double* rho = new double[m];
    double* alpha_hist = new double[m];
    DotLookupTable dw(n);
    KernelConfig cfg(n);

    cudaMemcpy(x.elems, x0, n * sizeof(double), cudaMemcpyHostToDevice);

    double val = func.f(x.elems, n);
    func.df(x.elems, g.elems, n);

    for (int k = 0; k < max_itr; k++) {
        double g_norm = std::sqrt(dot(g.elems, g.elems, n, &dw));
        if (g_norm < eps)
            break;

        // 1. Compute Search Direction d
        if (k == 0) {
            setVectorScalar<<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, -1.0, n);
        } else {
            // --- TWO-LOOP RECURSION ---
            cudaMemcpy(q.elems, g.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
            int bound = (k > m) ? m : k;
            for (int i = bound - 1; i >= 0; --i) {
                int idx = (k - 1 - i) % m;
                alpha_hist[idx] = rho[idx] * dot(S.elems + idx * n, q.elems, n, &dw);
                axpy<<<cfg.gridSize, cfg.blockSize>>>(-alpha_hist[idx], Y.elems + idx * n, q.elems,
                                                      n);
            }

            int last_idx = (k - 1) % m;
            double s_dot_y = dot(S.elems + last_idx * n, Y.elems + last_idx * n, n, &dw);
            double y_dot_y = dot(Y.elems + last_idx * n, Y.elems + last_idx * n, n, &dw);
            double gamma = (y_dot_y > 1e-18) ? (s_dot_y / y_dot_y) : 1.0;

            setVectorScalar<<<cfg.gridSize, cfg.blockSize>>>(r.elems, q.elems, gamma, n);

            for (int i = 0; i < bound; ++i) {
                int idx = (k - bound + i) % m;
                double beta = rho[idx] * dot(Y.elems + idx * n, r.elems, n, &dw);
                axpy<<<cfg.gridSize, cfg.blockSize>>>(alpha_hist[idx] - beta, S.elems + idx * n,
                                                      r.elems, n);
            }
            setVectorScalar<<<cfg.gridSize, cfg.blockSize>>>(d.elems, r.elems, -1.0, n);
        }

        // 2. BACKTRACKING LINE SEARCH
        double g_dot_d = dot(g.elems, d.elems, n, &dw);

        // Safety: If d is not a descent direction, reset to steepest descent
        if (g_dot_d >= 0) {
            setVectorScalar<<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, -1.0, n);
            g_dot_d = dot(g.elems, d.elems, n, &dw);
        }

        double step = 1.0;
        const double c1 = 1e-4;    // Sufficient decrease constant
        const double decay = 0.5;  // Backtracking rate
        bool success = false;

        // Save current state for history calculation
        cudaMemcpy(x_old.elems, x.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_old.elems, g.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);

        for (int ls_iter = 0; ls_iter < 25; ls_iter++) {
            // Trial x = x_old + step * d
            cudaMemcpy(x.elems, x_old.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
            axpy<<<cfg.gridSize, cfg.blockSize>>>(step, d.elems, x.elems, n);

            // HASTA: ovdje je neki error, vraća je isključivo isti rezultat, 
            // ChatGPT rekao da uradim ovo ispod pa je počelo raditi kako treba
            // Samo Rastrigin ostane vrtiti forever, ne znam zašto

            // Copy to host for CPU evaluation
            cudaMemcpy(host_x_trial, x.elems, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            double f_new = func.f(host_x_trial, n);
            
// ovo sam mijenjao
//            double f_new = func.f(x.elems, n);
//            cudaDeviceSynchronize();
            
            // Armijo Condition
            if (f_new <= val + c1 * step * g_dot_d) {
                val = f_new;
                success = true;
                break;
            }
            step *= decay;
        }

        if (!success) {
            // If LS fails, we usually reset history and try a very small step
            std::cout << "Line Search Failed! Resetting L-BFGS history..." << std::endl;
            k = -1;  // Reset loop (k++ will make it 0)
            continue;
        }

        // 3. Update Gradient and History
        func.df(x.elems, g.elems, n);

        int cur = k % m;
        // S_cur = x - x_old
        cudaMemcpy(S.elems + cur * n, x.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
        axpy<<<cfg.gridSize, cfg.blockSize>>>(-1.0, x_old.elems, S.elems + cur * n, n);

        // Y_cur = g - g_old
        cudaMemcpy(Y.elems + cur * n, g.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
        axpy<<<cfg.gridSize, cfg.blockSize>>>(-1.0, g_old.elems, Y.elems + cur * n, n);

        double sy = dot(S.elems + cur * n, Y.elems + cur * n, n, &dw);
        if (sy > 1e-12) {
            rho[cur] = 1.0 / sy;
        } else {
            // Skip history update if curvature is not positive
            k--;  // effectively "redo" this step index in history next time
        }
    }

    cudaMemcpy(x0, x.elems, n * sizeof(double), cudaMemcpyDeviceToHost);
    free(host_x_trial);
    delete[] rho;
    delete[] alpha_hist;
    return val;
}