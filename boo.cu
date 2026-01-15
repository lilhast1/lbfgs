#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

/*----------------------------------------KERNELI---------------------------------------------------*/

// CUDA kernel. Each thread takes care of one element of c, each thread block preforms reduction
__global__ void dotProduct(const double* a, const double* b, double* c, int n);
__global__ void setVectorScalar(double* r, const double* q, double factor, int n);
__global__ void axpy(double alpha, const double* x, double* y, int n);
__global__ void mulVecScal(double alpha, const double* x, double* y, int n);
__global__ void sum_reduction_kernel(const double* input, double* output, int n);

/*----------------------------------------Testovi---------------------------------------------------*/

struct QuadraticTest {
    double* d_temp_f;
    QuadraticTest(int n) {}
    ~QuadraticTest() {}

    double f(double* d_x, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) s += d_x[i] * d_x[i];
        return s;
    }
    void df(double* d_x, double* d_g, int n) {
        for (int i = 0; i < n; i++) d_g[i] = 2 * d_x[i];
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
    int N = 2;   // Problem size
    int M = 10;  // History size
    double* x0 = new double[N];

    // --- TEST 1: QUADRATIC ---
    for (int i = 0; i < N; i++) x0[i] = 5.0;  // Start far away
    QuadraticTest quad(N);
    printf("Starting Quadratic...\n");
    double final_f = lbfgs(N, M, x0, 1000, quad, 1e-6);
    printf("Quadratic Final F: %e (Target: 0)\n", final_f);

    // --- TEST 2: ROSENBROCK ---
    for (int i = 0; i < N; i++) x0[i] = -1.2;  // Standard starting point
    RosenbrockTest rosen(N);
    printf("Starting Rosenbrock...\n");
    final_f = lbfgs(N, M, x0, 5000, rosen, 1e-6);
    printf("Rosenbrock Final F: %e (Target: 0)\n", final_f);

    // --- TEST 2: Rastrigin ---
    for (int i = 0; i < N; i++) x0[i] = -1.2;  // Standard starting point
    RastriginTest rastrigin(N);
    printf("Starting Rastrigin...\n");
    final_f = lbfgs(N, M, x0, 5000, rastrigin, 1e-6);
    printf("Rastrigin Final F: %e (Target: 0)\n", final_f);

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

__global__ void axpy(double alpha, const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += alpha * x[i];
}

__global__ void setVectorScalar(double* r, const double* q, double factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        r[i] = q[i] * factor;
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

    double* rho = new double[m];
    double* alpha_hist = new double[m];
    double gamma;

    double* host_tmp = new double[n];

    DotLookupTable context_ndim_dot(n);

    cudaMemcpy(x.elems, x0, n * sizeof(double), cudaMemcpyHostToDevice);

    double val = func.f(x.elems, n);
    func.df(x.elems, g.elems, n);

    for (int k = 0; k < max_itr; k++) {
        std::cout << "ITR: " << k << " F = " << val << '\n';
        if (std::isnan(val))
            throw std::range_error(std::string("greska u iteraciji") + std::to_string(k));

        double g_norm = std::sqrt(dot(g.elems, g.elems, n, &context_ndim_dot));
        std::cout << "|g| = " << g_norm << '\n';
        if (g_norm < eps)
            break;

        // 2. Compute Search Direction d
        if (k == 0) {
            // First iteration: steepest descent
            // d = -g
            setVectorScalar<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d.elems, g.elems,
                                                                               -1.0, n);
        } else {
            // --- TWO-LOOP RECURSION ---

            // q = g
            cudaMemcpy(q.elems, g.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);

            int bound = (k > m) ? m : k;

            // Backward Loop
            for (int i = bound - 1; i >= 0; --i) {
                int idx = (k - 1 - i) % m;
                double* s_i = S.elems + idx * n;
                double* y_i = Y.elems + idx * n;

                // alpha_hist[i] = rho[i] * (s_i . q)
                double s_dot_q = dot(s_i, q.elems, n, &context_ndim_dot);
                alpha_hist[idx] = rho[idx] * s_dot_q;

                // q = q - alpha_i * y_i
                axpy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(-alpha_hist[idx], y_i,
                                                                        q.elems, n);
            }

            // Scaling (Hessian approximation initial guess)
            int last_idx = (k - 1) % m;
            double* s_last = S.elems + last_idx * n;
            double* y_last = Y.elems + last_idx * n;

            double s_dot_y = dot(s_last, y_last, n, &context_ndim_dot);  // This is 1/rho
            double y_dot_y = dot(y_last, y_last, n, &context_ndim_dot);
            gamma = s_dot_y / y_dot_y;

            // r = gamma * q
            setVectorScalar<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(r.elems, q.elems,
                                                                               gamma, n);

            // Forward Loop
            for (int i = 0; i < bound; ++i) {
                int idx = (k - bound + i) % m;
                double* s_i = S.elems + idx * n;
                double* y_i = Y.elems + idx * n;

                // beta = rho[i] * (y_i . r)
                double y_dot_r = dot(y_i, r.elems, n, &context_ndim_dot);
                double beta = rho[idx] * y_dot_r;

                // r = r + (alpha_hist[i] - beta) * s_i
                double weight = alpha_hist[idx] - beta;
                axpy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(weight, s_i, r.elems, n);
            }

            // Search direction d = -r
            setVectorScalar<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d.elems, r.elems,
                                                                               -1.0, n);
        }

        double g_dot_d = dot(g.elems, d.elems, n, &context_ndim_dot);
        std::cout << "g.d = " << g_dot_d << '\n';
        double step_size = std::fabs(1.0 / g_dot_d);

        if (k > 0) {
            // L = sqrt(dot(s_old, s_old) / dot(y_old, y_old))
            int last_idx = (k - 1) % m;
            double* s_last = S.elems + last_idx * n;
            double* y_last = Y.elems + last_idx * n;

            double s_norm_sq = dot(s_last, s_last, n, &context_ndim_dot);
            double y_norm_sq = dot(y_last, y_last, n, &context_ndim_dot);
            double L = std::sqrt(s_norm_sq / y_norm_sq);

            double beta_param = 0.01;

            double d_dot_d = dot(d.elems, d.elems, n, &context_ndim_dot);

            // step_size = (3 * (beta - 1) * dot(g, d)) / (2 * L * dot(d, d))
            step_size = (4 * (beta_param - 1.0) * g_dot_d) / (2 * L * d_dot_d + 20 * eps);

            double eigen_bound =
                1 / gamma + std::max(0.0, 1. / (gamma * gamma) * L * L + 1. / (1 + L * L / gamma));

            if (step_size < eigen_bound)
                step_size = eigen_bound;

            // Safeguard: step_size should be positive and not "inf"
            if (step_size <= 0 || std::isnan(step_size))
                step_size = std::fabs(1.0 / g_dot_d);
        }

        double f_old = val;

        // 2. BACKTRACKING
        bool success = false;
        for (int ls_iter = 0; ls_iter < 10; ls_iter++) {
            // Save x_old to x_trial (use d_r as a temp buffer to save memory)

            cudaMemcpy(r.elems, x.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
            // x_trial = x + step_size * d
            axpy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(step_size, d.elems, r.elems, n);

            cudaMemcpy(host_tmp, r.elems, n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            double f_new = func.f(host_tmp, n);
            // Armijo Condition (Sufficient Decrease)
            // f(x + step*d) <= f(x) + 1e-4 * step * (g^T d)
            if (f_new <= f_old + 1e-4 * step_size * g_dot_d) {
                val = f_new;
                // Commit the change to x
                cudaMemcpy(x.elems, r.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
                success = true;
                std::cout << "lnsrch gotov u " << ls_iter << " iteracija sa korakom = " << step_size
                          << '\n';
                break;
            }

            // If we didn't decrease, shrink the step and try again
            step_size *= 0.5;
        }

        // Save old x and g before update
        cudaMemcpy(x_old.elems, x.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_old.elems, g.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);

        // Update x: x = x + step_size * d
        axpy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(step_size, d.elems, x.elems, n);

        // Compute new function value and gradient
        val = func.f(x.elems, n);
        func.df(x.elems, g.elems, n);

        std::cout << '(' << x.elems[0] << ' ' << x.elems[1] << ")\n";

        // --- UPDATE HISTORY (S, Y, RHO) ---
        int current_idx = k % m;
        double* s_new = S.elems + current_idx * n;
        double* y_new = Y.elems + current_idx * n;

        // s_new = x - x_old
        // We can use axpy logic: s = x; s = s - 1.0 * x_old
        cudaMemcpy(s_new, x.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
        axpy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(-1.0, x_old.elems, s_new, n);

        // y_new = g - g_old
        cudaMemcpy(y_new, g.elems, n * sizeof(double), cudaMemcpyDeviceToDevice);
        axpy<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(-1.0, g_old.elems, y_new, n);

        // rho[i] = 1.0 / (s_dot_y)
        double sy = dot(s_new, y_new, n, &context_ndim_dot);
        if (sy > 1e-12) {  // Guard against division by zero
            rho[current_idx] = 1.0 / sy;
        } else {
            // rho[current_idx] = 0.0;
            continue;  // nojeva strategija
        }
    }

    cudaMemcpy(x0, x.elems, n * sizeof(double), cudaMemcpyDeviceToHost);

    delete[] rho;
    delete[] alpha_hist;

    return val;
}