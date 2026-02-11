/**
 * @file tmp_main.cu
 * @brief L-BFGS implementation using cuBLAS for vector operations and small CUDA kernels.
 *
 * This file contains small device kernels and host wrappers used to exercise
 * an L-BFGS implementation that relies on cuBLAS for vector operations and
 * custom CUDA kernels for simple elementwise operations and test functions.
 *
 * The style of Doxygen comments mirrors that used in `lbfgs_mixed_precision.cu`.
 * @date 2026-02-11
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#define BLOCK_SIZE 256

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                         cudaGetErrorString(_e));                                 \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t _s = (call);                                               \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",               \
                         __FILE__, __LINE__, (int)_s);                            \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#endif

/*------------------------------------------- KERNELS -------------------------------------------*/

/**
 * @brief Compute L-BFGS history vectors s = x_new - x_old and y = g_new - g_old.
 *
 * @param n Number of elements.
 * @param s Device pointer where s will be written.
 * @param y Device pointer where y will be written.
 * @param x_new Device pointer to the new iterate.
 * @param x_old Device pointer to the previous iterate.
 * @param g_new Device pointer to the new gradient.
 * @param g_old Device pointer to the previous gradient.
 */
__global__ void compute_sy_kernel(int n, float* s, float* y,
                                  const float* x_new, const float* x_old,
                                  const float* g_new, const float* g_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = x_new[i] - x_old[i];
        y[i] = g_new[i] - g_old[i];
    }
}

/**
 * @brief Fill an array with ones on the device.
 *
 * Used as a reusable buffer for computing sums via cuBLAS dot products.
 *
 * @param a Device array to fill with ones.
 * @param n Number of elements.
 */
__global__ void fill_ones(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 1.0f;
}

/**
 * @brief Count NaN/Inf elements in an array (debugging helper).
 *
 * Writes the number of non-finite entries into `out` using a block-wise
 * reduction followed by an atomicAdd.
 *
 * @param a Input device array.
 * @param n Number of elements.
 * @param out Device pointer to integer accumulator (must be initialized to 0).
 */
__global__ void count_nan_inf(const float* a, int n, int* out) {
    __shared__ int sh[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int v = 0;
    if (i < n) {
        float x = a[i];
        v = (!isfinite(x)) ? 1 : 0;
    }
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

/*----------------------------------------- WRAPPERS ------------------------------------------*/

/**
 * @brief Reusable cuBLAS helper to compute the sum of a device float vector.
 *
 * Internally allocates and reuses a device buffer of ones (`d_ones`) and
 * performs a dot product to compute the sum efficiently via cuBLAS.
 *
 * @param h cuBLAS handle.
 * @param d_v Device pointer to input vector.
 * @param n Number of elements.
 * @return float Sum of elements in `d_v`.
 */
static float* d_ones = nullptr;
static int ones_cap = 0;

static float gpu_sum_f32_cublas(cublasHandle_t h, const float* d_v, int n) {
    if (n > ones_cap) {
        if (d_ones) CUDA_CHECK(cudaFree(d_ones));
        CUDA_CHECK(cudaMalloc(&d_ones, n * sizeof(float)));
        int blocks = (n + 255) / 256;
        fill_ones<<<blocks, 256>>>(d_ones, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        ones_cap = n;
    }
    float s = 0.0f;
    CUBLAS_CHECK(cublasSdot(h, n, d_v, 1, d_ones, 1, &s));
    return s;
}

// -------------------- Test functions (f and df on GPU) --------------------
/**
 * @brief Kernel computing quadratic function contributions and gradient.
 *
 * f_i = x_i^2 and g_i = 2*x_i. Per-element contributions are written to
 * `f_vals` (if non-null) to allow device-side accumulation via cuBLAS.
 *
 * @tparam T Floating point type.
 * @param x Input device vector.
 * @param f_vals Per-element function contributions (optional).
 * @param g Gradient output (optional).
 * @param n Number of elements.
 */
template <typename T>
__global__ void quad_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T xi = x[i];
        if (f_vals) f_vals[i] = xi * xi;    // sum x^2
        if (g)      g[i] = (T)2 * xi;       // grad = 2x
    }
}

/**
 * @brief Wrapper for the quadratic test function used by the optimizer.
 *
 * Allocates temporary device storage for per-element function contributions
 * and exposes `f` and `df` methods compatible with the optimizer.
 *
 * @tparam T Floating point type used on device.
 */
template <typename T>
struct QuadraticTest {
    T* d_temp_f = nullptr;
    cublasHandle_t* h = nullptr;

    void set_handle(cublasHandle_t* ph) { h = ph; }

    QuadraticTest(int n) { CUDA_CHECK(cudaMalloc(&d_temp_f, n * sizeof(T))); }
    ~QuadraticTest() { if (d_temp_f) CUDA_CHECK(cudaFree(d_temp_f)); }

    double f(T* d_x, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quad_kernel<T><<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float result = gpu_sum_f32_cublas(*h, (float*)d_temp_f, n);
        return (double)result;
    }

    void df(T* d_x, T* d_g, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quad_kernel<T><<<blocks, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};
/**
 * @brief Non-atomic Rosenbrock kernel variant.
 *
 * Computes per-element contributions for the Rosenbrock function and a
 * non-overlapping gradient without atomics. The kernel writes `f_vals` for
 * indices 0..n-2 and `g` for indices 0..n-1 as a replacement for the atomic
 * version when suitable.
 *
 * @param x Input device vector.
 * @param f_vals Output per-element function contributions (optional).
 * @param g Output gradient vector (optional).
 * @param n Number of elements.
 */
__global__ void rosen_f_g_noatom(const float* x, float* f_vals, float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // f contribution for i in [0..n-2]
    if (f_vals && i < n - 1) {
        float xi = x[i];
        float xj = x[i + 1];
        float t1 = xj - xi * xi;
        float t2 = 1.0f - xi;
        f_vals[i] = 100.0f * t1 * t1 + t2 * t2;
    }
    if (f_vals && i == n - 1) {
        f_vals[i] = 0.0f;
    }

    // gradient for i in [0..n-1]
    if (g && i < n) {
        float gi = 0.0f;

        if (i == 0) {
            float x0 = x[0], x1 = x[1];
            float t1 = x1 - x0 * x0;
            float t2 = 1.0f - x0;
            gi = (-400.0f) * x0 * t1 - 2.0f * t2;
        } else if (i == n - 1) {
            float xm1 = x[n - 1], xm2 = x[n - 2];
            gi = 200.0f * (xm1 - xm2 * xm2);
        } else {
            float xim1 = x[i - 1];
            float xi   = x[i];
            float xip1 = x[i + 1];

            float t_left  = xi   - xim1 * xim1;
            float t_right = xip1 - xi   * xi;
            float t2      = 1.0f - xi;

            gi = 200.0f * t_left + (-400.0f) * xi * t_right - 2.0f * t2;
        }

        g[i] = gi;
    }
}

/**
 * @brief Wrapper for the non-atomic Rosenbrock test.
 *
 * Executes the `rosen_f_g_noatom` kernel and reduces per-element function
 * contributions via cuBLAS when needed.
 *
 * @tparam T Floating point type used on device.
 */
template <typename T>
struct RosenbrockNoAtomTest {
    T* d_temp_f;
    cublasHandle_t* h = nullptr;
    void set_handle(cublasHandle_t* ph) { h = ph; }

    RosenbrockNoAtomTest(int n) { CUDA_CHECK(cudaMalloc(&d_temp_f, n * sizeof(T))); }
    ~RosenbrockNoAtomTest() { if (d_temp_f) CUDA_CHECK(cudaFree(d_temp_f)); }

    RosenbrockNoAtomTest(const RosenbrockNoAtomTest&) = delete;
    RosenbrockNoAtomTest& operator=(const RosenbrockNoAtomTest&) = delete;

    double f(T* d_x, int n) {
        // Only f_vals needed; g null
        int block = 256;
        int grid  = (n + block - 1) / block;
        rosen_f_g_noatom<<<grid, block>>>((const float*)d_x, (float*)d_temp_f, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        float result = gpu_sum_f32_cublas(*h, (float*)d_temp_f, n);
        return (double)result;
    }

    void df(T* d_x, T* d_g, int n) {
        // Only g needed; f null
        int block = 256;
        int grid  = (n + block - 1) / block;
        rosen_f_g_noatom<<<grid, block>>>((const float*)d_x, nullptr, (float*)d_g, n);
        CUDA_CHECK(cudaGetLastError());
    }
};

/**
 * @brief Kernel computing Rastrigin function contributions and gradient.
 *
 * f_i = x_i^2 - 10*cos(2*pi*x_i), and global function value is 10*n + sum(f_i).
 *
 * @tparam T Floating point type.
 * @param x Input device vector.
 * @param f_vals Per-element function contributions (optional).
 * @param g Gradient output (optional).
 * @param n Number of elements.
 */
template <typename T>
__global__ void rastrigin_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T xi = x[i];
        if (f_vals)
            f_vals[i] = xi * xi - (T)10.0 * cos((T)2.0 * (T)M_PI * xi);
        if (g)
            g[i] = (T)2.0 * xi + (T)20.0 * (T)M_PI * sin((T)2.0 * (T)M_PI * xi);
    }
}

/**
 * @brief Wrapper for the Rastrigin test function.
 *
 * Executes the device kernel and reduces per-element function values using
 * cuBLAS to produce the global function value.
 *
 * @tparam T Floating point type.
 */
template <typename T>
struct RastriginTest {
    T* d_temp_f = nullptr;
    cublasHandle_t* h = nullptr;

    void set_handle(cublasHandle_t* ph) { h = ph; }

    RastriginTest(int n) { CUDA_CHECK(cudaMalloc(&d_temp_f, n * sizeof(T))); }
    ~RastriginTest() { if (d_temp_f) CUDA_CHECK(cudaFree(d_temp_f)); }

    double f(T* d_x, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rastrigin_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float sum_part = gpu_sum_f32_cublas(*h, (float*)d_temp_f, n);
        return (double)((T)10.0 * (T)n + (T)sum_part);
    }

    void df(T* d_x, T* d_g, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rastrigin_kernel<<<blocks, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

/**
 * @brief Kernel computing Ackley function contributions and gradient.
 *
 * Per-element outputs are written to `f_vals` and `g` when those pointers are
 * provided. The implementation uses a stabilized derivative for the radial
 * term to avoid dividing by zero.
 *
 * @tparam T Floating point type.
 * @param x Input device vector.
 * @param f_vals Per-element function contributions (optional).
 * @param g Gradient output (optional).
 * @param n Number of elements.
 */
template <typename T>
__global__ void ackley_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T xi = x[i];

        // f_i = -20 exp(-0.2 |x|) - exp(cos(2pi x)) + 20 + e
        T ax = fabs(xi);
        T exp1 = exp((T)-0.2 * ax);
        T c = cos((T)2.0 * (T)M_PI * xi);
        T exp2 = exp(c);

        if (f_vals) {
            f_vals[i] = (T)-20.0 * exp1 - exp2 + (T)20.0 + (T)M_E;
        }

        if (g) {
            // d/dx [-20 exp(-0.2|x|)] = 4 * x * exp(-0.2|x|) / max(|x|,eps)
            // (this matches your earlier stabilized form; avoids 0/0)
            T r = ax;
            if (r < (T)1e-12) r = (T)1e-12;
            T g1 = (T)4.0 * xi * exp1 / r;

            // d/dx [-exp(cos(2pi x))] = + 2pi * sin(2pi x) * exp(cos(2pi x))
            T s = sin((T)2.0 * (T)M_PI * xi);
            T g2 = (T)2.0 * (T)M_PI * s * exp2;

            g[i] = g1 + g2;
        }
    }
}

/**
 * @brief Wrapper for the Ackley test function.
 *
 * Executes the `ackley_kernel` and reduces per-element contributions via
 * cuBLAS to produce the scalar objective value.
 *
 * @tparam T Floating point type.
 */
template <typename T>
struct AckleyTest {
    T* d_temp_f = nullptr;
    cublasHandle_t* h = nullptr;

    void set_handle(cublasHandle_t* ph) { h = ph; }

    AckleyTest(int n) { CUDA_CHECK(cudaMalloc(&d_temp_f, n * sizeof(T))); }
    ~AckleyTest() { if (d_temp_f) CUDA_CHECK(cudaFree(d_temp_f));}

    double f(T* d_x, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ackley_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float result = gpu_sum_f32_cublas(*h, (float*)d_temp_f, n);
        return (double)result;
    }

    void df(T* d_x, T* d_g, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ackley_kernel<<<blocks, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

/*----------------------------------------- L-BFGS -----------------------------------------*/

/**
 * @brief L-BFGS optimizer that uses cuBLAS for vector operations and evaluates
 * objective/gradient on the GPU via the provided `Func` wrapper.
 *
 * The `Func` type must provide `set_handle(cublasHandle_t*)`, `f(d_x,n)`, and
 * `df(d_x,d_g,n)` methods so the optimizer can call into GPU-based functions.
 */
template <typename Func>
class lbfgs {
    float* d_x = nullptr;
    float* d_x_old = nullptr;
    float* d_g = nullptr;
    float* d_g_old = nullptr;
    float* d_d = nullptr;
    float* d_q = nullptr;
    float* d_r = nullptr;
    float* d_S = nullptr;
    float* d_Y = nullptr;

    float* rho = nullptr;     // host
    float* alpha_hist = nullptr; // host

    cublasHandle_t handle{};

    /**
     * @brief Allocate device memory and initialize cuBLAS handle.
     *
     * @param n Problem dimension.
     * @param m L-BFGS history size.
     */
    void init(std::size_t n, std::size_t m) {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_old, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_g, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_g_old, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_d, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_S, n * m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Y, n * m * sizeof(float)));

        rho = new float[m];
        alpha_hist = new float[m];
        for (std::size_t i = 0; i < m; ++i) { rho[i] = 0.0f; alpha_hist[i] = 0.0f; }
    }

    /**
     * @brief Free device memory and destroy cuBLAS handle.
     */
    void destroy() {
        if (d_x) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_x));
        }
    
        if (d_x_old) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_x_old));
        }
    
        if (d_g) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_g));
        }
    
        if (d_g_old) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_g_old));
        }
        
        if (d_d) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_d));
        }
        
        if (d_q) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_q));
        }
        
        if (d_r) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_r));
        }
        
        if (d_S) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_S));
        }
        
        if (d_Y) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_Y));
        }
        delete[] rho;
        delete[] alpha_hist;
        CUBLAS_CHECK(cublasDestroy(handle));

        d_x = d_x_old = d_g = d_g_old = d_d = d_q = d_r = d_S = d_Y = nullptr;
        rho = alpha_hist = nullptr;
    }

public:
    /**
     * @brief Run the L-BFGS optimization loop.
     *
     * @param problem_size Dimensionality `n`.
     * @param memory_size History `m`.
     * @param x0 Host pointer to initial guess; optimized solution is written back.
     * @param max_itr Maximum iterations.
     * @param func Objective wrapper implementing `f` and `df`.
     * @param eps Convergence tolerance for gradient norm.
     * @return double Final objective value.
     */
    double operator()(std::size_t problem_size, std::size_t memory_size, float* x0,
                      std::size_t max_itr, Func& func, float eps = 1e-9f) {

        init(problem_size, memory_size);
        func.set_handle(&handle);

        const int n = (int)problem_size;
        const int m = (int)memory_size;

        CUDA_CHECK(cudaMemcpy(d_x, x0, n * sizeof(float), cudaMemcpyHostToDevice));

        float val = (float)func.f(d_x, n);
        func.df(d_x, d_g, n);

        int mem = 0;              // number of valid (s,y) pairs
        bool restart = false;

        const float neg_one = -1.0f;
        const float one = 1.0f;

        for (int itr = 0; itr < (int)max_itr; ++itr) {
            // Optional NaN check (uncomment for debugging)
            /*
            int* d_bad = nullptr;
            CUDA_CHECK(cudaMalloc(&d_bad, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(int)));
            count_nan_inf<<<(n + 255)/256, 256>>>(d_g, n, d_bad);
            CUDA_CHECK(cudaGetLastError());
            int h_bad = 0;
            CUDA_CHECK(cudaMemcpy(&h_bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_bad));
            if (h_bad) {
                std::fprintf(stderr, "NaN/Inf in gradient at itr=%d\n", itr);
                break;
            }
            */

            float g_norm = 0.0f;
            CUBLAS_CHECK(cublasSnrm2(handle, n, d_g, 1, &g_norm));

            if (g_norm / (std::fabs(val) + 1.0f) <= eps) break;

            // Compute direction d
            if (restart || mem == 0) {
                restart = false;
                // d = -g
                CUBLAS_CHECK(cublasScopy(handle, n, d_g, 1, d_d, 1));
                CUBLAS_CHECK(cublasSscal(handle, n, &neg_one, d_d, 1));
            } else {
                // Clear host arrays each iteration to avoid stale slots
                for (int j = 0; j < m; ++j) { rho[j] = 0.0f; alpha_hist[j] = 0.0f; }

                int bound = (mem < m) ? mem : m;

                // q = g
                CUBLAS_CHECK(cublasScopy(handle, n, d_g, 1, d_q, 1));

                // Backward loop: i = bound-1..0 uses oldest->newest mapping
                for (int i = bound - 1; i >= 0; --i) {
                    int idx = (itr - 1 - i) % m;   // circular index of the i-th pair
                    if (idx < 0) idx += m;

                    float* s = d_S + idx * n;
                    float* y = d_Y + idx * n;

                    float sTq = 0.0f;
                    float yTs = 0.0f;
                    CUBLAS_CHECK(cublasSdot(handle, n, s, 1, d_q, 1, &sTq));
                    CUBLAS_CHECK(cublasSdot(handle, n, y, 1, s, 1, &yTs)); // y^T s

                    if (!std::isfinite(yTs) || std::fabs(yTs) < 1e-20f) {
                        rho[idx] = 0.0f;
                        alpha_hist[idx] = 0.0f;
                        continue;
                    }

                    rho[idx] = 1.0f / yTs;
                    alpha_hist[idx] = rho[idx] * sTq;

                    float neg_ai = -alpha_hist[idx];
                    CUBLAS_CHECK(cublasSaxpy(handle, n, &neg_ai, y, 1, d_q, 1)); // q -= a_i*y_i
                }

                // H0 scaling from most recent pair: last = (itr-1)%m
                int last = (itr - 1) % m;
                if (last < 0) last += m;

                float H0 = 1.0f;
                {
                    float* ylast = d_Y + last * n;
                    float* slast = d_S + last * n;
                    float ys = 0.0f, yy = 0.0f;
                    CUBLAS_CHECK(cublasSdot(handle, n, slast, 1, ylast, 1, &ys));
                    CUBLAS_CHECK(cublasSdot(handle, n, ylast, 1, ylast, 1, &yy));
                    if (std::isfinite(ys) && std::isfinite(yy) && std::fabs(yy) > 1e-20f) H0 = ys / yy;
                }

                // r = H0 * q
                CUBLAS_CHECK(cublasScopy(handle, n, d_q, 1, d_r, 1));
                CUBLAS_CHECK(cublasSscal(handle, n, &H0, d_r, 1));

                // Forward loop
                for (int i = 0; i < bound; ++i) {
                    int idx = (itr - bound + i) % m;
                    if (idx < 0) idx += m;
                    if (rho[idx] == 0.0f) continue;

                    float* s = d_S + idx * n;
                    float* y = d_Y + idx * n;

                    float yTr = 0.0f;
                    CUBLAS_CHECK(cublasSdot(handle, n, y, 1, d_r, 1, &yTr));

                    float beta = rho[idx] * yTr;
                    float coeff = alpha_hist[idx] - beta;

                    CUBLAS_CHECK(cublasSaxpy(handle, n, &coeff, s, 1, d_r, 1)); // r += (a-beta)s
                }

                // d = -r
                CUBLAS_CHECK(cublasScopy(handle, n, d_r, 1, d_d, 1));
                CUBLAS_CHECK(cublasSscal(handle, n, &neg_one, d_d, 1));
            }

            // g·d (for Armijo)
            float g_dot_d = 0.0f;
            CUBLAS_CHECK(cublasSdot(handle, n, d_g, 1, d_d, 1, &g_dot_d));

            // If not descent, restart
            if (!std::isfinite(g_dot_d) || g_dot_d >= 0.0f) {
                restart = true;
                mem = 0;
                // d = -g
                CUBLAS_CHECK(cublasScopy(handle, n, d_g, 1, d_d, 1));
                CUBLAS_CHECK(cublasSscal(handle, n, &neg_one, d_d, 1));
                CUBLAS_CHECK(cublasSdot(handle, n, d_g, 1, d_d, 1, &g_dot_d));
                if (!std::isfinite(g_dot_d) || g_dot_d >= 0.0f) break;
            }

            // Save old state
            CUBLAS_CHECK(cublasScopy(handle, n, d_x, 1, d_x_old, 1));
            CUBLAS_CHECK(cublasScopy(handle, n, d_g, 1, d_g_old, 1));

            // Armijo backtracking line search
            const float c1 = 1e-4f;
            const float decay = 0.5f;
            float step = 1.0f;
            bool success = false;

            for (int ls = 0; ls < 25; ++ls) {
                // x = x_old + step*d
                CUBLAS_CHECK(cublasScopy(handle, n, d_x_old, 1, d_x, 1));
                CUBLAS_CHECK(cublasSaxpy(handle, n, &step, d_d, 1, d_x, 1));

                float f_new = (float)func.f(d_x, n);

                // Armijo condition: f_new <= val + c1*step*(g·d)
                float rhs = val + c1 * step * g_dot_d;

                if (std::isfinite(f_new) && f_new <= rhs) {
                    val = f_new;
                    success = true;
                    break;
                }
                step *= decay;
            }

            if (!success) {
                // Hard restart; if still fails, terminate
                restart = true;
                mem = 0;

                // Try a couple of larger SD steps (escape flat/overshoot regions)
                CUBLAS_CHECK(cublasScopy(handle, n, d_g_old, 1, d_d, 1));
                CUBLAS_CHECK(cublasSscal(handle, n, &neg_one, d_d, 1));

                float step_fb = 1.0f;
                bool ok = false;
                for (int t = 0; t < 20; ++t) {
                    CUBLAS_CHECK(cublasScopy(handle, n, d_x_old, 1, d_x, 1));
                    CUBLAS_CHECK(cublasSaxpy(handle, n, &step_fb, d_d, 1, d_x, 1));
                    float f_try = (float)func.f(d_x, n);
                    if (std::isfinite(f_try) && f_try < val) {
                        val = f_try;
                        ok = true;
                        break;
                    }
                    step_fb *= 2.0f;
                    if (step_fb > 100.0f) break;
                }
                if (!ok) break;

                // Update gradient after fallback accept
                func.df(d_x, d_g, n);
                continue;
            }

            // New gradient at accepted x
            func.df(d_x, d_g, n);

            // Update history: s = x - x_old, y = g - g_old
            int pos = itr % m;
            compute_sy_kernel<<<(n + 255) / 256, 256>>>(n,
                d_S + pos * n, d_Y + pos * n,
                d_x, d_x_old, d_g, d_g_old);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Curvature check: y^T s should be positive and not tiny
            float yTs = 0.0f;
            CUBLAS_CHECK(cublasSdot(handle, n, d_Y + pos * n, 1, d_S + pos * n, 1, &yTs));

            if (std::isfinite(yTs) && yTs > 1e-12f) {
                if (mem < m) mem++;
            } else {
                // mark pair effectively inactive by not increasing mem; it will be overwritten anyway
                // (keeping mem as-is prevents "using" a bad pair count)
            }
        }

        CUDA_CHECK(cudaMemcpy(x0, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
        destroy();
        return (double)val;
    }
};

/*------------------------------------------ Main ------------------------------------------*/

/**
 * @brief Program entry: runs several benchmark tests (Quadratic, Rosenbrock,
 * Rastrigin, Ackley) using the L-BFGS harness and reports timing and final values.
 *
 * Initializes problem vectors, runs the optimizer for each test, and prints
 * results to stdout.
 */
int main() {
    using T = float;

    int N = 1 << 12;  // Problem size
    int M = 10;       // History size
    T* x0 = new T[N];

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    float elapsedTime = 0.0f;

    // --- TEST 1: QUADRATIC ---
    for (int i = 0; i < N; i++) x0[i] = 8.0f;
    lbfgs<QuadraticTest<T>> opt_quad;
    QuadraticTest<T> quad(N);

    std::printf("Starting: Quadratic...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    double final_f = opt_quad(N, M, x0, 1000, quad, 1e-6f);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    std::printf("Quadratic Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    // --- TEST 2: ROSENBROCK ---
    for (int i = 0; i < N; i++) x0[i] = -4.2f;
    lbfgs<RosenbrockNoAtomTest<T>> opt_rosen;
    RosenbrockNoAtomTest<T> rosen(N);

    std::printf("Starting: Rosenbrock...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    final_f = opt_rosen(N, M, x0, 500, rosen, 1e-6f);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    std::printf("Rosenbrock Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    // --- TEST 3: RASTRIGIN ---
    for (int i = 0; i < N; i++) x0[i] = -1.2f;
    lbfgs<RastriginTest<T>> opt_rastrigin;
    RastriginTest<T> rastrigin(N);

    std::printf("Starting: Rastrigin...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    final_f = opt_rastrigin(N, M, x0, 5000, rastrigin, 1e-6f);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    std::printf("Rastrigin Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    // --- TEST 4: ACKLEY ---
    for (int i = 0; i < N; i++) x0[i] = -4.0f;
    lbfgs<AckleyTest<T>> opt_ackley;
    AckleyTest<T> ackley(N);

    std::printf("Starting: Ackley...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    final_f = opt_ackley(N, M, x0, 5000, ackley, 1e-6f);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    std::printf("Ackley Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    delete[] x0;

    // clean up global ones buffer
    if (d_ones) CUDA_CHECK(cudaFree(d_ones));
    d_ones = nullptr;
    ones_cap = 0;

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    return 0;
}
