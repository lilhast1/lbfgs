// lbfgs_mixed_precision.cu
//
// Adds templating + FP16 support (recommended mode: FP16 storage with FP32 accumulation).
//
// Build (example):
//   nvcc -O3 -std=c++17 lbfgs_mixed_precision.cu -o lbfgs
//
// Switch precision by changing Scalar below:
//   using Scalar = double;   // FP64 baseline
//   using Scalar = float;    // FP32
//   using Scalar = __half;   // FP16 storage + FP32 accum (recommended)
//
// NOTE:
// - For FP16, we accumulate dot/reductions in float (Accum = float).
// - Objective/gradient kernels are templated for double/float and have FP16-specialized versions.
// - Here, f(x) is evaluated on GPU consistently
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BLOCK_SIZE 64

// =========================
//  Precision / Traits
// =========================
template <typename T>
struct NumericTraits;

template <>
struct NumericTraits<double> {
    using Accum = double;
    __host__ __device__ static constexpr double zero() { return 0.0; }
    __host__ __device__ static constexpr double one() { return 1.0; }
};

template <>
struct NumericTraits<float> {
    using Accum = float;
    __host__ __device__ static constexpr float zero() { return 0.0f; }
    __host__ __device__ static constexpr float one() { return 1.0f; }
};

template <>
struct NumericTraits<__half> {
    using Accum = float;  // IMPORTANT: accumulate in FP32
    __host__ __device__ static __half zero() { return __float2half(0.0f); }
    __host__ __device__ static __half one() { return __float2half(1.0f); }
};

// Scalar conversions (device-safe)
__host__ __device__ inline double to_double(double x) { return x; }
__host__ __device__ inline double to_double(float x) { return (double)x; }
__host__ __device__ inline double to_double(__half x) { return (double)__half2float(x); }

__host__ __device__ inline float to_float(double x) { return (float)x; }
__host__ __device__ inline float to_float(float x) { return x; }
__host__ __device__ inline float to_float(__half x) { return __half2float(x); }

__host__ __device__ inline double from_double(double x) { return x; }
__host__ __device__ inline float from_double_f(float x) { return x; }
__host__ __device__ inline __half from_float_half(float x) { return __float2half(x); }

// Generic scalar multiply-add for device code
template <typename T>
__host__ __device__ inline T madd(T a, T b, T c) {
    return a * b + c;
}
template <>
__host__ __device__ inline __half madd(__half a, __half b, __half c) {
    float af = __half2float(a);
    float bf = __half2float(b);
    float cf = __half2float(c);
    return __float2half(af * bf + cf);
}

// =========================
//  Error checking
// =========================
inline void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(e) << std::endl;
        throw std::runtime_error("CUDA failure");
    }
}

// =========================
//  Datatypes (templated)
// =========================
template <typename T>
struct GpuMatrix {
    int n, m;
    T* elems;
    GpuMatrix(int n, int m) : n(n), m(m), elems(nullptr) {
        cudaCheck(cudaMalloc(&elems, (size_t)n * (size_t)m * sizeof(T)), "cudaMalloc GpuMatrix");
    }
    ~GpuMatrix() { cudaFree(elems); }
};

template <typename T>
struct GpuVector {
    int n;
    T* elems;
    GpuVector(int n) : n(n), elems(nullptr) {
        cudaCheck(cudaMalloc(&elems, (size_t)n * sizeof(T)), "cudaMalloc GpuVector");
    }
    ~GpuVector() { cudaFree(elems); }
};

template <typename T>
struct UnifiedVector {
    int n;
    T* elems;
    UnifiedVector(int n) : n(n), elems(nullptr) {
        cudaCheck(cudaMallocManaged(&elems, (size_t)n * sizeof(T)), "cudaMallocManaged UnifiedVector");
    }
    ~UnifiedVector() { cudaFree(elems); }
};

// Dot lookup table stores partial sums in Accum (float for half)
template <typename Scalar>
struct DotLookupTable {
    using Acc = typename NumericTraits<Scalar>::Accum;

    Acc* d_partial;
    Acc* h_partial;
    int max_grid_size;

    DotLookupTable(int n) : d_partial(nullptr), h_partial(nullptr), max_grid_size(0) {
        max_grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaCheck(cudaMalloc(&d_partial, (size_t)max_grid_size * sizeof(Acc)), "cudaMalloc d_partial");
        h_partial = (Acc*)malloc((size_t)max_grid_size * sizeof(Acc));
        if (!h_partial) throw std::bad_alloc();
        clean();
    }
    ~DotLookupTable() {
        cudaFree(d_partial);
        free(h_partial);
    }

    void clean() {
        cudaMemset(d_partial, 0, (size_t)max_grid_size * sizeof(Acc));
        memset(h_partial, 0, (size_t)max_grid_size * sizeof(Acc));
    }
};

struct KernelConfig {
    int blockSize;
    int gridSize;

    KernelConfig(int n) : blockSize(256), gridSize(0) {
        int numSMs = 0;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        gridSize = 32 * numSMs;
        int maxBlocks = (n + blockSize - 1) / blockSize;
        if (gridSize > maxBlocks) gridSize = maxBlocks;
        if (gridSize < 1) gridSize = 1;
    }
};

// =========================
//  Kernels (templated)
// =========================

// y = y + alpha * x
template <typename T>
__global__ void axpy_kernel(T alpha, const T* x, T* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = madd(alpha, x[i], y[i]);
    }
}

// r = factor * q
template <typename T>
__global__ void setVectorScalar_kernel(T* r, const T* q, T factor, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        // r[i] = q[i] * factor
        if constexpr (std::is_same<T, __half>::value) {
            float qi = __half2float(q[i]);
            float fi = __half2float(factor);
            r[i] = __float2half(qi * fi);
        } else {
            r[i] = q[i] * factor;
        }
    }
}

// Dot product partial reduction kernel
template <typename Scalar>
__global__ void dotProduct_kernel(
    const Scalar* a,
    const Scalar* b,
    typename NumericTraits<Scalar>::Accum* c,
    int n
) {
    using Acc = typename NumericTraits<Scalar>::Accum;
    __shared__ Acc buf[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Acc sum = (Acc)0;
    while (tid < n) {
        if constexpr (std::is_same<Scalar, __half>::value) {
            float af = __half2float(a[tid]);
            float bf = __half2float(b[tid]);
            sum += (Acc)(af * bf);
        } else {
            sum += (Acc)(a[tid] * b[tid]);
        }
        tid += gridDim.x * blockDim.x;
    }

    buf[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) c[blockIdx.x] = buf[0];
}

// Sum reduction to single Acc using atomicAdd
template <typename Acc>
__global__ void sum_reduction_kernel_acc(const Acc* input, Acc* output, int n) {
    __shared__ Acc sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Acc val = (i < n) ? input[i] : (Acc)0;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// =========================
//  Wrappers: gpu_sum, dot
// =========================
template <typename Acc>
Acc gpu_sum_acc(const Acc* d_elements, int n) {
    static Acc* d_res = nullptr;
    if (!d_res) cudaCheck(cudaMalloc(&d_res, sizeof(Acc)), "cudaMalloc d_res");

    cudaCheck(cudaMemset(d_res, 0, sizeof(Acc)), "cudaMemset d_res");
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_reduction_kernel_acc<<<blocks, BLOCK_SIZE>>>(d_elements, d_res, n);
    cudaCheck(cudaGetLastError(), "sum_reduction_kernel_acc launch");

    Acc h_res;
    cudaCheck(cudaMemcpy(&h_res, d_res, sizeof(Acc), cudaMemcpyDeviceToHost), "cudaMemcpy d_res->host");
    return h_res;
}

template <typename Scalar>
typename NumericTraits<Scalar>::Accum
dot(const Scalar* a, const Scalar* b, int n, DotLookupTable<Scalar>* context = nullptr) {
    using Acc = typename NumericTraits<Scalar>::Accum;

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    if (gridSize.x < 1) gridSize.x = 1;

    size_t partial_results_size = (size_t)gridSize.x * sizeof(Acc);

    Acc *d_c = nullptr, *h_c = nullptr;

    if (!context) {
        h_c = (Acc*)malloc(partial_results_size);
        if (!h_c) throw std::bad_alloc();
        cudaCheck(cudaMalloc(&d_c, partial_results_size), "cudaMalloc d_c partial");
        dotProduct_kernel<<<gridSize, blockSize>>>(a, b, d_c, n);
        cudaCheck(cudaGetLastError(), "dotProduct_kernel launch");
        cudaCheck(cudaMemcpy(h_c, d_c, partial_results_size, cudaMemcpyDeviceToHost), "cudaMemcpy partial->host");
    } else {
        dotProduct_kernel<<<gridSize, blockSize>>>(a, b, context->d_partial, n);
        cudaCheck(cudaGetLastError(), "dotProduct_kernel (context) launch");
        cudaCheck(cudaMemcpy(context->h_partial, context->d_partial, partial_results_size, cudaMemcpyDeviceToHost),
                  "cudaMemcpy ctx partial->host");
        h_c = context->h_partial;
    }

    Acc result = (Acc)0;
    for (int i = 0; i < (int)gridSize.x; i++) result += h_c[i];

    if (!context) {
        cudaFree(d_c);
        free(h_c);
    } else {
        context->clean();
    }

    return result;
}

// =========================
//  Test functions (templated)
// =========================

template <typename Scalar>
__global__ void quadratic_kernel(const Scalar* x, typename NumericTraits<Scalar>::Accum* f_vals, Scalar* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = to_float(x[i]);
        if (f_vals) f_vals[i] = (typename NumericTraits<Scalar>::Accum)(xi * xi);
        if (g) {
            float gi = 2.0f * xi;
            if constexpr (std::is_same<Scalar, __half>::value) g[i] = __float2half(gi);
            else g[i] = (Scalar)gi;
        }
    }
}

template <typename Scalar>
struct QuadraticTest {
    using Acc = typename NumericTraits<Scalar>::Accum;
    Acc* d_temp_f;
    int n;

    QuadraticTest(int n) : d_temp_f(nullptr), n(n) {
        cudaCheck(cudaMalloc(&d_temp_f, (size_t)n * sizeof(Acc)), "cudaMalloc Quadratic d_temp_f");
    }
    ~QuadraticTest() { cudaFree(d_temp_f); }

    Acc f(const Scalar* d_x, int n_) {
        (void)n_;
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quadratic_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, nullptr, n);
        cudaCheck(cudaGetLastError(), "quadratic_kernel f launch");
        return gpu_sum_acc(d_temp_f, n);
    }

    void df(const Scalar* d_x, Scalar* d_g, int n_) {
        (void)n_;
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quadratic_kernel<<<blocks, BLOCK_SIZE>>>(d_x, nullptr, d_g, n);
        cudaCheck(cudaGetLastError(), "quadratic_kernel df launch");
    }
};

template <typename Scalar>
__global__ void rosen_kernel_t(const Scalar* x, typename NumericTraits<Scalar>::Accum* f_vals, Scalar* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        float x_curr = to_float(x[i]);
        float x_next = to_float(x[i + 1]);
        float t1 = (x_next - x_curr * x_curr);
        float t2 = (1.0f - x_curr);

        if (f_vals) f_vals[i] = (typename NumericTraits<Scalar>::Accum)(100.0f * t1 * t1 + t2 * t2);

        if (g) {
            // Atomic adds in Accum space, then cast back is messy; for simplicity, do atomic on float for FP16/FP32,
            // and on double for FP64.
            if constexpr (std::is_same<Scalar, double>::value) {
                atomicAdd((double*)&g[i], (double)(-400.0f * x_curr * t1 - 2.0f * t2));
                atomicAdd((double*)&g[i + 1], (double)(200.0f * t1));
            } else {
                // For float/half, keep a float gradient buffer (handled in test struct)
                // This path should not be used directly.
            }
        }
    }
}

template <typename Scalar>
__global__ void rosen_grad_float_kernel(const Scalar* x, float* g_float, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        float x_curr = to_float(x[i]);
        float x_next = to_float(x[i + 1]);
        float t1 = (x_next - x_curr * x_curr);
        float t2 = (1.0f - x_curr);

        atomicAdd(&g_float[i], -400.0f * x_curr * t1 - 2.0f * t2);
        atomicAdd(&g_float[i + 1], 200.0f * t1);
    }
}

template <typename Scalar>
__global__ void cast_float_to_scalar_kernel(const float* src, Scalar* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = src[i];
        if constexpr (std::is_same<Scalar, __half>::value) dst[i] = __float2half(v);
        else dst[i] = (Scalar)v;
    }
}

template <typename Scalar>
struct RosenbrockTest {
    using Acc = typename NumericTraits<Scalar>::Accum;
    Acc* d_temp_f;
    float* d_g_float;  // for half/float gradients
    int n;

    RosenbrockTest(int n) : d_temp_f(nullptr), d_g_float(nullptr), n(n) {
        cudaCheck(cudaMalloc(&d_temp_f, (size_t)n * sizeof(Acc)), "cudaMalloc Rosen d_temp_f");
        if constexpr (!std::is_same<Scalar, double>::value) {
            cudaCheck(cudaMalloc(&d_g_float, (size_t)n * sizeof(float)), "cudaMalloc Rosen d_g_float");
        }
    }
    ~RosenbrockTest() {
        cudaFree(d_temp_f);
        if (d_g_float) cudaFree(d_g_float);
    }

    Acc f(const Scalar* d_x, int n_) {
        (void)n_;
        cudaCheck(cudaMemset(d_temp_f, 0, (size_t)n * sizeof(Acc)), "cudaMemset Rosen d_temp_f");
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rosen_kernel_t<<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, (Scalar*)nullptr, n);
        cudaCheck(cudaGetLastError(), "rosen_kernel_t f launch");
        return gpu_sum_acc(d_temp_f, n);
    }

    void df(const Scalar* d_x, Scalar* d_g, int n_) {
        (void)n_;
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if constexpr (std::is_same<Scalar, double>::value) {
            cudaCheck(cudaMemset(d_g, 0, (size_t)n * sizeof(Scalar)), "cudaMemset Rosen g");
            // For double only, reuse rosen_kernel_t atomic on g
            rosen_kernel_t<<<blocks, BLOCK_SIZE>>>(d_x, (Acc*)nullptr, d_g, n);
            cudaCheck(cudaGetLastError(), "rosen_kernel_t df launch");
        } else {
            cudaCheck(cudaMemset(d_g_float, 0, (size_t)n * sizeof(float)), "cudaMemset Rosen g_float");
            rosen_grad_float_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_g_float, n);
            cudaCheck(cudaGetLastError(), "rosen_grad_float_kernel launch");
            cast_float_to_scalar_kernel<<<blocks, BLOCK_SIZE>>>(d_g_float, d_g, n);
            cudaCheck(cudaGetLastError(), "cast_float_to_scalar_kernel launch");
        }
    }
};

template <typename Scalar>
__global__ void rastrigin_kernel_t(const Scalar* x, typename NumericTraits<Scalar>::Accum* f_vals, Scalar* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = to_float(x[i]);
        if (f_vals) f_vals[i] = (typename NumericTraits<Scalar>::Accum)(xi * xi - 10.0f * cosf(2.0f * (float)M_PI * xi));
        if (g) {
            float gi = 2.0f * xi + 20.0f * (float)M_PI * sinf(2.0f * (float)M_PI * xi);
            if constexpr (std::is_same<Scalar, __half>::value) g[i] = __float2half(gi);
            else g[i] = (Scalar)gi;
        }
    }
}

template <typename Scalar>
struct RastriginTest {
    using Acc = typename NumericTraits<Scalar>::Accum;
    Acc* d_temp_f;
    int n;

    RastriginTest(int n) : d_temp_f(nullptr), n(n) {
        cudaCheck(cudaMalloc(&d_temp_f, (size_t)n * sizeof(Acc)), "cudaMalloc Rastrigin d_temp_f");
    }
    ~RastriginTest() { cudaFree(d_temp_f); }

    Acc f(const Scalar* d_x, int n_) {
        (void)n_;
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rastrigin_kernel_t<<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, (Scalar*)nullptr, n);
        cudaCheck(cudaGetLastError(), "rastrigin_kernel_t f launch");
        Acc base = (Acc)(10.0f * (float)n);
        return base + gpu_sum_acc(d_temp_f, n);
    }

    void df(const Scalar* d_x, Scalar* d_g, int n_) {
        (void)n_;
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rastrigin_kernel_t<<<blocks, BLOCK_SIZE>>>(d_x, (Acc*)nullptr, d_g, n);
        cudaCheck(cudaGetLastError(), "rastrigin_kernel_t df launch");
    }
};

// =========================
//  L-BFGS (templated)
// =========================
template <typename Scalar, typename Func>
typename NumericTraits<Scalar>::Accum
lbfgs(int n, int m, Scalar* x0, int max_itr, Func func, typename NumericTraits<Scalar>::Accum eps) {
    using Acc = typename NumericTraits<Scalar>::Accum;

    UnifiedVector<Scalar> x(n), g(n);
    GpuVector<Scalar> x_old(n), g_old(n), d(n), q(n), r(n);
    GpuMatrix<Scalar> S(n, m), Y(n, m);

    // Host history scalars in Acc
    Acc* rho = new Acc[m];
    Acc* alpha_hist = new Acc[m];
    for (int i = 0; i < m; i++) {
        rho[i] = (Acc)0;
        alpha_hist[i] = (Acc)0;
    }

    DotLookupTable<Scalar> dw(n);
    KernelConfig cfg(n);

    cudaCheck(cudaMemcpy(x.elems, x0, (size_t)n * sizeof(Scalar), cudaMemcpyHostToDevice), "cudaMemcpy x0->x");

    Acc val = func.f(x.elems, n);
    func.df(x.elems, g.elems, n);

    for (int k = 0; k < max_itr; k++) {
        Acc g_norm = (Acc)std::sqrt((double)dot(x.elems ? g.elems : g.elems, g.elems, n, &dw));
        if (g_norm < eps) break;

        // 1) Search direction
        if (k == 0) {
            // d = -g
            if constexpr (std::is_same<Scalar, __half>::value) {
                setVectorScalar_kernel<<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, __float2half(-1.0f), n);
            } else {
                setVectorScalar_kernel<<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, (Scalar)-1, n);
            }
            cudaCheck(cudaGetLastError(), "setVectorScalar_kernel d=-g");
        } else {
            // q = g
            cudaCheck(cudaMemcpy(q.elems, g.elems, (size_t)n * sizeof(Scalar), cudaMemcpyDeviceToDevice), "cudaMemcpy g->q");

            int bound = (k > m) ? m : k;

            for (int i = bound - 1; i >= 0; --i) {
                int idx = (k - 1 - i) % m;
                Acc s_dot_q = dot(S.elems + (size_t)idx * n, q.elems, n, &dw);
                alpha_hist[idx] = rho[idx] * s_dot_q;

                // q = q - alpha[idx]*Y[idx]
                Scalar a;
                if constexpr (std::is_same<Scalar, __half>::value) a = __float2half(-(float)alpha_hist[idx]);
                else a = (Scalar)(-alpha_hist[idx]);
                axpy_kernel<<<cfg.gridSize, cfg.blockSize>>>(a, Y.elems + (size_t)idx * n, q.elems, n);
                cudaCheck(cudaGetLastError(), "axpy_kernel q update");
            }

            int last_idx = (k - 1) % m;
            Acc s_dot_y = dot(S.elems + (size_t)last_idx * n, Y.elems + (size_t)last_idx * n, n, &dw);
            Acc y_dot_y = dot(Y.elems + (size_t)last_idx * n, Y.elems + (size_t)last_idx * n, n, &dw);
            Acc gamma = (y_dot_y > (Acc)1e-18) ? (s_dot_y / y_dot_y) : (Acc)1;

            // r = gamma * q
            Scalar gsc;
            if constexpr (std::is_same<Scalar, __half>::value) gsc = __float2half((float)gamma);
            else gsc = (Scalar)gamma;
            setVectorScalar_kernel<<<cfg.gridSize, cfg.blockSize>>>(r.elems, q.elems, gsc, n);
            cudaCheck(cudaGetLastError(), "setVectorScalar_kernel r=gamma*q");

            for (int i = 0; i < bound; ++i) {
                int idx = (k - bound + i) % m;
                Acc y_dot_r = dot(Y.elems + (size_t)idx * n, r.elems, n, &dw);
                Acc beta = rho[idx] * y_dot_r;
                Acc coeff = alpha_hist[idx] - beta;

                Scalar c;
                if constexpr (std::is_same<Scalar, __half>::value) c = __float2half((float)coeff);
                else c = (Scalar)coeff;

                axpy_kernel<<<cfg.gridSize, cfg.blockSize>>>(c, S.elems + (size_t)idx * n, r.elems, n);
                cudaCheck(cudaGetLastError(), "axpy_kernel r update");
            }

            // d = -r
            if constexpr (std::is_same<Scalar, __half>::value) {
                setVectorScalar_kernel<<<cfg.gridSize, cfg.blockSize>>>(d.elems, r.elems, __float2half(-1.0f), n);
            } else {
                setVectorScalar_kernel<<<cfg.gridSize, cfg.blockSize>>>(d.elems, r.elems, (Scalar)-1, n);
            }
            cudaCheck(cudaGetLastError(), "setVectorScalar_kernel d=-r");
        }

        // 2) Backtracking line search (GPU evaluation)
        Acc g_dot_d = dot(g.elems, d.elems, n, &dw);

        // if not descent, fallback to steepest descent
        if (g_dot_d >= (Acc)0) {
            if constexpr (std::is_same<Scalar, __half>::value) {
                setVectorScalar_kernel<<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, __float2half(-1.0f), n);
            } else {
                setVectorScalar_kernel<<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, (Scalar)-1, n);
            }
            cudaCheck(cudaGetLastError(), "reset d=-g");
            g_dot_d = dot(g.elems, d.elems, n, &dw);
        }

        Acc step = (Acc)1;
        const Acc c1 = (Acc)1e-4;
        const Acc decay = (Acc)0.5;
        bool success = false;

        cudaCheck(cudaMemcpy(x_old.elems, x.elems, (size_t)n * sizeof(Scalar), cudaMemcpyDeviceToDevice), "x->x_old");
        cudaCheck(cudaMemcpy(g_old.elems, g.elems, (size_t)n * sizeof(Scalar), cudaMemcpyDeviceToDevice), "g->g_old");

        for (int ls_iter = 0; ls_iter < 25; ls_iter++) {
            // x = x_old + step*d
            cudaCheck(cudaMemcpy(x.elems, x_old.elems, (size_t)n * sizeof(Scalar), cudaMemcpyDeviceToDevice), "x_old->x");
            Scalar st;
            if constexpr (std::is_same<Scalar, __half>::value) st = __float2half((float)step);
            else st = (Scalar)step;

            axpy_kernel<<<cfg.gridSize, cfg.blockSize>>>(st, d.elems, x.elems, n);
            cudaCheck(cudaGetLastError(), "axpy_kernel line search x");

            // Evaluate f(x) on GPU
            Acc f_new = func.f(x.elems, n);

            if (f_new <= val + c1 * step * g_dot_d) {
                val = f_new;
                success = true;
                break;
            }
            step *= decay;
        }

        if (!success) {
            std::cout << "Line Search Failed! Resetting L-BFGS history..." << std::endl;
            k = -1;
            continue;
        }

        // 3) Update gradient
        func.df(x.elems, g.elems, n);

        int cur = k % m;

        // S_cur = x - x_old
        cudaCheck(cudaMemcpy(S.elems + (size_t)cur * n, x.elems, (size_t)n * sizeof(Scalar), cudaMemcpyDeviceToDevice),
                  "copy x to S");
        Scalar minus_one;
        if constexpr (std::is_same<Scalar, __half>::value) minus_one = __float2half(-1.0f);
        else minus_one = (Scalar)-1;

        axpy_kernel<<<cfg.gridSize, cfg.blockSize>>>(minus_one, x_old.elems, S.elems + (size_t)cur * n, n);
        cudaCheck(cudaGetLastError(), "S = x - x_old");

        // Y_cur = g - g_old
        cudaCheck(cudaMemcpy(Y.elems + (size_t)cur * n, g.elems, (size_t)n * sizeof(Scalar), cudaMemcpyDeviceToDevice),
                  "copy g to Y");
        axpy_kernel<<<cfg.gridSize, cfg.blockSize>>>(minus_one, g_old.elems, Y.elems + (size_t)cur * n, n);
        cudaCheck(cudaGetLastError(), "Y = g - g_old");

        Acc sy = dot(S.elems + (size_t)cur * n, Y.elems + (size_t)cur * n, n, &dw);

        if (sy > (Acc)1e-12) {
            rho[cur] = (Acc)1 / sy;
        } else {
            // Skip update if curvature condition fails
            k--;
        }
    }

    cudaCheck(cudaMemcpy(x0, x.elems, (size_t)n * sizeof(Scalar), cudaMemcpyDeviceToHost), "x->x0 host");
    delete[] rho;
    delete[] alpha_hist;
    return val;
}

// =========================
//  Main
// =========================
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    // ===== Select precision here =====
    // using Scalar = double;
    // using Scalar = float;
    using Scalar = __half;   // FP16 storage + FP32 accumulation (recommended)

    using Acc = typename NumericTraits<Scalar>::Accum;

    int N = 1 << 16;  // Problem size
    int M = 10;       // History size

    // Host x0 (Scalar)
    Scalar* x0 = (Scalar*)malloc((size_t)N * sizeof(Scalar));
    if (!x0) throw std::bad_alloc();

    // --- TEST 1: QUADRATIC ---
    for (int i = 0; i < N; i++) {
        if constexpr (std::is_same<Scalar, __half>::value) x0[i] = __float2half(5.0f);
        else x0[i] = (Scalar)5.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    QuadraticTest<Scalar> quad(N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Starting Quadratic...\n");
    Acc final_f = lbfgs<Scalar>(N, M, x0, 1000, quad, (Acc)1e-4);
    printf("Quadratic Final F: %e (Target: 0)\n", (double)final_f);
    std::cout << "Time elapsed (ctor only): " << duration.count() << " ms\n";

    // --- TEST 2: ROSENBROCK ---
    for (int i = 0; i < N; i++) {
        if constexpr (std::is_same<Scalar, __half>::value) x0[i] = __float2half(-1.2f);
        else x0[i] = (Scalar)-1.2;
    }

    start = std::chrono::high_resolution_clock::now();
    RosenbrockTest<Scalar> rosen(N);
    end = std::chrono::high_resolution_clock::now();

    printf("Starting Rosenbrock...\n");
    final_f = lbfgs<Scalar>(N, M, x0, 5000, rosen, (Acc)1e-4);
    duration = end - start;
    printf("Rosenbrock Final F: %e (Target: 0)\n", (double)final_f);
    std::cout << "Time elapsed (ctor only): " << duration.count() << " ms\n";

    // --- TEST 3: RASTRIGIN ---
    for (int i = 0; i < N; i++) {
        if constexpr (std::is_same<Scalar, __half>::value) x0[i] = __float2half(-1.2f);
        else x0[i] = (Scalar)-1.2;
    }

    start = std::chrono::high_resolution_clock::now();
    RastriginTest<Scalar> rastrigin(N);
    end = std::chrono::high_resolution_clock::now();

    printf("Starting Rastrigin...\n");
    final_f = lbfgs<Scalar>(N, M, x0, 5000, rastrigin, (Acc)1e-4);
    duration = end - start;
    printf("Rastrigin Final F: %e (Target: 0)\n", (double)final_f);
    std::cout << "Time elapsed (ctor only): " << duration.count() << " ms\n";

    free(x0);
    return 0;
}
