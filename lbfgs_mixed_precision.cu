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

template <typename T>
struct GpuMatrix {
    int n, m;
    T* elems;
    GpuMatrix(int n, int m) : n(n), m(m) { cudaMalloc(&elems, n * m * sizeof(T)); }
    ~GpuMatrix() { cudaFree(elems); }
};

template <typename T>
struct GpuVector {
    int n;
    T* elems;
    GpuVector(int n) : n(n) { cudaMalloc(&elems, n * sizeof(T)); }
    ~GpuVector() { cudaFree(elems); }
};

template <typename T>
struct UnifiedVector {
    int n;
    T* elems;
    UnifiedVector(int n) : n(n) { cudaMallocManaged(&elems, n * sizeof(T)); }
    ~UnifiedVector() { cudaFree(elems); }
};

template <typename T>
struct DotLookupTable {
    T* d_partial;
    T* h_partial;
    int max_grid_size;

    DotLookupTable(int n) {
        max_grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&d_partial, max_grid_size * sizeof(T));
        h_partial = (T*)malloc(max_grid_size * sizeof(T));
    }
    ~DotLookupTable() {
        cudaFree(d_partial);
        free(h_partial);
    }

    void clean() {
        cudaMemset(d_partial, 0, max_grid_size * sizeof(T));
        memset(h_partial, 0, max_grid_size * sizeof(T));
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

template <typename T>
__global__ void dotProduct(const T* a, const T* b, T* c, int n);

template <typename T>
__global__ void setVectorScalar(T* r, const T* q, T factor, int n);

template <typename T>
__global__ void axpy(T alpha, const T* x, T* y, int n);

template <typename T>
__global__ void mulVecScal(T alpha, const T* x, T* y, int n);

template <typename T>
__global__ void sum_reduction_kernel(const T* input, T* output, int n);

template <typename T>
__global__ void dot_partial_f32_to_f64(const float* a, const float* b, double* partial, int n);


/*----------------------------------------Testovi---------------------------------------------------*/

template <typename T>
__global__ void quad_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T xi = x[i];
        if (f_vals) f_vals[i] = xi * xi;
        if (g)      g[i] = (T)2 * xi;
    }
}

template <typename T>
struct QuadraticTest {
    T* d_temp_f;

    QuadraticTest(int n) : d_temp_f(nullptr) {
        cudaMalloc(&d_temp_f, n * sizeof(T));
    }
    ~QuadraticTest() {
        if (d_temp_f) cudaFree(d_temp_f);
    }

    double f(T* d_x, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quad_kernel<T><<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr, n);
        return gpu_sum(d_temp_f, n);
    }

    void df(T* d_x, T* d_g, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quad_kernel<T><<<blocks, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
    }
};

template <typename T>
__global__ void rosen_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        T x_curr = x[i];
        T x_next = x[i + 1];
        T t1 = (x_next - x_curr * x_curr);
        T t2 = (1.0 - x_curr);

        if (f_vals)
            f_vals[i] = 100.0 * t1 * t1 + t2 * t2;

        // Gradient components (requires atomicAdd because indices overlap)
        if (g) {
            atomicAdd(&g[i], -400.0 * x_curr * t1 - 2.0 * t2);
            atomicAdd(&g[i + 1], 200.0 * t1);
        }
    }
}

// No-atom version (for testing)
template <typename T>
__global__ void rosen_fg_kernel_noatom(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // f contribution uses i in [0..n-2]
    if (f_vals && i < n - 1) {
        T xi = x[i];
        T xj = x[i + 1];
        T t1 = xj - xi * xi;
        T t2 = (T)1 - xi;
        f_vals[i] = (T)100 * t1 * t1 + t2 * t2;
    }

    // gradient uses i in [0..n-1]
    if (g && i < n) {
        T gi = (T)0;

        if (i == 0) {
            T x0 = x[0], x1 = x[1];
            T t1 = x1 - x0 * x0;
            T t2 = (T)1 - x0;
            gi = (T)(-400) * x0 * t1 - (T)2 * t2;
        } else if (i == n - 1) {
            T xm1 = x[n - 1], xm2 = x[n - 2];
            gi = (T)200 * (xm1 - xm2 * xm2);
        } else {
            T xim1 = x[i - 1];
            T xi   = x[i];
            T xip1 = x[i + 1];
            T t_left  = xi - xim1 * xim1;
            T t_right = xip1 - xi * xi;
            T t2 = (T)1 - xi;
            gi = (T)200 * t_left + (T)(-400) * xi * t_right - (T)2 * t2;
        }

        g[i] = gi;
    }
}

template <typename T>
struct RosenbrockTest {
    T* d_temp_f;
    RosenbrockTest(int n) { cudaMalloc(&d_temp_f, n * sizeof(T)); }
    ~RosenbrockTest() { cudaFree(d_temp_f); }

    double f(T* d_x, int n) {
        cudaMemset(d_temp_f, 0, n * sizeof(T));
        rosen_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr, n);
        return gpu_sum(d_temp_f, n);
    }
    void df(T* d_x, T* d_g, int n) {
        cudaMemset(d_g, 0, n * sizeof(T));
        rosen_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
    }
};

template <typename T>
struct RosenbrockNoAtomTest {
    T* d_temp_f;
    RosenbrockNoAtomTest(int n) { cudaMalloc(&d_temp_f, n * sizeof(T)); }
    ~RosenbrockNoAtomTest() { cudaFree(d_temp_f); }

    double f(T* d_x, int n) {
        cudaMemset(d_temp_f, 0, n * sizeof(T));
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rosen_fg_kernel_noatom<<<blocks, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr, n);
        return gpu_sum(d_temp_f, n); // sum only [0..n-2] meaningful; fine if last is 0
    }

    void df(T* d_x, T* d_g, int n) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rosen_fg_kernel_noatom<<<blocks, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
    }

};

template <typename T>
__global__ void rastrigin_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T xi = x[i];
        if (f_vals)
            f_vals[i] = xi * xi - 10.0 * cos(2.0 * M_PI * xi);
        if (g)
            g[i] = 2.0 * xi + 20.0 * M_PI * sin(2.0 * M_PI * xi);
    }
}

template <typename T>
struct RastriginTest {
    T* d_temp_f;
    RastriginTest(int n) { cudaMalloc(&d_temp_f, n * sizeof(T)); }
    ~RastriginTest() { cudaFree(d_temp_f); }

    double f(T* d_x, int n) {
        rastrigin_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr,
                                                                            n);
        return 10.0 * n + gpu_sum(d_temp_f, n);
    }
    void df(T* d_x, T* d_g, int n) {
        rastrigin_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
    }
};

template <typename T>
__global__ void ackley_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T xi = x[i];
        if (f_vals)
            f_vals[i] = -20.0 * exp(-0.2 * sqrt(xi * xi)) - exp(cos(2.0 * M_PI * xi)) + 20.0 + M_E;
        if (g)
            g[i] = 4.0 * xi * exp(-0.2 * sqrt(xi * xi)) / sqrt(xi * xi) + 2.0 * M_PI * sin(2.0 * M_PI * xi) * exp(cos(2.0 * M_PI * xi));
    }
}

template <typename T>
struct AckleyTest {
    T* d_temp_f;
    AckleyTest(int n) { cudaMalloc(&d_temp_f, n * sizeof(T)); }
    ~AckleyTest() { cudaFree(d_temp_f); }

    double f(T* d_x, int n) {
        ackley_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_temp_f, (T*)nullptr,
                                                                                n);
        return gpu_sum(d_temp_f, n);
    }
    void df(T* d_x, T* d_g, int n) {
        ackley_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, (T*)nullptr, d_g, n);
    }
};

/*----------------------------------------Wrapperfje---------------------------------------------------*/

// neka su vektori a i b na gpu alocirani vraca a . b
template <typename T>
double dot(const T* a, const T* b, int n, DotLookupTable<T>* context = nullptr);

template <typename Func, typename T>
double lbfgs(int n, int m, T* x0, int max_itr, Func func, const double eps = 1e-9);

int main(int argc, char* argv[]) {

    //using T = double;
    using T = float;
    
    int N = 1 << 12;  // Problem size
    int M = 10;       // History size
    T* x0 = new T[N];

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    float elapsedTime;

    // --- TEST 1: QUADRATIC ---
    for (int i = 0; i < N; i++) x0[i] = 10.0;  // Start far away
    QuadraticTest<T> quad(N);
    
    printf("Starting Quadratic...\n");
    
    cudaEventRecord(startEvent);
    double final_f = lbfgs(N, M, x0, 1000, quad, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Quadratic Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    // --- TEST 2: ROSENBROCK ---
    for (int i = 0; i < N; i++) x0[i] = -4.2;  // Standard starting point
    RosenbrockTest<T> rosen(N);

    printf("Starting Rosenbrock...\n");

    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, rosen, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Rosenbrock Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

/*    
    // TEST 3: ROSENBROCK NO ATOM
    for (int i = 0; i < N; i++) x0[i] = -4.2;  // Standard starting point
    RosenbrockNoAtomTest<T> rosen_noatom(N);
    
    printf("Starting Rosenbrock No Atom...\n");

    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, rosen_noatom, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Rosenbrock No Atom Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";
*/



    // --- TEST 4: Rastrigin ---
    for (int i = 0; i < N; i++) x0[i] = -1.2;  // Standard starting point
    RastriginTest<T> rastrigin(N);

    printf("Starting Rastrigin...\n");

    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, rastrigin, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Rastrigin Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    // --- TEST 5: Ackley ---
    for (int i = 0; i < N; i++) x0[i] = -4;
    AckleyTest<T> ackley(N);
    printf("Starting Ackley...\n");

    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, ackley, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Ackley Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";



    delete[] x0;

    return 0;
}

/*----------------------------------------IMPL---------------------------------------------------*/

/*----------------------------------------Datatypes---------------------------------------------------*/

/*----------------------------------------KERNELI---------------------------------------------------*/

template <typename T>
__global__ void mulVecScal(T alpha, const T* x, T* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += alpha * x[i];
}

// y = y + alpha * x
template <typename T>
__global__ void axpy(T alpha, const T* x, T* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] += alpha * x[i];
    }
}

// r = factor * q
template <typename T>
__global__ void setVectorScalar(T* r, const T* q, T factor, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        r[i] = q[i] * factor;
    }
}

template <typename T>
__global__ void dotProduct(const T* a, const T* b, T* c, int n) {
    // Get global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T buf[BLOCK_SIZE];

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

__global__ void dot_atomic_f32(const float* a, const float* b, float* out, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += a[i] * b[i];

    // block reduce in shared
    __shared__ float buf[256];
    buf[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, buf[0]);
}

template <typename T>
__global__ void sum_reduction_kernel(const T* input, T* output, int n) {
    __shared__ T sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    T val = (i < n) ? input[i] : 0.0;
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

__global__ void dot_partial_f32_to_f64(const float* a, const float* b, double* partial, int n) {
    __shared__ double buf[256];
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += (double)a[i] * (double)b[i];

    buf[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = buf[0];
}

/*----------------------------------------Wrapperfje---------------------------------------------------*/

// Helper to get sum of a device array
    
template <typename T>
double gpu_sum(T* d_elements, int n) {
    static T* d_res = nullptr;
    if (!d_res)
        cudaMalloc(&d_res, sizeof(T));

    cudaMemset(d_res, 0, sizeof(T));
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_reduction_kernel<<<blocks, BLOCK_SIZE>>>(d_elements, d_res, n);

    T h_res{};
    cudaMemcpy(&h_res, d_res, sizeof(T), cudaMemcpyDeviceToHost);
    return (double)h_res;
}

template <typename T>
double dot(const T* a, const T* b, int n, DotLookupTable<T>* context) {
    dim3 blockSize = BLOCK_SIZE;
    dim3 gridSize = (n + blockSize.x - 1) / blockSize.x;

    auto datasize = sizeof(T) * n;
    auto partial_results_size = sizeof(T) * gridSize.x;

    T *d_c, *h_c;  // parcijalne sume

    if (!context) {
        h_c = (T*)malloc(partial_results_size);
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


inline float dot_f32(const float* a, const float* b, int n, KernelConfig cfg) {
    static float* d_out = nullptr;
    if (!d_out) cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));
    dot_atomic_f32<<<cfg.gridSize, 256>>>(a, b, d_out, n);
    float h;
    cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

double dot_f32_accum_f64(const float* a, const float* b, int n, const KernelConfig& cfg) {
    static double* d_partial = nullptr;
    static double* h_partial = nullptr;
    static int cap = 0;

    int blocks = cfg.gridSize;
    if (blocks <= 0) blocks = 1;

    if (blocks > cap) {
        if (d_partial) cudaFree(d_partial);
        if (h_partial) free(h_partial);
        cudaMalloc(&d_partial, blocks * sizeof(double));
        h_partial = (double*)malloc(blocks * sizeof(double));
        cap = blocks;
    }

    dot_partial_f32_to_f64<<<blocks, 256>>>(a, b, d_partial, n);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double s = 0.0;
    for (int i = 0; i < blocks; ++i) s += h_partial[i];
    return s;
}


#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                  \
  cudaError_t _e = (call);                                     \
  if (_e != cudaSuccess) {                                     \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
           cudaGetErrorString(_e));                            \
    std::abort();                                              \
  }                                                            \
} while(0)
#endif

template <typename Func, typename T>
double lbfgs(int n, int m, T* x0, int max_itr, Func func, const double eps) {
    UnifiedVector<T> x(n), g(n);
    GpuVector<T> x_old(n), g_old(n), d(n), q(n), r(n);
    GpuMatrix<T> S(n, m), Y(n, m);

    // NOTE: host_x_trial is no longer needed because we evaluate f(x) on DEVICE
    // (and your Rosenbrock/Rastrigin tests expect device pointers).
    // Keep it removed to avoid accidental host-pointer calls.
    // T* host_x_trial = (T*)malloc(n * sizeof(T));

    T* rho        = new T[m];
    T* alpha_hist = new T[m];
    for (int i = 0; i < m; ++i) { rho[i] = (T)0; alpha_hist[i] = (T)0; }

    DotLookupTable<T> dw(n);
    KernelConfig cfg(n);



    CUDA_CHECK(cudaMemcpy(x.elems, x0, n * sizeof(T), cudaMemcpyHostToDevice));

    double val = func.f(x.elems, n);
    func.df(x.elems, g.elems, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int mem = 0;            // how many (s,y) pairs are valid: 0..m
    bool restart = false;   // forces steepest descent direction next iter

    for (int k = 0; k < max_itr; k++) {
        // gradient norm
        double g_norm = std::sqrt(dot_f32(g.elems, g.elems, n, cfg));
        if (g_norm < eps) break;

        // 1) Compute search direction d
        if (restart || mem == 0) {
            restart = false;
            setVectorScalar<T><<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, (T)-1, n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // q = g
            CUDA_CHECK(cudaMemcpy(q.elems, g.elems, n * sizeof(T), cudaMemcpyDeviceToDevice));

            int bound = (mem < m) ? mem : m;

            // First loop (backwards)
            for (int i = bound - 1; i >= 0; --i) {
                int idx = (k - 1 - i) % m;
                if (idx < 0) idx += m;

                // if rho[idx]==0, pair is inactive; skip
                if (rho[idx] == (T)0) { alpha_hist[idx] = (T)0; continue; }

                double s_dot_q = dot_f32(S.elems + idx * n, q.elems, n, cfg);
                alpha_hist[idx] = (T)(rho[idx] * (T)s_dot_q);

                axpy<T><<<cfg.gridSize, cfg.blockSize>>>((T)(-alpha_hist[idx]),
                                                         Y.elems + idx * n,
                                                         q.elems, n);
                CUDA_CHECK(cudaGetLastError());
            }

            // Scaling gamma using the most recent pair (k-1)
            int last_idx = (k - 1) % m;
            if (last_idx < 0) last_idx += m;

            double gamma = 1.0;
            if (rho[last_idx] != (T)0) {
                double s_dot_y = dot_f32(S.elems + last_idx * n, Y.elems + last_idx * n, n, cfg);
                double y_dot_y = dot_f32(Y.elems + last_idx * n, Y.elems + last_idx * n, n, cfg);
                gamma = (y_dot_y > 1e-18) ? (s_dot_y / y_dot_y) : 1.0;
            }

            setVectorScalar<T><<<cfg.gridSize, cfg.blockSize>>>(r.elems, q.elems, (T)gamma, n);
            CUDA_CHECK(cudaGetLastError());

            // Second loop (forwards)
            for (int i = 0; i < bound; ++i) {
                int idx = (k - bound + i) % m;
                if (idx < 0) idx += m;

                if (rho[idx] == (T)0) continue;

                double y_dot_r = dot_f32(Y.elems + idx * n, r.elems, n, cfg);
                double beta = (double)rho[idx] * y_dot_r;

                axpy<T><<<cfg.gridSize, cfg.blockSize>>>((T)(alpha_hist[idx] - (T)beta),
                                                         S.elems + idx * n,
                                                         r.elems, n);
                CUDA_CHECK(cudaGetLastError());
            }

            setVectorScalar<T><<<cfg.gridSize, cfg.blockSize>>>(d.elems, r.elems, (T)-1, n);
            CUDA_CHECK(cudaGetLastError());
        }

        // 2) Backtracking line search (GPU f(x))
        double g_dot_d = dot_f32(g.elems, d.elems, n, cfg);

        // If not a descent direction, restart to steepest descent
        if (g_dot_d >= 0.0) {
            restart = true;
            mem = 0;
            for (int i = 0; i < m; ++i) rho[i] = (T)0;
            setVectorScalar<T><<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, (T)-1, n);
            CUDA_CHECK(cudaGetLastError());
            g_dot_d = dot_f32(g.elems, d.elems, n, cfg);
        }

        const double c1 = 1e-4;
        const double decay = 0.5;
        double step = 1.0;
        bool success = false;

        // Save old state
        CUDA_CHECK(cudaMemcpy(x_old.elems, x.elems, n * sizeof(T), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(g_old.elems, g.elems, n * sizeof(T), cudaMemcpyDeviceToDevice));

        double f_old = val;

        for (int ls_iter = 0; ls_iter < 25; ls_iter++) {
            // x = x_old + step * d
            CUDA_CHECK(cudaMemcpy(x.elems, x_old.elems, n * sizeof(T), cudaMemcpyDeviceToDevice));
            axpy<T><<<cfg.gridSize, cfg.blockSize>>>((T)step, d.elems, x.elems, n);
            CUDA_CHECK(cudaGetLastError());

            double f_new = func.f(x.elems, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (f_new <= val + c1 * step * g_dot_d) {
                val = f_new;
                success = true;
                break;
            }
            step *= decay;

            if (fabs(f_new - f_old) < 1e-12) {
                // Function value not changing much; likely stuck. Break to fallback.
                //
                // Ako se stavi true, nastavlja u broju iteracija presitnim koracima
                // sve do globalnog optimuma (ako je dostupan)
                //
                // Ako se stavi false, ide na fallback odmah i time se izbjegava gubljenje vremena,
                // povećavajući korak da se izađe iz ravnih područja.
                success = false;
                break;
            }
            else f_old = f_new;


        }

        if (!success) {
            // Hard restart + fallback with larger steps to escape flat regions
            std::cout << "Line Search Failed -> restart + fallback step\n";
            restart = true;
            mem = 0;
            for (int i = 0; i < m; ++i) rho[i] = (T)0;

            // fallback: try larger steepest descent steps until decrease
            setVectorScalar<T><<<cfg.gridSize, cfg.blockSize>>>(d.elems, g.elems, (T)-1, n);
            CUDA_CHECK(cudaGetLastError());

            double step_fb = 0.1;
            bool ok = false;

            for (int t = 0; t < 40; ++t) {
            CUDA_CHECK(cudaMemcpy(x.elems, x_old.elems, n * sizeof(T), cudaMemcpyDeviceToDevice));
            axpy<T><<<cfg.gridSize, cfg.blockSize>>>((T)step_fb, d.elems, x.elems, n);
            CUDA_CHECK(cudaGetLastError());

            double f_try = func.f(x.elems, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (f_try < val) {
                val = f_try;
                ok = true;
                break;
            }
            step_fb *= 2.0;
            if (step_fb > 10.0) break;
            }

            if (!ok) {
                std::cout << "Fallback also failed. Terminating.\n";
                break;
            }

            // update gradient after accepted fallback step
            func.df(x.elems, g.elems, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            continue;
        }

        // 3) Update gradient
        func.df(x.elems, g.elems, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4) Update history S/Y
        int cur = k % m;

        // S_cur = x - x_old
        CUDA_CHECK(cudaMemcpy(S.elems + cur * n, x.elems, n * sizeof(T), cudaMemcpyDeviceToDevice));
        axpy<T><<<cfg.gridSize, cfg.blockSize>>>((T)-1, x_old.elems, S.elems + cur * n, n);
        CUDA_CHECK(cudaGetLastError());

        // Y_cur = g - g_old
        CUDA_CHECK(cudaMemcpy(Y.elems + cur * n, g.elems, n * sizeof(T), cudaMemcpyDeviceToDevice));
        axpy<T><<<cfg.gridSize, cfg.blockSize>>>((T)-1, g_old.elems, Y.elems + cur * n, n);
        CUDA_CHECK(cudaGetLastError());

        double sy = dot_f32(S.elems + cur * n, Y.elems + cur * n, n, cfg);

        if (sy > 1e-12) {
            rho[cur] = (T)(1.0 / sy);
            if (mem < m) mem++;
        } else {
            // Mark this slot inactive; do NOT k-- (that causes weird behavior)
            rho[cur] = (T)0;
            // keep mem as-is (still valid older pairs)
        }
    }

    CUDA_CHECK(cudaMemcpy(x0, x.elems, n * sizeof(T), cudaMemcpyDeviceToHost));
    delete[] rho;
    delete[] alpha_hist;
    return val;
}
