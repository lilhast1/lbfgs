#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>

#ifndef M_PI 
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

__global__ void update_x_kernel(int n, float* x, const float* d, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += alpha * d[i];
}

__global__ void compute_sy_kernel(int n, float* s, float* y, const float* x_new,
                                  const float* x_old, const float* g_new, const float* g_old) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s[i] = x_new[i] - x_old[i];
        y[i] = g_new[i] - g_old[i];
    }
}

__global__ void scale_vector_kernel(int n, float* r, const float* q, float H0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        r[i] = H0 * q[i];
}

template <typename Func>
class lbfgs {
    float* d_x;
    float* d_x_old;
    float* d_g;
    float* d_g_old;
    float* d_d;
    float* d_q;
    float* d_r;
    float* d_S;
    float* d_Y;
    float* rho;
    float* a;

    cublasHandle_t
        handle;  // cublas za start vicemo sta hrnja kaze ako ne moze rucno cemo ove kernele

    void init(std::size_t problem_size, std::size_t memory_size) {
        cublasCreate(&handle);

        cudaMalloc(&d_x, problem_size * sizeof(float));
        cudaMalloc(&d_x_old, problem_size * sizeof(float));
        cudaMalloc(&d_g, problem_size * sizeof(float));
        cudaMalloc(&d_g_old, problem_size * sizeof(float));
        cudaMalloc(&d_d, problem_size * sizeof(float));
        cudaMalloc(&d_q, problem_size * sizeof(float));
        cudaMalloc(&d_r, problem_size * sizeof(float));
        cudaMalloc(&d_S, problem_size * memory_size * sizeof(float));
        cudaMalloc(&d_Y, problem_size * memory_size * sizeof(float));

        rho = new float[memory_size];
        a = new float[memory_size];
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
        delete[] rho;
        delete[] a;
        cublasDestroy(handle);
    }

   public:
    void operator()(const std::size_t problem_size, const std::size_t memory_size, float* x0,
                    const std::size_t max_itr, Func func, const float eps = 1e-9) {
        init(problem_size, memory_size);

        cudaMemcpy(d_x, x0, problem_size * sizeof(float), cudaMemcpyHostToDevice);
        float val = func.f(d_x, problem_size);
        func.df(d_x, d_g, problem_size);

        float neg_one = -1.0f;
        cublasScopy(handle, problem_size, d_g, 1, d_d, 1);
        cublasSscal(handle, problem_size, &neg_one, d_d, 1);
        float alpha = 1.0f;
        cudaMemcpy(d_x_old, d_x, problem_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_g_old, d_g, problem_size * sizeof(float), cudaMemcpyDeviceToDevice);

        update_x_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_x, d_d, alpha);

        val = func.f(d_x, problem_size);
        func.df(d_x, d_g, problem_size);
        compute_sy_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_S, d_Y, d_x, d_x_old,
                                                               d_g, d_g_old);

        for (int itr = 0; itr < max_itr; itr++) {
            float g_norm;
            cublasSnrm2(handle, problem_size, d_g, 1, &g_norm);

            if (g_norm / (std::fabs(val) + 1.0f) <= eps) {
                break;
            }

            int bound = itr > memory_size ? memory_size : itr;
            int curr = (itr - 1) % memory_size;
            cudaMemcpy(d_q, d_g, problem_size * sizeof(float), cudaMemcpyDeviceToDevice);

            for (int i = bound - 1; i >= 0; --i) {
                int idx = (itr - 1 - i) % memory_size;
                float dot_sq, dot_sy;
                cublasSdot(handle, problem_size, d_S + idx * problem_size, 1, d_q, 1, &dot_sq);
                cublasSdot(handle, problem_size, d_S + idx * problem_size, 1,
                           d_Y + idx * problem_size, 1, &dot_sy);
                rho[idx] = 1.0f / dot_sy;
                a[idx] = rho[idx] * dot_sq;
                float neg_a = -a[idx];
                cublasSaxpy(handle, problem_size, &neg_a, d_Y + idx * problem_size, 1, d_q, 1);
            }

            float ys, yy;
            cublasSdot(handle, problem_size, d_Y + curr * problem_size, 1,
                       d_S + curr * problem_size, 1, &ys);
            cublasSdot(handle, problem_size, d_Y + curr * problem_size, 1,
                       d_Y + curr * problem_size, 1, &yy);
            scale_vector_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_r, d_q,
                                                                     ys / yy);

            // Forward Loop
            for (int i = 0; i < bound; ++i) {
                int idx = (itr - bound + i) % memory_size;
                float b;
                cublasSdot(handle, problem_size, d_Y + idx * problem_size, 1, d_r, 1, &b);
                float factor = a[idx] - (rho[idx] * b);
                cublasSaxpy(handle, problem_size, &factor, d_S + idx * problem_size, 1, d_r, 1);
            }
            float neg_one = -1.0f;
            cublasScopy(handle, problem_size, d_r, 1, d_d, 1);
            cublasSscal(handle, problem_size, &neg_one, d_d, 1);

            float alpha = 1.0;
            cudaMemcpy(d_x_old, d_x, problem_size * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_g_old, d_g, problem_size * sizeof(float), cudaMemcpyDeviceToDevice);

            update_x_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_x, d_d, alpha);

            val = func.f(d_x, problem_size);
            func.df(d_x, d_g, problem_size);

            int history_pos = (itr % memory_size);
            compute_sy_kernel<<<(problem_size + 255) / 256, 256>>>(
                problem_size, d_S + history_pos * problem_size, d_Y + history_pos * problem_size,
                d_x, d_x_old, d_g, d_g_old);
        }
        cudaMemcpy(x0, d_x, problem_size * sizeof(float), cudaMemcpyDeviceToHost);
        destroy();
    }
};

template <typename T>
__global__ void quad_kernel(const T* x, T* f_vals, T* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T xi = x[i];
        if (f_vals) f_vals[i] = xi * xi;
        if (g)      g[i] = (T)2 * xi;
    }
}

// Test: kvadratna funkcija - wrapper oko kernela iznad
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

// Kernel za računanje Rosenbrockove funkcije i njenog gradijenta
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

// Wrapper za Rosenbrock test
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

// Wrapper za not-atomic Rosenbrock test
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

// Kernel za računanje Rastriginove funkcije i njenog gradijenta
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

// Wrapper za Rastrigin test
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

// Kernel za računanje Ackleyjeve funkcije i njenog gradijenta
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

// Wrapper za Ackley test
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
    using T = float;
    
    int N = 1 << 12;  // Problem size
    int M = 10;       // History size
    T* x0 = new T[N];

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    float elapsedTime;

    // --- TEST 1: QUADRATIC ---
    for (int i = 0; i < N; i++) x0[i] = 8.0;  // Start far away
    QuadraticTest<T> quad(N);
    lbfgs<QuadraticTest<T>> opt;

    printf("Starting: Quadratic...\n");
    
    cudaEventRecord(startEvent);
    opt(N, M, x0, 1000, quad, 1e-6);
    double final_f = quad.f(x0, N);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Quadratic Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    // --- TEST 2: ROSENBROCK [-5, 10]^N ---
    for (int i = 0; i < N; i++) x0[i] = -4.2;  // Standard starting point
    RosenbrockTest<T> rosen(N);

    printf("Starting: Rosenbrock...\n");

    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, rosen, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Rosenbrock Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

/*  Sporo, jako sporo.

    // --- TEST 3: ROSENBROCK NO ATOM ---
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



    // --- TEST 4: Rastrigin [-5.12, 5.12]^N ---
    for (int i = 0; i < N; i++) x0[i] = -1.2;  // Standard starting point
    RastriginTest<T> rastrigin(N);

    printf("Starting: Rastrigin...\n");

    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, rastrigin, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Rastrigin Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";

    // --- TEST 5: Ackley [-32.768, 32.768]^N ---
    for (int i = 0; i < N; i++) x0[i] = -4;
    AckleyTest<T> ackley(N);
    printf("Starting: Ackley...\n");

    cudaEventRecord(startEvent);
    final_f = lbfgs(N, M, x0, 5000, ackley, 1e-6);
    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Ackley Final F: %e (Target: 0)\n", final_f);
    std::cout << "Time elapsed: " << elapsedTime << " ms\n";



    delete[] x0;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    return 0;
}
