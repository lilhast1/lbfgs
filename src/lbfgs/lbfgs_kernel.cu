#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#include "../utils/functions.h"
#include "lbfgs_kernel.h"

namespace lbfgs {
namespace basic {
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

// bez linesearch
class lbfgs {
    double* d_x;
    double* d_x_old;
    double* d_g;
    double* d_g_old;
    double* d_d;
    double* d_q;
    double* d_r;
    double* d_S;
    double* d_Y;
    double* rho;
    double* a;

    cublasHandle_t
        handle;  // cublas za start vicemo sta hrnja kaze ako ne moze rucno cemo ove kernele

    void init(std::size_t problem_size, std::size_t memory_size) {
        cublasCreate(&handle);

        cudaMallocManaged(&d_x, problem_size * sizeof(double));
        cudaMallocManaged(&d_x_old, problem_size * sizeof(double));
        cudaMallocManaged(&d_g, problem_size * sizeof(double));
        cudaMallocManaged(&d_g_old, problem_size * sizeof(double));
        cudaMalloc(&d_d, problem_size * sizeof(double));
        cudaMalloc(&d_q, problem_size * sizeof(double));
        cudaMalloc(&d_r, problem_size * sizeof(double));
        cudaMalloc(&d_S, problem_size * memory_size * sizeof(double));
        cudaMalloc(&d_Y, problem_size * memory_size * sizeof(double));

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
        delete[] rho;
        delete[] a;
        cublasDestroy(handle);
    }

   public:
    void operator()(const std::size_t problem_size, const std::size_t memory_size, double* x0,
                    const std::size_t max_itr, Func& func, const double eps = 1e-9) {
        init(problem_size, memory_size);

        cudaMemcpy(d_x, x0, problem_size * sizeof(double), cudaMemcpyHostToDevice);
        double val = func(d_x, problem_size);
        func.df(d_x, d_g, problem_size);

        double neg_one = -1.0;
        cublasDcopy(handle, problem_size, d_g, 1, d_d, 1);
        cublasDscal(handle, problem_size, &neg_one, d_d, 1);
        double alpha = 1.0;
        cudaMemcpy(d_x_old, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_g_old, d_g, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

        update_x_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_x, d_d, alpha);

        val = func(d_x, problem_size);
        func.df(d_x, d_g, problem_size);
        compute_sy_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_S, d_Y, d_x, d_x_old,
                                                               d_g, d_g_old);

        for (int itr = 0; itr < max_itr; itr++) {
            double g_norm;
            cublasDnrm2(handle, problem_size, d_g, 1, &g_norm);

            if (g_norm / (std::fabs(val) + 1.0) <= eps) {
                break;
            }

            int bound = itr > memory_size ? memory_size : itr;
            int curr = (itr - 1) % memory_size;
            cudaMemcpy(d_q, d_g, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

            for (int i = bound - 1; i >= 0; --i) {
                int idx = (itr - 1 - i) % memory_size;
                double dot_sq, dot_sy;
                cublasDdot(handle, problem_size, d_S + idx * problem_size, 1, d_q, 1, &dot_sq);
                cublasDdot(handle, problem_size, d_S + idx * problem_size, 1,
                           d_Y + idx * problem_size, 1, &dot_sy);
                rho[idx] = 1.0 / dot_sy;
                a[idx] = rho[idx] * dot_sq;
                double neg_a = -a[idx];
                cublasDaxpy(handle, problem_size, &neg_a, d_Y + idx * problem_size, 1, d_q, 1);
            }

            double ys, yy;
            cublasDdot(handle, problem_size, d_Y + curr * problem_size, 1,
                       d_S + curr * problem_size, 1, &ys);
            cublasDdot(handle, problem_size, d_Y + curr * problem_size, 1,
                       d_Y + curr * problem_size, 1, &yy);
            scale_vector_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_r, d_q,
                                                                     ys / yy);

            // Forward Loop
            for (int i = 0; i < bound; ++i) {
                int idx = (itr - bound + i) % memory_size;
                double b;
                cublasDdot(handle, problem_size, d_Y + idx * problem_size, 1, d_r, 1, &b);
                double factor = a[idx] - (rho[idx] * b);
                cublasDaxpy(handle, problem_size, &factor, d_S + idx * problem_size, 1, d_r, 1);
            }
            double neg_one = -1.0;
            cublasDcopy(handle, problem_size, d_r, 1, d_d, 1);
            cublasDscal(handle, problem_size, &neg_one, d_d, 1);

            double alpha = 1.0;
            cudaMemcpy(d_x_old, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_g_old, d_g, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

            update_x_kernel<<<(problem_size + 255) / 256, 256>>>(problem_size, d_x, d_d, alpha);

            val = func(d_x, problem_size);
            func.df(d_x, d_g, problem_size);

            int history_pos = (itr % memory_size);
            compute_sy_kernel<<<(problem_size + 255) / 256, 256>>>(
                problem_size, d_S + history_pos * problem_size, d_Y + history_pos * problem_size,
                d_x, d_x_old, d_g, d_g_old);
        }
        cudaMemcpy(x0, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToHost);
        destroy();
    }
};
}  // namespace basic
}  // namespace lbfgs