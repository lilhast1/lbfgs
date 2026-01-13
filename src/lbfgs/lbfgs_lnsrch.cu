#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "lbfgs_lnsrch.h"

namespace lbfgs {
namespace lnsrch {
template <typename Func>
class lbfgs {
    double *d_x, *d_x_old, *d_g, *d_g_old, *d_d, *d_q, *d_r, *d_S, *d_Y;
    double* d_x_next;  // Temporary buffer for line search
    double *rho, *a;
    cublasHandle_t handle;

    void init(std::size_t problem_size, std::size_t memory_size) {
        cublasCreate(&handle);
        cudaMalloc(&d_x, problem_size * sizeof(double));
        cudaMalloc(&d_x_old, problem_size * sizeof(double));
        cudaMalloc(&d_g, problem_size * sizeof(double));
        cudaMalloc(&d_g_old, problem_size * sizeof(double));
        cudaMalloc(&d_d, problem_size * sizeof(double));
        cudaMalloc(&d_q, problem_size * sizeof(double));
        cudaMalloc(&d_r, problem_size * sizeof(double));
        cudaMalloc(&d_x_next, problem_size * sizeof(double));
        cudaMalloc(&d_S, problem_size * memory_size * sizeof(double));
        cudaMalloc(&d_Y, problem_size * memory_size * sizeof(double));

        rho = new double[memory_size];
        a = new double[memory_size];
    }

    void destroy() {
        cublasDestroy(handle);
        cudaFree(d_x);
        cudaFree(d_x_old);
        cudaFree(d_g);
        cudaFree(d_g_old);
        cudaFree(d_d);
        cudaFree(d_q);
        cudaFree(d_r);
        cudaFree(d_x_next);
        cudaFree(d_S);
        cudaFree(d_Y);
        delete[] rho;
        delete[] a;
    }

   public:
    void operator()(const std::size_t problem_size, const std::size_t memory_size, double* x0,
                    const std::size_t max_itr, Func func, const double eps = 1e-9) {
        init(problem_size, memory_size);

        cudaMemcpy(d_x, x0, problem_size * sizeof(double), cudaMemcpyHostToDevice);

        // Initial evaluation
        double val = func.f(d_x, problem_size);
        func.df(d_x, d_g, problem_size);

        // Initial direction: d = -g
        cublasDcopy(handle, problem_size, d_g, 1, d_d, 1);
        double neg_one = -1.0;
        cublasDscal(handle, problem_size, &neg_one, d_d, 1);

        for (int itr = 0; itr < max_itr; itr++) {
            // Check convergence
            double g_norm;
            cublasDnrm2(handle, problem_size, d_g, 1, &g_norm);
            if (g_norm / (std::fabs(val) + 1.0) <= eps)
                break;

            // --- TWO-LOOP RECURSION TO FIND d_d ---
            if (itr > 0) {
                int bound = itr > (int)memory_size ? (int)memory_size : itr;
                int curr = (itr - 1) % (int)memory_size;
                cublasDcopy(handle, problem_size, d_g, 1, d_q, 1);

                for (int i = bound - 1; i >= 0; --i) {
                    int idx = (itr - 1 - i) % (int)memory_size;
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

                // Scale q into r: r = (ys/yy) * q
                double H0 = ys / yy;
                cublasDcopy(handle, problem_size, d_q, 1, d_r, 1);
                cublasDscal(handle, problem_size, &H0, d_r, 1);

                for (int i = 0; i < bound; ++i) {
                    int idx = (itr - bound + i) % (int)memory_size;
                    double b;
                    cublasDdot(handle, problem_size, d_Y + idx * problem_size, 1, d_r, 1, &b);
                    double factor = a[idx] - (rho[idx] * b);
                    cublasDaxpy(handle, problem_size, &factor, d_S + idx * problem_size, 1, d_r, 1);
                }
                // d = -r
                cublasDcopy(handle, problem_size, d_r, 1, d_d, 1);
                cublasDscal(handle, problem_size, &neg_one, d_d, 1);
            }

            // --- BACKTRACKING LINE SEARCH (ARMIJO) ---
            double alpha = 1.0;
            const double c1 = 1e-4;    // Sufficient decrease constant
            const double rho_b = 0.5;  // Backtracking multiplier

            double g_dot_d;
            cublasDdot(handle, problem_size, d_g, 1, d_d, 1, &g_dot_d);

            // If g_dot_d >= 0, d is not a descent direction (shouldn't happen in L-BFGS)
            if (g_dot_d >= 0) {
                // Reset to steepest descent if search direction fails
                cublasDcopy(handle, problem_size, d_g, 1, d_d, 1);
                cublasDscal(handle, problem_size, &neg_one, d_d, 1);
                cublasDdot(handle, problem_size, d_g, 1, d_d, 1, &g_dot_d);
            }

            double f_old = val;
            while (alpha > 1e-10) {
                // x_next = x + alpha * d
                cudaMemcpy(d_x_next, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);
                cublasDaxpy(handle, problem_size, &alpha, d_d, 1, d_x_next, 1);

                double f_next = func.f(d_x_next, problem_size);

                // Armijo Condition: f(x + a*d) <= f(x) + c1 * a * (g^T * d)
                if (f_next <= f_old + c1 * alpha * g_dot_d) {
                    val = f_next;
                    break;
                }
                alpha *= rho_b;
            }

            // --- UPDATE HISTORY ---
            cudaMemcpy(d_x_old, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_g_old, d_g, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

            // Commit the new x
            cudaMemcpy(d_x, d_x_next, problem_size * sizeof(double), cudaMemcpyDeviceToDevice);

            // Get new gradient
            func.df(d_x, d_g, problem_size);

            // Store s = x_new - x_old and y = g_new - g_old
            int history_pos = (itr % (int)memory_size);
            double* S_ptr = d_S + history_pos * problem_size;
            double* Y_ptr = d_Y + history_pos * problem_size;

            // S = x - x_old
            cublasDcopy(handle, problem_size, d_x, 1, S_ptr, 1);
            cublasDaxpy(handle, problem_size, &neg_one, d_x_old, 1, S_ptr, 1);

            // Y = g - g_old
            cublasDcopy(handle, problem_size, d_g, 1, Y_ptr, 1);
            cublasDaxpy(handle, problem_size, &neg_one, d_g_old, 1, Y_ptr, 1);
        }

        cudaMemcpy(x0, d_x, problem_size * sizeof(double), cudaMemcpyDeviceToHost);
        destroy();
    }
};
}  // namespace lnsrch
}  // namespace lbfgs