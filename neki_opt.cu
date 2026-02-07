// ====== DROP-IN PERFORMANCE PATCH (single-block, copy/paste) ======
// Goals:
// 1) Eliminate massive D2H traffic in dot() (your current dot copies O(n/BS) floats to host every call)
// 2) Eliminate thousands of tiny custom kernels for axpy/setVectorScalar/copies by using cuBLAS
// 3) Remove Unified Memory (cudaMallocManaged) from the hot path (page migration + overhead)
// 4) Fix a critical bug: BLOCK_SIZE was 64 while kernels were often launched with 256 threads -> shared OOB
//
// Build note: link with cuBLAS: add -lcublas (or in VS: cublas.lib)
// ===================================================================

#include <cublas_v2.h>

// FIX: must match your actual launch sizes / shared usage
#undef  BLOCK_SIZE
#define BLOCK_SIZE 256

// Simple CUDA error macro (you already have CUDA_CHECK later; keep one)
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

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) do {                                \
  cublasStatus_t _s = (call);                                  \
  if (_s != CUBLAS_STATUS_SUCCESS) {                           \
    printf("cuBLAS error %s:%d: status=%d\n",                  \
           __FILE__, __LINE__, (int)_s);                       \
    std::abort();                                              \
  }                                                            \
} while(0)
#endif

// Replace UnifiedVector usage in lbfgs with pure device vectors
template <typename T>
struct GpuVector {
    int n;
    T* elems;
    GpuVector(int n) : n(n), elems(nullptr) { CUDA_CHECK(cudaMalloc(&elems, n * sizeof(T))); }
    ~GpuVector() { if (elems) cudaFree(elems); }
    GpuVector(const GpuVector&) = delete;
    GpuVector& operator=(const GpuVector&) = delete;
};

template <typename T>
struct GpuMatrix {
    int n, m;
    T* elems;
    GpuMatrix(int n, int m) : n(n), m(m), elems(nullptr) { CUDA_CHECK(cudaMalloc(&elems, (size_t)n * m * sizeof(T))); }
    ~GpuMatrix() { if (elems) cudaFree(elems); }
    GpuMatrix(const GpuMatrix&) = delete;
    GpuMatrix& operator=(const GpuMatrix&) = delete;
};

// cublas helpers for float/double
template <typename T> struct Blas;
template <> struct Blas<float> {
    static cublasStatus_t dot(cublasHandle_t h, int n, const float* a, int ia, const float* b, int ib, float* out) {
        return cublasSdot(h, n, a, ia, b, ib, out);
    }
    static cublasStatus_t axpy(cublasHandle_t h, int n, const float* alpha, const float* x, int ix, float* y, int iy) {
        return cublasSaxpy(h, n, alpha, x, ix, y, iy);
    }
    static cublasStatus_t copy(cublasHandle_t h, int n, const float* x, int ix, float* y, int iy) {
        return cublasScopy(h, n, x, ix, y, iy);
    }
    static cublasStatus_t scal(cublasHandle_t h, int n, const float* alpha, float* x, int ix) {
        return cublasSscal(h, n, alpha, x, ix);
    }
};
template <> struct Blas<double> {
    static cublasStatus_t dot(cublasHandle_t h, int n, const double* a, int ia, const double* b, int ib, double* out) {
        return cublasDdot(h, n, a, ia, b, ib, out);
    }
    static cublasStatus_t axpy(cublasHandle_t h, int n, const double* alpha, const double* x, int ix, double* y, int iy) {
        return cublasDaxpy(h, n, alpha, x, ix, y, iy);
    }
    static cublasStatus_t copy(cublasHandle_t h, int n, const double* x, int ix, double* y, int iy) {
        return cublasDcopy(h, n, x, ix, y, iy);
    }
    static cublasStatus_t scal(cublasHandle_t h, int n, const double* alpha, double* x, int ix) {
        return cublasDscal(h, n, alpha, x, ix);
    }
};

// Small RAII wrapper for cuBLAS handle
struct CublasCtx {
    cublasHandle_t h{};
    CublasCtx() {
        CUBLAS_CHECK(cublasCreate(&h));
        // keep results on host for dot scalars (only 4/8 bytes per call)
        CUBLAS_CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
    }
    ~CublasCtx() { if (h) cublasDestroy(h); }
    CublasCtx(const CublasCtx&) = delete;
    CublasCtx& operator=(const CublasCtx&) = delete;
};

// ---------- Replace these wrappers (dot/axpy/setVectorScalar) ----------
template <typename T>
static inline double dot_blas(CublasCtx& blas, const T* a, const T* b, int n) {
    T out{};
    CUBLAS_CHECK(Blas<T>::dot(blas.h, n, a, 1, b, 1, &out));
    return (double)out;
}

// y = y + alpha * x (device)
template <typename T>
static inline void axpy_blas(CublasCtx& blas, T alpha, const T* x, T* y, int n) {
    CUBLAS_CHECK(Blas<T>::axpy(blas.h, n, &alpha, x, 1, y, 1));
}

// r = factor * q (device): r = q; r *= factor
template <typename T>
static inline void setVectorScalar_blas(CublasCtx& blas, T* r, const T* q, T factor, int n) {
    CUBLAS_CHECK(Blas<T>::copy(blas.h, n, q, 1, r, 1));
    CUBLAS_CHECK(Blas<T>::scal(blas.h, n, &factor, r, 1));
}

// ---------- OPTIONAL but important: reduce synchronizations ----------
// Replace most cudaDeviceSynchronize() in inner loops with stream semantics if you later add streams.
// For now: keep minimal sync only where host uses results (e.g., f_new comparisons).

// ===================== PATCHED L-BFGS CORE =====================
// Paste this over your existing lbfgs() implementation.
// Requirements:
// - Your func.f / func.df still take device pointers (as in your tests)
// - Remove DotLookupTable / KernelConfig usage for dot/axpy/setVectorScalar; keep your cfg only for objective kernels if needed.
template <typename Func, typename T>
double lbfgs(int n, int m, T* x0, int max_itr, Func func, const double eps = 1e-9) {
    // Use pure device allocations (no Unified Memory)
    GpuVector<T> x(n), g(n);
    GpuVector<T> x_old(n), g_old(n), d(n), q(n), r(n);
    GpuMatrix<T> S(n, m), Y(n, m);

    // cuBLAS handle (huge win: removes thousands of micro-kernel launches and D2H partial copies)
    CublasCtx blas;

    // host-side history scalars (tiny)
    T* rho        = new T[m];
    T* alpha_hist = new T[m];
    for (int i = 0; i < m; ++i) { rho[i] = (T)0; alpha_hist[i] = (T)0; }

    CUDA_CHECK(cudaMemcpy(x.elems, x0, (size_t)n * sizeof(T), cudaMemcpyHostToDevice));

    // Initial f and g
    double val = func.f(x.elems, n);
    func.df(x.elems, g.elems, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int mem = 0;
    bool restart = false;

    for (int k = 0; k < max_itr; k++) {
        // gradient norm (dot returns only a scalar -> tiny D2H, OK)
        double gg = dot_blas<T>(blas, g.elems, g.elems, n);
        double g_norm = std::sqrt(gg);
        if (g_norm < eps) break;

        // 1) Search direction d
        if (restart || mem == 0) {
            restart = false;
            // d = -g
            setVectorScalar_blas<T>(blas, d.elems, g.elems, (T)-1, n);
        } else {
            // q = g
            CUBLAS_CHECK(Blas<T>::copy(blas.h, n, g.elems, 1, q.elems, 1));

            int bound = (mem < m) ? mem : m;

            // First loop (backwards)
            for (int i = bound - 1; i >= 0; --i) {
                int idx = (k - 1 - i) % m;
                if (idx < 0) idx += m;
                if (rho[idx] == (T)0) { alpha_hist[idx] = (T)0; continue; }

                double s_dot_q = dot_blas<T>(blas, S.elems + (size_t)idx * n, q.elems, n);
                alpha_hist[idx] = (T)((double)rho[idx] * s_dot_q);

                // q = q - alpha*Y
                T neg_alpha = (T)(-alpha_hist[idx]);
                axpy_blas<T>(blas, neg_alpha, Y.elems + (size_t)idx * n, q.elems, n);
            }

            // gamma scaling using last pair
            int last_idx = (k - 1) % m;
            if (last_idx < 0) last_idx += m;

            double gamma = 1.0;
            if (rho[last_idx] != (T)0) {
                double s_dot_y = dot_blas<T>(blas, S.elems + (size_t)last_idx * n, Y.elems + (size_t)last_idx * n, n);
                double y_dot_y = dot_blas<T>(blas, Y.elems + (size_t)last_idx * n, Y.elems + (size_t)last_idx * n, n);
                gamma = (y_dot_y > 1e-18) ? (s_dot_y / y_dot_y) : 1.0;
            }

            // r = gamma * q
            setVectorScalar_blas<T>(blas, r.elems, q.elems, (T)gamma, n);

            // Second loop (forwards)
            for (int i = 0; i < bound; ++i) {
                int idx = (k - bound + i) % m;
                if (idx < 0) idx += m;
                if (rho[idx] == (T)0) continue;

                double y_dot_r = dot_blas<T>(blas, Y.elems + (size_t)idx * n, r.elems, n);
                double beta = (double)rho[idx] * y_dot_r;

                // r = r + (alpha - beta) * S
                T coeff = (T)((double)alpha_hist[idx] - beta);
                axpy_blas<T>(blas, coeff, S.elems + (size_t)idx * n, r.elems, n);
            }

            // d = -r
            setVectorScalar_blas<T>(blas, d.elems, r.elems, (T)-1, n);
        }

        // 2) Backtracking line search
        double g_dot_d = dot_blas<T>(blas, g.elems, d.elems, n);

        // If not descent, restart
        if (g_dot_d >= 0.0) {
            restart = true;
            mem = 0;
            for (int i = 0; i < m; ++i) rho[i] = (T)0;
            setVectorScalar_blas<T>(blas, d.elems, g.elems, (T)-1, n);
            g_dot_d = dot_blas<T>(blas, g.elems, d.elems, n);
        }

        const double c1 = 1e-4;
        const double decay = 0.5;
        double step = 1.0;
        bool success = false;

        // Save old state
        CUBLAS_CHECK(Blas<T>::copy(blas.h, n, x.elems, 1, x_old.elems, 1));
        CUBLAS_CHECK(Blas<T>::copy(blas.h, n, g.elems, 1, g_old.elems, 1));

        double f_old = val;

        for (int ls_iter = 0; ls_iter < 25; ls_iter++) {
            // x = x_old
            CUBLAS_CHECK(Blas<T>::copy(blas.h, n, x_old.elems, 1, x.elems, 1));
            // x += step*d
            axpy_blas<T>(blas, (T)step, d.elems, x.elems, n);

            double f_new = func.f(x.elems, n);
            CUDA_CHECK(cudaDeviceSynchronize()); // host uses f_new immediately

            if (f_new <= val + c1 * step * g_dot_d) {
                val = f_new;
                success = true;
                break;
            }
            step *= decay;

            if (fabs(f_new - f_old) < 1e-12) { success = false; break; }
            f_old = f_new;
        }

        if (!success) {
            std::cout << "Line Search Failed -> restart + fallback step\n";
            restart = true;
            mem = 0;
            for (int i = 0; i < m; ++i) rho[i] = (T)0;

            // fallback: steepest descent
            setVectorScalar_blas<T>(blas, d.elems, g.elems, (T)-1, n);

            double step_fb = 0.1;
            bool ok = false;

            for (int t = 0; t < 40; ++t) {
                CUBLAS_CHECK(Blas<T>::copy(blas.h, n, x_old.elems, 1, x.elems, 1));
                axpy_blas<T>(blas, (T)step_fb, d.elems, x.elems, n);

                double f_try = func.f(x.elems, n);
                CUDA_CHECK(cudaDeviceSynchronize());

                if (f_try < val) { val = f_try; ok = true; break; }
                step_fb *= 2.0;
                if (step_fb > 10.0) break;
            }

            if (!ok) {
                std::cout << "Fallback also failed. Terminating.\n";
                break;
            }

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
        CUBLAS_CHECK(Blas<T>::copy(blas.h, n, x.elems, 1, S.elems + (size_t)cur * n, 1));
        T minus1 = (T)-1;
        CUBLAS_CHECK(Blas<T>::axpy(blas.h, n, &minus1, x_old.elems, 1, S.elems + (size_t)cur * n, 1));

        // Y_cur = g - g_old
        CUBLAS_CHECK(Blas<T>::copy(blas.h, n, g.elems, 1, Y.elems + (size_t)cur * n, 1));
        CUBLAS_CHECK(Blas<T>::axpy(blas.h, n, &minus1, g_old.elems, 1, Y.elems + (size_t)cur * n, 1));

        double sy = dot_blas<T>(blas, S.elems + (size_t)cur * n, Y.elems + (size_t)cur * n, n);

        if (sy > 1e-12) {
            rho[cur] = (T)(1.0 / sy);
            if (mem < m) mem++;
        } else {
            rho[cur] = (T)0;
        }
    }

    CUDA_CHECK(cudaMemcpy(x0, x.elems, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost));
    delete[] rho;
    delete[] alpha_hist;
    return val;
}

// ====== END PATCH ======
