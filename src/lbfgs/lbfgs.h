namespace lbfgs {
namespace ghjkm {
// void copy_kernel(int n, const double* x, double* y);
// void scal_kernel(int n, double alpha, double* x);
// void axpy_kernel(int n, double alpha, const double* x, double* y);
// void dot_kernel(int n, const double* x, const double* y, double* res);
// void nrm2_kernel(int n, const double* x, double* res);
// void update_x_kernel(int n, double* x, const double* d, double alpha);
// void compute_sy_kernel(int n, double* s, double* y, const double* x_new, const double* x_old,
//                        const double* g_new, const double* g_old);
// void scale_vector_kernel(int n, double* r, const double* q, double H0);
template <typename Func>
class lbfgs;
}  // namespace ghjkm
}  // namespace lbfgs