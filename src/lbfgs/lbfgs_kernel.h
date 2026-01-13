#include <cstddef>

#include "../utils/functions.h"

namespace lbfgs {
namespace basic {
// void update_x_kernel(int n, double* x, const double* d, double alpha);
// void compute_sy_kernel(int n, double* s, double* y, const double* x_new, const double* x_old,
//                        const double* g_new, const double* g_old);
// void scale_vector_kernel(int n, double* r, const double* q, double H0);
class lbfgs {
   public:
    void operator()(const std::size_t problem_size, const std::size_t memory_size, double* x0,
                    const std::size_t max_itr, Func& func, const double eps = 1e-9);
};
}  // namespace basic
}  // namespace lbfgs