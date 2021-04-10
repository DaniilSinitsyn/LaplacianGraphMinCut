#include "cusp_cg.h"
#include <cusp/csr_matrix.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>
#include <cusp/precond/diagonal.h>
#include <tbb/parallel_for.h>

Eigen::VectorXf cusp_cg_solve(const SpMat &A_, const Eigen::VectorXf &b_) {
  cusp::csr_matrix<int, float, cusp::host_memory> A(A_.rows(), A_.cols(),
                                                    A_.nonZeros());
  cusp::array1d<float, cusp::host_memory> b(b_.size());

  tbb::parallel_for(tbb::blocked_range<int>(0, A_.rows() + 1), [&](auto r) {
    for (auto i = r.begin(); i < r.end(); ++i) {
      A.row_offsets[i] = A_.outerIndexPtr()[i];
    }
  });

  tbb::parallel_for(tbb::blocked_range<int>(0, A_.nonZeros()), [&](auto r) {
    for (auto i = r.begin(); i < r.end(); ++i) {
      A.column_indices[i] = A_.innerIndexPtr()[i];
      A.values[i] = A_.valuePtr()[i];
    }
  });

  tbb::parallel_for(tbb::blocked_range<int>(0, A_.rows()), [&](auto r) {
    for (auto i = r.begin(); i < r.end(); ++i) {
      b[i] = b_[i];
    }
  });

  cusp::csr_matrix<int, float, cusp::device_memory> A_gpu(A);

  cusp::array1d<float, cusp::device_memory> x_gpu(A.num_rows, 0);
  cusp::array1d<float, cusp::device_memory> b_gpu(b);

  cusp::monitor<float> monitor(b_gpu, 2000, 1e-6, 1e-2, true);

  cusp::precond::diagonal<float, cusp::device_memory> M(A_gpu);

  cusp::krylov::cg(A_gpu, x_gpu, b_gpu, monitor, M);

  cusp::array1d<float, cusp::host_memory> x(x_gpu);
  cusp::array1d<float, cusp::host_memory>::view x_view(x);
  Eigen::VectorXf ans(A_.rows());

  tbb::parallel_for(tbb::blocked_range<int>(0, A_.rows()), [&](auto r) {
    for (auto i = r.begin(); i < r.end(); ++i) {
      ans[i] = x_view[i];
    }
  });

  return ans;
}

