#ifndef GRAPHLALPLACIAN_HPP
#define GRAPHLALPLACIAN_HPP

#include <Eigen/Eigen>
#include <tbb/tbb.h>
#include <vector>

class GraphLaplacian;

namespace Eigen {
namespace internal {
template <>
struct traits<GraphLaplacian>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
} // namespace internal
} // namespace Eigen

struct GraphLaplacianRow {
  GraphLaplacianRow(const Eigen::VectorXd &coeffs,
                    const std::vector<long> &idxs)
      : coeffs(coeffs), idxs(idxs) {}
  Eigen::VectorXd coeffs;
  std::vector<long> idxs;
};

class GraphLaplacian : public Eigen::EigenBase<GraphLaplacian> {
public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Index rows() const { return n; }
  Index cols() const { return n; }
  struct DummyIterator {
    DummyIterator() {}
    DummyIterator(const GraphLaplacian &, Index) {}
    DummyIterator operator++() { return DummyIterator(); }
    double value() { return 0; }
    Index index() { return 0; }
    operator bool() { return false; }
  };
  using InnerIterator = DummyIterator;

  template <typename Rhs>
  Eigen::Product<GraphLaplacian, Rhs, Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return Eigen::Product<GraphLaplacian, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }

  GraphLaplacian(const std::vector<GraphLaplacianRow> &rows_)
      : rows_(rows_), n(this->rows_.size()) {}

  int outerSize() const { return n; }
  GraphLaplacian adjoint() const { return *this; }
  Eigen::VectorXd col(int j) const {
    Eigen::VectorXd ans(n);
    ans.setZero();
    ans(j) = 1;
    return (*this) * ans;
  }

  std::vector<GraphLaplacianRow> rows_;
  int n;
};

namespace Eigen {
namespace internal {

template <typename Rhs>
struct generic_product_impl<GraphLaplacian, Rhs, SparseShape, DenseShape,
                            GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<GraphLaplacian, Rhs,
                                generic_product_impl<GraphLaplacian, Rhs>> {
  typedef typename Product<GraphLaplacian, Rhs>::Scalar Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest &dst, const GraphLaplacian &lhs,
                            const Rhs &rhs, const Scalar &) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, rhs.rows()), [&](auto r) {
      for (auto i = r.begin(); i < r.end(); ++i) {
        double s = lhs.rows_[i].coeffs[0] * rhs[i];
        for (int e_i = 1; e_i < lhs.rows_[i].coeffs.size(); ++e_i) {
          s -= lhs.rows_[i].coeffs[e_i] * rhs[lhs.rows_[i].idxs[e_i]];
        }
        dst[i] += s;
      }
    });
  }
};

} // namespace internal
} // namespace Eigen

class GraphDiagonalPreconditioner {
public:
  GraphDiagonalPreconditioner() : evaluated(false) {}

  explicit GraphDiagonalPreconditioner(const GraphLaplacian &) {}

  GraphDiagonalPreconditioner &analyzePattern(const GraphLaplacian &) {
    return *this;
  }

  GraphDiagonalPreconditioner &factorize(const GraphLaplacian &g) {
    if (evaluated)
      return *this;
    diagonal.resize(g.rows());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, diagonal.rows()),
                      [&](auto r) {
                        for (auto i = r.begin(); i < r.end(); ++i)
                          diagonal(i) = g.rows_[i].coeffs(0);
                      });

    evaluated = true;
    return *this;
  }

  template <typename MatrixType>
  GraphDiagonalPreconditioner &compute(const GraphLaplacian &g) {
    return factorize(g);
  }

  template <typename Rhs> inline const Rhs solve(const Rhs &b) const {
    Eigen::VectorXd ans = b;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, diagonal.rows()),
                      [&](auto r) {
                        for (auto i = r.begin(); i < r.end(); ++i)
                          ans(i) /= diagonal(i);
                      });
    return ans.array() - ans.mean();
  }

  Eigen::ComputationInfo info() { return Eigen::Success; }

private:
  bool evaluated;
  Eigen::VectorXd diagonal;
};

#endif // GRAPHLALPLACIAN_HPP
