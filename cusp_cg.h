#include <Eigen/Eigen>

using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor>;

Eigen::VectorXf cusp_cg_solve(const SpMat &A, const Eigen::VectorXf &b);

