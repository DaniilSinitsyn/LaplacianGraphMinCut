#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <ranges>
#include <set>
#include <stack>
#include <vector>

#include "ApproximateCholesky.hpp"
#include "GraphLalplacian.hpp"
#include "cg.h"
#include <Eigen/Eigen>
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMatR = Eigen::SparseMatrix<double, Eigen::RowMajor>;

struct Edge;
struct Vert {
  Vert(int n) : n(n) {}
  int n;
  std::vector<Edge> edges;

  bool visited = false;
  int level = 0;
};

struct Edge {
  Edge(int v1, int v2, double cap) : v1(v1), v2(v2), cap(cap) {}
  int v1, v2;
  double cap;
  std::vector<Vert *> verts;
  int neigh(int v) { return v == v1 ? v2 : v1; }
};

struct Graph {
  Graph(const int ver_n)
      : verts_number(ver_n + 2), maxflow_upper_limit(0.0), chol(verts_number) {
    for (int i = 0; i < verts_number; ++i)
      verts.push_back(i);
    potentials_.resize(verts_number);
    potentials_.setZero();
    potentials_backup.resize(verts_number);
    b = Eigen::VectorXd::Zero(ver_n + 2);
    b(0) = 1;
    b(verts_number - 1) = -1;
    source_sum = 0;
    terminal_sum = 0;
  }

  void AddNode(int n, double source, double terminal) {
    if (abs(source) > 1e-8) {
      edges.emplace_back(0, n + 1, source);
      source_sum += source;
      verts[0].edges.push_back(edges.back());
      verts[n + 1].edges.push_back(edges.back());
      chol.AddEdge(0, n + 1, source);
    }
    if (abs(terminal) > 1e-8) {
      edges.emplace_back(n + 1, verts_number - 1, terminal);
      terminal_sum += terminal;
      verts[verts_number - 1].edges.push_back(edges.back());
      verts[n + 1].edges.push_back(edges.back());
      chol.AddEdge(n + 1, verts_number - 1, terminal);
    }
  }

  void AddEdge(int n1, int n2, double capacity, double reverseCapacity) {
    double w = (capacity + reverseCapacity);
    if (abs(w) > 1e-8) {
      edges.emplace_back(n1 + 1, n2 + 1, w);
      verts[n1 + 1].edges.push_back(edges.back());
      verts[n2 + 1].edges.push_back(edges.back());
      chol.AddEdge(n1 + 1, n2 + 1, w);
    }
  }

  double ComputeMaxFlow() {
    maxflow_upper_limit = std::min(source_sum, terminal_sum);

    rho = 3 * std::pow(edges.size(), 1.0 / 3.0) * std::pow(eps, -2.0 / 3.0);

    double min_dist = maxflow_upper_limit * 0.01;
    std::pair<double, double> flow_pair(0, maxflow_upper_limit);
    double good = 0;
    while (flow_pair.second - flow_pair.first > min_dist) {
      double center = (flow_pair.first + flow_pair.second) / 2.0;
      double f = knownFlowCut(center);
      std::cout << flow_pair.first << ' ' << flow_pair.second << " - " << f
                << "(" << center << ")" << '\n';
      if (f >= 0) {
        good = center;
        flow_pair.second = center;
        potentials_backup = potentials;
      } else {
        break;
      }
      break;
    }
    std::cout << cut_thresh << std::endl;
    return good;
  }
  inline bool IsNodeOnSrcSide(int n) const {
    return potentials_backup(n) >= cut_thresh;
  }

private:
  double knownFlowCut(double maxFlow) {
    weights.resize(edges.size());
    weights.setOnes();
    size_t N = 5 * std::pow(eps, -8.0 / 3.0) *
               std::pow(edges.size(), 1.0 / 3.0) * log(edges.size());

    for (size_t i = 0; i < N; ++i) {
      solveSparse(maxFlow);
      double mu = weights.sum();
      Eigen::VectorXd prev_w = weights;
      for (int j = 0; j < weights.size(); ++j) {
        weights(j) =
            weights(j) * (1 + eps / rho * abs(flow(j)) / edges[j].cap) +
            mu * eps * eps / edges.size() / rho;
      }

      potentials = potentials.array() - potentials(verts_number - 1);
      potentials /= potentials(0);

      double min_cut_capacity = std::numeric_limits<double>::max();
      double potential_max = potentials.maxCoeff();
      double good_cap = 0;
      std::cout << "Pot max : " << potential_max << std::endl;
      for (int k = 1; k < 40; ++k) {
        double a = potential_max * double(k) / 40;
        double cap = 0;
        for (size_t k = 0; k < edges.size(); ++k) {
          auto &e = edges[k];
          double max = std::max(potentials[e.v1], potentials[e.v2]);
          double min = std::min(potentials[e.v1], potentials[e.v2]);
          if (min < a && a < max) {
            cap += abs(e.cap);
          }
        }
        if (min_cut_capacity > cap && cap != 0) {
          cut_thresh = a;
          min_cut_capacity = cap;
          good_cap = cap;
        }
      }

      if (min_cut_capacity < 0)
        return -1;
      if (min_cut_capacity < maxFlow / (1 - 7 * eps))
        return min_cut_capacity;
    }
    return -1.0;
  }
#define EIGEN_SOLVER false
  void solveSparse(double maxFlow) {
    // exportPreconditioner(maxFlow);

    std::vector<GraphLaplacianRow> rows;
    rows.reserve(verts.size());
    for (auto &v : verts) {
      std::vector<long> idxs(v.edges.size() + 1);
      idxs[0] = v.n;
      Eigen::VectorXd coeffs(v.edges.size() + 1);
      double s = 0;
      for (size_t i = 0; i < v.edges.size(); ++i) {
        auto &e = v.edges[i];
        double cap = e.cap * e.cap;
        idxs[i + 1] = e.neigh(v.n);
        coeffs[i + 1] = cap;
        s += cap;
      }
      coeffs[0] = s;
      rows.emplace_back(coeffs, idxs);
    }

    GraphLaplacian laplacain(rows);

    std::cout << maxFlow << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    SpMat B(verts_number, edges.size());
    B.reserve(Eigen::VectorXi::Constant(edges.size(), 2));

    for (size_t i = 0; i < edges.size(); ++i) {
      B.insert(edges[i].v1, i) = -1;
      B.insert(edges[i].v2, i) = 1;
    }

    Eigen::VectorXd resistances(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
      resistances(i) = edges[i].cap * edges[i].cap;
    }
    SpMat A = B * resistances.asDiagonal() * SpMat(B.transpose());
    A.makeCompressed();

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Construction of A : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                       start)
                     .count()
              << std::endl;

    static int ii = 0;
    SpMatR A_ = A;
    if (ii == 0) {
#if EIGEN_SOLVER
      cg.analyzePattern(A_);
#else
      cg.analyzePattern(laplacain);
#endif
      ii++;
    }
#if EIGEN_SOLVER
    cg.factorize(A_);
#else
    cg.factorize(laplacain);
#endif

    cg.setMaxIterations(100);
    cg.setTolerance(1e-3);
    potentials_.setZero();
    {
      Eigen::VectorXd bb = double(maxFlow) * b;
      auto start = std::chrono::high_resolution_clock::now();
      Eigen::VectorXd x = cg.solve(bb);
      auto stop = std::chrono::high_resolution_clock::now();
      potentials_ = x;
      std::cout << "elapsed time : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                         start)
                       .count()
                << std::endl;
    }
#if 1
    std::cout << (A * potentials_ - maxFlow * b).norm() << std::endl;
    std::cout << cg.error() << " " << cg.iterations() << std::endl;
#endif
    potentials = potentials_;
    flow = resistances.array() * (SpMat(B.transpose()) * potentials).array();
  }
#if EIGEN_SOLVER
  Eigen::ConjugateGradient<SpMatR, Eigen::Lower | Eigen::Upper,
                           Eigen::DiagonalPreconditioner<double>>
      cg;
#else
  Eigen::ConjugateGradient<GraphLaplacian, Eigen::Lower | Eigen::Upper,
                           GraphDiagonalPreconditioner>
      cg;
#endif
  void exportPreconditioner(double maxFlow) {
    (void)maxFlow;
    constructSpanningTree();
    srand(time(0));
    Eigen::VectorXd rhs = Eigen::VectorXd::Random(b.rows());
    rhs = rhs.array() - rhs.mean();
    // rhs = maxFlow * b;
    auto x = solve(rhs);

    std::cout << edges.size() << std::endl;
    std::cout << verts.size() << std::endl;

    std::cout << eliminations.size() << std::endl;

    SpMat B(verts_number, eliminations.size());
    B.reserve(Eigen::VectorXi::Constant(eliminations.size(), 2));

    Eigen::VectorXd resistances(eliminations.size());
    int i = 0;
    for (auto &e : eliminations) {
      B.insert(e.parent, i) = -1;
      B.insert(e.i, i) = 1;

      resistances(i) = e.cap;
      ++i;
    }

    B.makeCompressed();
    A1 = B * resistances.asDiagonal() * SpMat(B.transpose());
    A1.makeCompressed();
    std::cout << "err tree : " << (A1 * x - rhs).norm() << std::endl;

    cg1.setMaxIterations(1000);

    cg1.analyzePattern(A1);

    cg1.factorize(A1);
    Eigen::VectorXd ans = cg1.solveWithGuess(rhs, x);
    potentials_ = ans;
    std::cout << cg1.error() << " " << cg1.iterations() << std::endl;
    std::cout << (ans - x).norm() << std::endl;

    std::cout << (A1 * ans - rhs).norm() << std::endl;
  }

  SpMat A1;
  Eigen::ConjugateGradient<
      SpMat, Eigen::Upper | Eigen::Lower,
      Eigen::IncompleteCholesky<double, Eigen::Upper | Eigen::Lower>>
      cg1;

public:
  Eigen::VectorXd solve(const Eigen::VectorXd &b_) const {
    Eigen::VectorXd ans, lb = b_;

    for (auto i = eliminations.rbegin(); i != eliminations.rend(); ++i) {
      lb[i->parent] += lb[i->i];
    }

    ans.resize(b_.rows());
    ans[root] = 0;

    for (size_t i = 0; i < eliminations.size(); ++i) {
      auto &e = eliminations[i];
      ans[e.i] = lb[e.i] / e.cap + ans[e.parent];
    }

    ans = ans.array() - ans.mean();
    return ans;
  }

private:
  struct Elim {
    Elim(int i, int parent, double cap) : i(i), parent(parent), cap(cap) {}
    int i;
    int parent;
    double cap;
  };
  std::vector<Elim> eliminations;

  int root;

  void constructSpanningTree() {
    root = 0;
    eliminations.clear();
    eliminations.reserve(verts.size() - 1);

    verts[root].visited = true;
    std::stack<std::pair<int, int>> vert_ids;
    vert_ids.push({root, 0});
    while (!vert_ids.empty()) {
      auto [v_i, level] = vert_ids.top();
      vert_ids.pop();
      for (auto e : verts[v_i].edges) {
        int neigh = e.neigh(v_i);
        if (!verts[neigh].visited) {
          verts[neigh].visited = true;
          verts[v_i].level = level + 1;

          double cap = e.cap * e.cap;

          if (neigh != verts_number - 1) {
            vert_ids.push({neigh, level + 1});
            eliminations.emplace_back(neigh, v_i, cap);
          }
        }
      }
    }
    int max_i = 0, max_lev = 0;
    double cap;
    for (auto &e : verts.back().edges) {
      int i = e.neigh(verts_number - 1);
      if (verts[i].level > max_lev) {
        max_i = i;
        max_lev = verts[i].level;
        cap = e.cap;
      }
    }
    eliminations.emplace_back(verts_number - 1, max_i, cap);
    verts.back().level = max_lev + 1;
  }

  std::deque<Edge> edges;
  std::deque<Vert> verts;

  const int verts_number;

  Eigen::VectorXd b;

  Eigen::VectorXd weights;
  Eigen::VectorXd flow;
  Eigen::VectorXd potentials;
  Eigen::VectorXd potentials_backup;

  Eigen::VectorXd potentials_;

  double maxflow_upper_limit;
  double source_sum, terminal_sum;
  double eps = 0.01;
  double rho;
  double cut_thresh;

  approximate_cholesky::ApproximateCholesky chol;
};

#endif // GRAPH_HPP
