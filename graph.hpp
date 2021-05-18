#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <Eigen/Eigen>
using float_type = float;
using VecX = Eigen::Matrix<float_type, -1, 1>;
using SpMat = Eigen::SparseMatrix<float_type, Eigen::RowMajor>;
typedef Eigen::Triplet<float_type> Triple;

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

#include "GraphLalplacian.hpp"
#include "cg.h"

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
      : verts_number(ver_n + 2), maxflow_upper_limit(0.0),
        laplacain(ver_n + 2) {
    NonZeros = Eigen::VectorXi::Ones(verts_number);
    for (int i = 0; i < verts_number; ++i)
      verts.push_back(i);
    potentials.resize(verts_number);
    potentials.setZero();
    b = VecX::Zero(ver_n + 2);
    source_sum = 0;
    terminal_sum = 0;
    fillThreshCounter();
  }

  void fillThreshCounter() {
    thresh_counter.clear();
    int N = 40;
    for (int k = 1; k < N; ++k) {
      double a = 1.0 * double(k) / double(N);
      thresh_counter.emplace_back(a, 0.0);
    }
  }

  void AddNode(int n, double source, double terminal) {
    if (abs(source) > 1e-8) {
      edges.emplace_back(0, n + 1, source);
      source_sum += source;
      verts[0].edges.push_back(edges.back());
      verts[n + 1].edges.push_back(edges.back());

      double cap = source * source;
      laplacain.rows_[0].push(n + 1, cap);
      laplacain.rows_[n + 1].push(0, cap);
      NonZeros(0)++;
      NonZeros(n + 1)++;
      triples.push_back(Triple(0, n + 1, -cap));
      triples.push_back(Triple(n + 1, 0, -cap));
    }
    if (abs(terminal) > 1e-8) {
      edges.emplace_back(n + 1, verts_number - 1, terminal);
      terminal_sum += terminal;
      verts[verts_number - 1].edges.push_back(edges.back());
      verts[n + 1].edges.push_back(edges.back());

      double cap = terminal * terminal;
      laplacain.rows_[verts_number - 1].push(n + 1, cap);
      laplacain.rows_[n + 1].push(verts_number - 1, cap);

      NonZeros(verts_number - 1)++;
      NonZeros(n + 1)++;
      triples.push_back(Triple(verts_number - 1, n + 1, -cap));
      triples.push_back(Triple(n + 1, verts_number - 1, -cap));
    }
  }

  void AddEdge(int n1, int n2, double capacity, double reverseCapacity) {
    double w = (capacity + reverseCapacity);
    if (abs(w) > 1e-8) {
      edges.emplace_back(n1 + 1, n2 + 1, w);
      verts[n1 + 1].edges.push_back(edges.back());
      verts[n2 + 1].edges.push_back(edges.back());

      double cap = w * w;
      laplacain.rows_[n1 + 1].push(n2 + 1, cap);
      laplacain.rows_[n2 + 1].push(n1 + 1, cap);

      NonZeros(n1 + 1)++;
      NonZeros(n2 + 1)++;
      triples.push_back(Triple(n1 + 1, n2 + 1, -cap));
      triples.push_back(Triple(n2 + 1, n1 + 1, -cap));
    }
  }

  double ComputeMaxFlow() {
    for (int i = 0; i < verts_number; ++i) {
      triples.push_back(Triple(i, i, laplacain.rows_[i].coeffs[0]));
    }
    maxflow_upper_limit = std::min(source_sum, terminal_sum) / 2;
    b(0) = maxflow_upper_limit;
    b(verts_number - 1) = -b(0);
    double f = knownFlowCut(maxflow_upper_limit);

    return f;
  }
  inline bool IsNodeOnSrcSide(int n) const {
    return potentials(n) >= cut_thresh;
  }

private:
  double knownFlowCut(double maxFlow) {
    solveSparse(maxFlow);
    potentials = potentials.array() - potentials(verts_number - 1);
    potentials /= potentials(0);
    double max_a = potentials.maxCoeff();
    std::cout << "Potential max " << max_a << std::endl;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, thresh_counter.size()), [&](auto r) {
          for (auto i = r.begin(); i < r.end(); ++i) {
            auto &t = thresh_counter[i];
            double a = max_a * t.first;
            t.second = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, edges.size()), 0.0,
                [&](auto r, double total) {
                  for (auto i = r.begin(); i < r.end(); ++i) {
                    auto &e = edges[i];
                    double max = std::max(potentials[e.v1], potentials[e.v2]);
                    double min = std::min(potentials[e.v1], potentials[e.v2]);
                    if (min < a && a < max)
                      total += e.cap;
                  }
                  return total;
                },
                std::plus<double>());
          }
        });
    // tbb::parallel_for(
    //    tbb::blocked_range<size_t>(0, thresh_counter.size()), [&](auto r) {
    //      for (auto i = r.begin(); i < r.end(); ++i) {
    //        double cap = 0.0;
    //        double a = max_a * thresh_counter[i].first;
    //        for (auto &e : edges) {
    //          double max = std::max(potentials[e.v1], potentials[e.v2]);
    //          double min = std::min(potentials[e.v1], potentials[e.v2]);
    //          if (min < a && a < max) {
    //            cap += abs(e.cap);
    //          }
    //        }
    //        thresh_counter[i].second = cap;
    //      }
    //    });

    int max_i = std::distance(
        thresh_counter.begin(),
        std::max_element(thresh_counter.begin(), thresh_counter.end(),
                         [](auto &a, auto &b) { return a.second > b.second; }));
    cut_thresh = thresh_counter[max_i].first;
    std::cout << cut_thresh << " " << thresh_counter[max_i].second << std::endl;
    ;
    return thresh_counter[max_i].second;
  }
#define EIGEN_SOLVER true
  void solveSparse(double maxFlow) {
    std::cout << maxFlow << std::endl;
    SpMat A(verts_number, verts_number);
    A.setFromTriplets(triples.begin(), triples.end());
    A.makeCompressed();

    float tol = 1e-3;
    int max_iter = 50;
#if 1
    {
      Eigen::DiagonalPreconditioner<float_type> diag(A);
      CG(A, potentials, b, diag, max_iter, tol);
    }
#else

    static int ii = 0;
    if (ii == 0) {
#if EIGEN_SOLVER
      cg.analyzePattern(A);
#else
      cg.analyzePattern(laplacain);
#endif
      ii++;
    }
#if EIGEN_SOLVER
    cg.factorize(A);
#else
    cg.factorize(laplacain);
#endif

    cg.setMaxIterations(max_iter);
    cg.setTolerance(tol);
    { potentials = cg.solve(b); }
#if 1
    std::cout << (A * potentials - b).norm() << std::endl;
    std::cout << cg.error() << " " << cg.iterations() << std::endl;
#endif
#endif
  }
#if EIGEN_SOLVER
  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper,
                           Eigen::DiagonalPreconditioner<float_type>>
      cg;
#else
  Eigen::ConjugateGradient<GraphLaplacian, Eigen::Lower | Eigen::Upper,
                           GraphDiagonalPreconditioner>
      cg;
#endif

private:
  std::vector<std::pair<double, double>> thresh_counter;
  Eigen::VectorXi NonZeros;
  std::deque<Edge> edges;
  std::deque<Vert> verts;
  std::vector<Triple> triples;

  const int verts_number;

  VecX b;

  VecX weights;
  VecX flow;
  VecX potentials;

  double maxflow_upper_limit;
  double source_sum, terminal_sum;
  double eps = 0.01;
  double rho;
  double cut_thresh;

  GraphLaplacian laplacain;
};

#endif // GRAPH_HPP
