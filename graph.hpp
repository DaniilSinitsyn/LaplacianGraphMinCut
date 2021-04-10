#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <random>
#include <ranges>
#include <stack>
#include <vector>

#include <Eigen/Eigen>

#include "cusp_cg.h"

struct Vert {
  int n;
};

struct Edge {
  Edge(int v1, int v2, float cap) : v1(v1), v2(v2), cap(cap) {}
  int v1, v2;
  float cap;
};

struct Graph {
  Graph(const int ver_n) : verts_number(ver_n + 2), maxflow_upper_limit(0.0) {
    potentials_.resize(verts_number);
    potentials_.setZero();
    b = Eigen::VectorXf::Zero(ver_n + 2);
    b(0) = 1;
    b(verts_number - 1) = -1;
    source_sum = 0;
    terminal_sum = 0;
  }

  void AddNode(int n, float source, float terminal) {
    if (abs(source) > 1e-8) {
      edges.emplace_back(0, n + 1, source);
      source_sum += source;
    }
    if (abs(terminal) > 1e-8)
      edges.emplace_back(n + 1, verts_number - 1, terminal);
    terminal_sum += terminal;
  }

  void AddEdge(int n1, int n2, float capacity, float reverseCapacity) {
    float w = (capacity + reverseCapacity);
    if (abs(w) > 1e-8)
      edges.emplace_back(n1 + 1, n2 + 1, w);
  }

  float ComputeMaxFlow() {
    maxflow_upper_limit = std::min(source_sum, terminal_sum);

    rho = 3 * std::pow(edges.size(), 1.0 / 3.0) * std::pow(eps, -2.0 / 3.0);

    float min_dist = maxflow_upper_limit * 0.01;
    std::pair<float, float> flow_pair(0, maxflow_upper_limit);
    float good;
    while (flow_pair.second - flow_pair.first > min_dist) {
      float center = (flow_pair.first + flow_pair.second) / 2.0;
      float f = knownFlowCut(center);
      std::cout << flow_pair.first << ' ' << flow_pair.second << " - " << f
                << "(" << center << ")" << '\n';
      if (f >= 0) {
        good = center;
        flow_pair.second = center;
        potentials_backup = potentials;
      } else {
        break;
      }

      return f;
    }
    std::cout << cut_thresh << std::endl;
    return good;
  }
  inline bool IsNodeOnSrcSide(int n) const {
    return potentials_backup(n) >= cut_thresh - 0.05;
  }

private:
  float knownFlowCut(float maxFlow) {
    weights.resize(edges.size());
    weights.setOnes();
    size_t N = 5 * std::pow(eps, -8.0 / 3.0) *
               std::pow(edges.size(), 1.0 / 3.0) * log(edges.size());

    for (size_t i = 0; i < N; ++i) {
      solveSparse(maxFlow);
      float mu = weights.sum();
      Eigen::VectorXf prev_w = weights;
      for (int j = 0; j < weights.size(); ++j) {
        weights(j) =
            weights(j) * (1 + eps / rho * abs(flow(j)) / edges[j].cap) +
            mu * eps * eps / edges.size() / rho;
      }
      std::cout << i << " " << (prev_w - weights).norm() << std::endl;

      potentials = potentials.array() - potentials(verts_number - 1);
      potentials /= potentials(0);

      float min_cut_capacity = std::numeric_limits<float>::max();
      float potential_max = potentials.maxCoeff();
      float good_cap = 0;
      std::cout << "Pot max : " << potential_max << std::endl;
      for (int k = 1; k < 40; ++k) {
        float a = potential_max * float(k) / 40;
        float curr_cut = 0;
        float cap = 0;
        for (size_t k = 0; k < edges.size(); ++k) {
          auto &e = edges[k];
          float max = std::max(potentials[e.v1], potentials[e.v2]);
          float min = std::min(potentials[e.v1], potentials[e.v2]);
          if (min < a && a < max) {
            // curr_cut += e.cap;
            cap += abs(e.cap);
            if (min == potentials[e.v1])
              curr_cut += flow(k);
            else
              curr_cut -= flow(k);
          }
        }
        if (min_cut_capacity > curr_cut && curr_cut != 0) {
          cut_thresh = a;
          min_cut_capacity = curr_cut;
          good_cap = cap;
        }
      }
      std::cout << min_cut_capacity << ' ' << cut_thresh << std::endl;
      if (min_cut_capacity < 0)
        return -1;
      if (min_cut_capacity < maxFlow / (1 - 7 * eps))
        return good_cap;
    }
    return -1.0;
  }

  void solveSparse(float maxFlow) {
    SpMat B(verts_number, edges.size());
    B.reserve(Eigen::VectorXi::Constant(edges.size(), 2));
    for (size_t i = 0; i < edges.size(); ++i) {
      B.insert(edges[i].v1, i) = -1;
      B.insert(edges[i].v2, i) = 1;
    }

    Eigen::VectorXf resistances(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
      resistances(i) = edges[i].cap * edges[i].cap / weights(i);
    }
    SpMat A = B * resistances.asDiagonal() * SpMat(B.transpose());
    A.makeCompressed();

#if 1
    potentials_ = cusp_cg_solve(A, maxFlow * b);
#else
    static int ii = 0;
    if (ii == 0) {
      cg.analyzePattern(A);
      ii++;
    }
    cg.factorize(A);
    cg.setMaxIterations(1000);
    cg.setTolerance(1e-6);

    potentials_ = cg.solveWithGuess(maxFlow * b, potentials_);
    std::cout << cg.error() << " " << cg.iterations() << std::endl;
#endif
    potentials = potentials_;

    flow = resistances.array() * (SpMat(B.transpose()) * potentials).array();

    auto kek = (B * flow).eval();
    std::cout << kek(0) << ' ' << kek(verts_number - 1) << ' '
              << kek.segment(1, verts_number - 2).norm() << std::endl;
  }

  Eigen::ConjugateGradient<SpMat, Eigen::Upper | Eigen::Lower,
                           Eigen::DiagonalPreconditioner<float>>
      cg;

  std::vector<Edge> edges;

  const int verts_number;

  Eigen::VectorXf b;

  Eigen::VectorXf weights;
  Eigen::VectorXf flow;
  Eigen::VectorXf potentials;
  Eigen::VectorXf potentials_backup;

  Eigen::VectorXf potentials_;

  float maxflow_upper_limit;
  float source_sum, terminal_sum;
  float eps = 0.01;
  float rho;
  float cut_thresh;
};

#endif // GRAPH_HPP
