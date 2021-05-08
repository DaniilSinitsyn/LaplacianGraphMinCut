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

#include "cg.h"
#include <Eigen/Eigen>
#include <tbb/tbb.h>

using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor>;

struct Edge;
struct Vert {
  Vert(int n) : n(n) {}
  int n;
  std::vector<Edge *> edges;

  bool visited = false;
  int parent = -1;
  int degree = 1;
  float x = 0;
  float cap = 0;
};

struct Edge {
  Edge(int v1, int v2, float cap) : v1(v1), v2(v2), cap(cap) {}
  int v1, v2;
  float cap;
  std::vector<Vert *> verts;
  int neigh(int v) { return v == v1 ? v2 : v1; }
};

struct Graph {
  Graph(const int ver_n) : verts_number(ver_n + 2), maxflow_upper_limit(0.0) {
    for (int i = 0; i < verts_number; ++i)
      verts.push_back(i);
    potentials_.resize(verts_number);
    potentials_.setZero();
    potentials_backup.resize(verts_number);
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
      verts[0].edges.push_back(&edges.back());
      verts[n + 1].edges.push_back(&edges.back());
    }
    if (abs(terminal) > 1e-8) {
      edges.emplace_back(n + 1, verts_number - 1, terminal);
      terminal_sum += terminal;
      verts[verts_number - 1].edges.push_back(&edges.back());
      verts[n + 1].edges.push_back(&edges.back());
    }
  }

  void AddEdge(int n1, int n2, float capacity, float reverseCapacity) {
    float w = (capacity + reverseCapacity);
    if (abs(w) > 1e-8) {
      edges.emplace_back(n1 + 1, n2 + 1, w);
      verts[n1 + 1].edges.push_back(&edges.back());
      verts[n2 + 1].edges.push_back(&edges.back());
    }
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
    }
    std::cout << cut_thresh << std::endl;
    return good;
  }
  inline bool IsNodeOnSrcSide(int n) const {
    return potentials_backup(n) >= cut_thresh;
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

      potentials = potentials.array() - potentials(verts_number - 1);
      potentials /= potentials(0);

      float min_cut_capacity = std::numeric_limits<float>::max();
      float potential_max = potentials.maxCoeff();
      float good_cap = 0;
      std::cout << "Pot max : " << potential_max << std::endl;
      for (int k = 1; k < 40; ++k) {
        float a = potential_max * float(k) / 40;
        float cap = 0;
        for (size_t k = 0; k < edges.size(); ++k) {
          auto &e = edges[k];
          float max = std::max(potentials[e.v1], potentials[e.v2]);
          float min = std::min(potentials[e.v1], potentials[e.v2]);
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

  void solveSparse(float maxFlow) {
    std::cout << maxFlow << std::endl; exit(0);
    constructSpanningTree();
    // exportPreconditioner(maxFlow);
    // exit(0);
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
    std::ofstream file("graph.txt");
    file << std::setprecision(10);
    for (size_t i = 0; i < edges.size(); ++i) {
      float r = edges[i].cap * edges[i].cap / weights(i);
      file << (edges[i].v1 + 1) << ' ' << (edges[i].v2 + 1) << ' ' << r << '\n';
    }

    file.close();
    // exit(0);
    float tol = 1e-6;
    int max_iter = 100;
    int status = CG(A, potentials_, (maxFlow * b).eval(), *this, max_iter, tol);

    std::cout << (A * potentials_ - maxFlow * b).norm() << std::endl;
    std::cout << tol << ' ' << status << ' ' << max_iter << std::endl;
    static int ii = 0;
    if (ii == 0) {
      cg.analyzePattern(A);
      ii++;
    }
    cg.factorize(A);
    cg.setMaxIterations(10000);
    cg.setTolerance(1e-6);
    potentials_.setZero();
    auto start = std::chrono::high_resolution_clock::now();
    potentials_ = cg.solveWithGuess(maxFlow * b, potentials_);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "elapsed time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                       start)
                     .count()
              << std::endl;

    std::cout << (A * potentials_ - maxFlow * b).norm() << std::endl;

    std::cout << cg.error() << " " << cg.iterations() << std::endl;
    std::cout << potentials_.head<10>().transpose() << std::endl;
    std::cout << potentials_.tail<10>().transpose() << std::endl;

    exit(0);
    potentials = potentials_;
    flow = resistances.array() * (SpMat(B.transpose()) * potentials).array();

    auto kek = (B * flow).eval();
    std::cout << kek(0) << ' ' << kek(verts_number - 1) << ' '
              << kek.segment(1, verts_number - 2).norm() << std::endl;
  }

  Eigen::ConjugateGradient<SpMat, Eigen::Upper | Eigen::Lower,
                           Eigen::DiagonalPreconditioner<float>>
      cg;

  void exportPreconditioner(float maxFlow) {
    exit(0);
    (void)maxFlow;
    srand(time(0));
    Eigen::VectorXf rhs = Eigen::VectorXf::Random(b.rows());
    rhs = rhs.array() - rhs.mean();
    auto x = solve(rhs);
    std::cout << edges.size() << std::endl;
    std::cout << verts.size() << std::endl;

    std::cout << eliminations.size() << std::endl;

    SpMat B(verts_number, eliminations.size());
    B.reserve(Eigen::VectorXi::Constant(eliminations.size(), 2));

    int i = 0;
    for (auto &e : eliminations) {
      B.insert(e.parent, i) = 1;
      B.insert(e.i, i) = -1;
      ++i;
    }

    Eigen::VectorXf resistances(eliminations.size());
    i = 0;
    for (auto &e : eliminations) {
      resistances(i) = e.cap * e.cap;
      i++;
    }

    B.makeCompressed();
    SpMat A = B * resistances.asDiagonal() * SpMat(B.transpose());
    A.makeCompressed();
    std::ofstream file("precon.dat");
    std::ofstream file1("stats.txt");

    std::cout << (A * Eigen::VectorXf::Ones(x.rows())).norm() / A.norm()
              << std::endl;

    std::cout << (A * x - rhs).norm() << std::endl;
    Eigen::VectorXf shit = (A * x);
    for (auto &e : eliminations) {
      file << e.i << ' ' << e.parent << ' ' << e.cap << '\n';
    }
    Eigen::ConjugateGradient<SpMat, Eigen::Upper | Eigen::Lower,
                             Eigen::DiagonalPreconditioner<float>>
        cg1;
    cg1.setMaxIterations(1000);
    cg1.setTolerance(1e-6);

    cg1.analyzePattern(A);

    cg1.factorize(A);
    Eigen::VectorXf ans = cg1.solveWithGuess(rhs, x);
    ans = ans.array() - ans.mean();
    for (int i = 0; i < shit.rows(); ++i) {
      file1 << shit[i] << ' ' << rhs[i] << ' ' << x[i] << ' ' << ans[i] << ' '
            << (shit[i] - rhs[i]) << std::endl;
    }

    file1.close();
    file.close();
    std::cout << cg1.error() << " " << cg1.iterations() << std::endl;
    std::cout << (ans - x).norm() << std::endl;

    std::cout << (A * ans - rhs).norm() << std::endl;

    exit(0);
  }

public:
  Eigen::VectorXf solve(const Eigen::VectorXf &b_) const {
    Eigen::VectorXf ans, lb = b_.array() - b_.mean();
    for (auto i = eliminations.rbegin(); i != eliminations.rend(); ++i) {
      lb[i->parent] += lb[i->i];
    }
    ans.resize(b_.rows());
    ans[root] = 0;
    for (size_t i = 0; i < eliminations.size(); ++i) {
      auto &e = eliminations[i];
      ans[e.i] = lb[e.i] / e.cap / e.cap + ans[e.parent];
    }

    ans = ans.array() - ans.mean();
    return ans;
  }

private:
  struct Elim {
    Elim(int i, int parent, float cap, int level)
        : i(i), parent(parent), cap(cap), level(level) {}
    int i;
    int parent;
    float cap;
    int level;
  };
  std::vector<Elim> eliminations;

  int root = 10;

  void constructSpanningTree() {
    for (auto &v : verts) {
      float sum = 0;
      for (auto &e : v.edges) {
        sum += e->cap;
      }
      v.cap = sum / 2;
    }
    eliminations.clear();
    eliminations.reserve(verts.size() - 1);

    verts[root].visited = true;
    verts[root].parent = -1;
    std::queue<std::pair<int, int>> vert_ids;
    vert_ids.push({root, 0});
    while (!vert_ids.empty()) {
      auto [v_i, level] = vert_ids.front();
      vert_ids.pop();
      for (auto *e : verts[v_i].edges) {
        int neigh = e->neigh(v_i);
        if (!verts[neigh].visited) {
          verts[neigh].visited = true;
          verts[neigh].parent = v_i;

          float cap = verts[neigh].cap;
          eliminations.emplace_back(neigh, v_i, cap, level + 1);

          if (neigh != verts_number - 1 && neigh != 0)
            vert_ids.push({e->neigh(v_i), level + 1});
        }
      }
    }
    // std::sort(eliminations.begin(), eliminations.end(),
    //          [](auto a, auto b) { return a.level < b.level; });
  }

  std::deque<Edge> edges;
  std::deque<Vert> verts;

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
