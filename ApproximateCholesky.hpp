#ifndef APPROXIMATECHOLESKY_HPP
#define APPROXIMATECHOLESKY_HPP

#include <Eigen/Eigen>
#include <deque>
#include <map>
#include <memory>
#include <vector>

namespace approximate_cholesky {

static constexpr int MultiDegreeMult = 20;

struct Edge;

struct EdgePtr {
  EdgePtr(Edge *e) : e(e) {}
  Edge *e;
  EdgePtr *next = nullptr;
  EdgePtr *prev = nullptr;
  void del() {
    if (next)
      next->prev = prev;
    if (prev)
      prev->next = next;
  }
};

struct Vert {
  Vert(int n) : n(n) {}

  /** call when edge become invalid */
  void decreaseDegree(bool multiedge, float edge_w) {
    degree -= 1;
    multidegree -= (multiedge ? MultiDegreeMult : 1);
    weight -= edge_w;
  }

  /** call when new edge is added to graph
   *
   * new edge `multidegree` property is known at compile time
   * */
  void increaseDegree(float edge_w, bool multiedge) {
    degree += 1;
    if (multiedge) {
      multidegree += MultiDegreeMult;
    } else {
      multidegree += 1;
    }
    weight += edge_w;
  }

  int n;                    // vertex id
  int degree = 0;           // number of distinct edges
  int multidegree = 0;      // number of edges with multiedges repetitions
                            // (MultiDegreeMult)
  EdgePtr *edges = nullptr; // adjoint edges
  float weight = 0;         // summ of valid edges weight
};

struct Edge {
  Edge(Vert *v1, Vert *v2, float weight, bool multiedge = false)
      : v1(v1), v2(v2), multiedge(multiedge), weight(weight),
        multiweight(weight / (multiedge ? MultiDegreeMult : 1)) {
    e1 = std::make_unique<EdgePtr>(this);
    e2 = std::make_unique<EdgePtr>(this);

    v1->increaseDegree(weight, multiedge);
    v2->increaseDegree(weight, multiedge);

    if (!v1->edges) {
      v1->edges = e1.get();
    } else {
      e1->next = v1->edges;
      v1->edges->prev = e1.get();
      v1->edges = e1.get();
    }

    if (!v2->edges) {
      v2->edges = e2.get();
    } else {
      e2->next = v2->edges;
      v2->edges->prev = e2.get();
      v2->edges = e2.get();
    }
  }

  Vert *v1 = nullptr;
  Vert *v2 = nullptr;
  std::unique_ptr<EdgePtr> e1, e2;
  /** get adjoint vertex */
  Vert *neigh(Vert *v) { return v == v1 ? v2 : v1; }

  void unvalidate() {
    e1->del();
    e2->del();
    v1->decreaseDegree(multiedge, weight);
    v2->decreaseDegree(multiedge, weight);
  }
  bool multiedge;    // flag  if this is initial edge, which requires repetition
  float weight;      // weight
  float multiweight; // normalized bu `multiedge` property weight
};

/**
 * Approximate Sparse LDLT decomposition of Graph Laplacian based on
 *
 * Approximate Gaussian Elimination for Laplacians – Fast, Sparse, and Simple
 *                    by Rasmus Kyng, Sushant Sachdeva
 *
 */
class ApproximateCholesky {
  struct L {
    L(int n) {
      rows.resize(n);
      cols.resize(n);
    }

    /**
     * Eliminate vertex and all its edges (star) from graph
     */
    void pushVertexStar(Vert &v) {
      float denom = v.weight;
      if (abs(denom) < 1e-8)
        return;

      /* normed by denom */
      // rows[v.n].emplace_back(1, v.n);
      // cols[v.n].emplace_back(1, v.n);

      std::map<std::pair<int, int>, float> val_map;
      for (auto e_ptr = v.edges; e_ptr != nullptr; e_ptr = e_ptr->next) {
        auto e = e_ptr->e;
        Vert *v1 = e->neigh(&v);

        /** NOTE that sign in Laplacian is negated */
        float w = -e->weight / denom;

        auto elem = val_map.find({v.n, v1->n});
        if (elem != val_map.end())
          elem->second += w;
        else
          val_map.insert({{v.n, v1->n}, w});

        /** unvalidate edge and decrease multidegree of v1 */
        e->unvalidate();
      }
      for (auto &[coord, w] : val_map) {
        cols[coord.first].emplace_back(coord.second, w);
        rows[coord.second].emplace_back(coord.first, w);
      }
    }

    /**
     * last matrix of elimination if 1x1
     */
    void pushLast() {
      // int n = rows.size() - 1;

      // cols[n].emplace_back(n, 1);
      // rows[n].emplace_back(n, 1);
    }

    using Val = std::pair<int, float>;
    std::vector<std::vector<Val>> rows;
    std::vector<std::vector<Val>> cols;
  };

public:
  /** initialized containers, verticies with zeros */
  ApproximateCholesky(const int n) : lmat(n) {
    srand(time(0));
    dmat.resize(n);
    verts.reserve(n);
    for (int i = 0; i < n; ++i)
      verts.emplace_back(i);
  }

  void AddEdge(int n1, int n2, float w) {
    edges.emplace_back(&verts[n1], &verts[n2], w, true);
  }

  void factorize() {
    for (size_t i = 0; i < verts.size() - 1; ++i) {
      dmat(i) = verts[i].weight;

      sampleCliqueApproximation(&verts[i]);
      lmat.pushVertexStar(verts[i]);
    }
    dmat(verts.size() - 1) = verts.back().weight;
    lmat.pushLast();
  }

  Eigen::VectorXf solve(const Eigen::VectorXf b) const {
    Eigen::VectorXf y = b;
    for (int i = 0; i < dmat.rows(); ++i) {
      for (auto &row : lmat.rows[i]) {
        y[i] -= y[row.first] * row.second;
      }
    }

    for (int i = 0; i < dmat.rows(); ++i) {
      if (abs(dmat[i]) >= 1e-6)
        y[i] /= dmat[i];
    }

    for (int i = dmat.rows() - 1; i >= 0; --i) {
      for (auto &col : lmat.cols[i]) {
        y[i] -= y[col.first] * col.second;
      }
    }

    return y.array() - y.mean();
  }

private:
  /** Approximate Clique for vert `v` and its neighbours
   *  saves new edges directly to `edges` and `verts` containers
   */
  void sampleCliqueApproximation(Vert *v) {
    for (int i = 0; i < v->degree; ++i) {
      float r = (rand() % (v->multidegree * 100));
      r *= v->weight / float(v->multidegree * 100);

      Edge *e1 = nullptr;

      for (auto *e_ptr = v->edges; e_ptr != nullptr; e_ptr = e_ptr->next) {
        auto e = e_ptr->e;

        if (r <= e->weight) {
          e1 = e;
          break;
        } else {
          r -= e->weight;
        }
      }

      int r1 = rand() % v->multidegree;
      Edge *e2 = nullptr;
      for (auto *e_ptr = v->edges; e_ptr != nullptr; e_ptr = e_ptr->next) {
        auto e = e_ptr->e;
        int edge_repetition = e->multiedge ? MultiDegreeMult : 1;
        if (r1 <= edge_repetition) {
          e2 = e;
          break;
        } else {
          r1 -= edge_repetition;
        }
      }

      if (e1 && e2) {
        Vert *v1 = e1->neigh(v);
        Vert *v2 = e2->neigh(v);

        if (v1->n != v2->n) {
          float w = e1->multiweight * e2->multiweight /
                    (e1->multiweight + e2->multiweight);
          edges.emplace_back(v1, v2, w);
        }
      }
    }
  }

  std::deque<Edge> edges;
  std::vector<Vert> verts;

  L lmat;
  Eigen::VectorXf dmat;
};

} // namespace approximate_cholesky

#endif // APPROXIMATECHOLESKY_HPP
