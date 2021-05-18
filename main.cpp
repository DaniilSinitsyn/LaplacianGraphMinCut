#include "graph.hpp"
#include <chrono>
#include <fstream>
#include <memory>
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage " << argv[0] << " number_of_threads\n";
    return 0;
  }
  int N = std::atoi(argv[1]);
#if 0
  int verts = 532661, edges = 1065322;
  std::ifstream file(
      "graph.txt");
#else
  int verts = 11537960, edges = 23075920;
  std::ifstream file("herzjesu/graph.txt");
#endif
  Eigen::initParallel();
  Graph g(verts);
  omp_set_num_threads(N);
  Eigen::setNbThreads(N);
  tbb::task_scheduler_init init(N);

  for (int i = 0; i < verts + edges; ++i) {
    char c;
    file >> c;
    if (c == 'n') {
      int n;
      double source, terminal;
      file >> n >> source >> terminal;
      g.AddNode(n, source / 1e+4, terminal / 1e+4);
    }
    if (c == 'e') {
      int n1, n2;
      double cap, revCap;
      file >> n1 >> n2 >> cap >> revCap;
      g.AddEdge(n1, n2, cap / 1e+4, revCap / 1e+4);
    }
  }
  std::cout << "Solution start" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  g.ComputeMaxFlow();
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << "elapsed time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(stop -
                                                                     start)
                   .count()
            << std::endl;
  return 0;
}
