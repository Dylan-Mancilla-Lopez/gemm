// Mide GFLOP/s de cada versión. Salida CSV: version,n,threads,gflops_median,gflops_min,gflops_max,reps
//   gemm_bench sizes [hilos [versión]]   tamaños cuadrados; filtros opcionales.
//   OpenBLAS y nuestras versiones OpenMP deben medirse en procesos separados: los hilos de un
//   runtime se quedan en espera activa tras cada llamada y le roban CPU al otro.
//   gemm_bench sweep_tiled      barrido de tiles de la versión tiled (n = 1024)
//   gemm_bench sweep_packed     barrido de MC/KC de la versión packed (n = 1024)
#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "gemm/gemm.hpp"

extern "C" void openblas_set_num_threads(int);

namespace {

using Fn = std::function<void(int, const double*, const double*, double*)>;

int g_only_threads = 0;  // 0 = todas; si no, solo las corridas con ese número de hilos
std::string g_only_version;  // vacío = todas; si no, solo esa versión

void run(const std::string& name, int n, int threads, const Fn& fn, int reps) {
  if (g_only_threads != 0 && threads != g_only_threads) return;
  if (!g_only_version.empty() && name != g_only_version) return;
  std::vector<double> A(std::size_t(n) * n), B(A.size()), C(A.size());
  std::mt19937 rng(n);
  std::uniform_real_distribution<double> d(-1, 1);
  for (auto& x : A) x = d(rng);
  for (auto& x : B) x = d(rng);
  fn(n, A.data(), B.data(), C.data());  // warmup: páginas tocadas, caché y frecuencia
  std::vector<double> g;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    fn(n, A.data(), B.data(), C.data());
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    g.push_back(2.0 * n * n * double(n) / s / 1e9);
  }
  std::sort(g.begin(), g.end());
  std::printf("%s,%d,%d,%.2f,%.2f,%.2f,%d\n", name.c_str(), n, threads, g[g.size() / 2], g.front(),
              g.back(), reps);
  std::fflush(stdout);
}

int reps_for(int n) { return n <= 512 ? 15 : n <= 1024 ? 7 : 5; }

void sizes() {
  for (int n : {128, 256, 384, 512, 768, 1024, 1536, 2048}) {
    const int r = reps_for(n);
    if (n <= 1024) {
      run("naive_ijk", n, 1, [](int n, auto A, auto B, auto C) { gemm::naive_ijk(n, n, n, A, B, C); }, n <= 512 ? r : 3);
    }
    run("loop_ikj", n, 1, [](int n, auto A, auto B, auto C) { gemm::loop_ikj(n, n, n, A, B, C); }, r);
    run("tiled", n, 1, [](int n, auto A, auto B, auto C) { gemm::tiled(n, n, n, A, B, C); }, r);
    run("loop_ikj_O3", n, 1, [](int n, auto A, auto B, auto C) { gemm::o3::loop_ikj(n, n, n, A, B, C); }, r);
    run("tiled_O3", n, 1, [](int n, auto A, auto B, auto C) { gemm::o3::tiled(n, n, n, A, B, C); }, r);
    run("simd_unpacked", n, 1, [](int n, auto A, auto B, auto C) { gemm::simd_unpacked(n, n, n, A, B, C); }, r);
    run("packed", n, 1, [](int n, auto A, auto B, auto C) { gemm::packed(n, n, n, A, B, C, {}, 1); }, r);
    for (int t : {4, 8})
      run("packed", n, t, [t](int n, auto A, auto B, auto C) { gemm::packed(n, n, n, A, B, C, {}, t); }, r);
    for (int t : {1, 4, 8}) {
      openblas_set_num_threads(t);
      run("openblas", n, t, [](int n, auto A, auto B, auto C) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
      }, r);
    }
  }
}

// Se barre la versión -O3: con -O2 el bucle interno es escalar y el tile casi no importa.
void sweep_tiled() {
  for (int ti : {16, 64, 128})
    for (int tk : {64, 128, 256, 512})
      for (int tj : {128, 256, 512, 1024})
        run("tiled_O3_TI" + std::to_string(ti) + "_TK" + std::to_string(tk) + "_TJ" + std::to_string(tj),
            2048, 1, [=](int n, auto A, auto B, auto C) { gemm::o3::tiled(n, n, n, A, B, C, ti, tk, tj); }, 3);
}

void sweep_packed() {
  for (int mc : {24, 48, 72, 96, 144, 288})
    for (int kc : {64, 128, 192, 256, 384, 512})
      run("packed_MC" + std::to_string(mc) + "_KC" + std::to_string(kc), 1024, 1,
          [=](int n, auto A, auto B, auto C) { gemm::packed(n, n, n, A, B, C, {.MC = mc, .KC = kc}, 1); }, 5);
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "sizes";
  if (argc > 2) g_only_threads = std::atoi(argv[2]);
  if (argc > 3) g_only_version = argv[3];
  std::printf("version,n,threads,gflops_median,gflops_min,gflops_max,reps\n");
  if (mode == "sizes") sizes();
  else if (mode == "sweep_tiled") sweep_tiled();
  else if (mode == "sweep_packed") sweep_packed();
  else return 2;
}
