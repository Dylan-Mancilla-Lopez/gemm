// Techos del roofline medidos en esta máquina (no calculados con la frecuencia nominal):
//   1) pico de FMA en doble precisión: 12 cadenas independientes de vfmadd231pd sobre ymm.
//      Latencia de FMA = 4 ciclos y 2 puertos => hacen falta >= 8 cadenas para no esperar.
//   2) ancho de banda a DRAM con una triada tipo STREAM: a[i] = b[i] + s*c[i]
//      (24 B por elemento, convención STREAM: no cuenta el write-allocate).
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double fma_gflops(int threads) {
  constexpr long kIters = 400'000'000;
  std::vector<double> best;
  for (int rep = 0; rep < 5; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    double sink = 0;
#pragma omp parallel num_threads(threads) reduction(+ : sink)
    {
      // Valores iniciales distintos: si todas las cadenas empiezan igual, GCC demuestra que
      // son idénticas y las fusiona en una sola (pasó en la primera versión de este archivo).
      __m256d a0 = _mm256_set1_pd(1.00), a1 = _mm256_set1_pd(1.01), a2 = _mm256_set1_pd(1.02),
              a3 = _mm256_set1_pd(1.03), a4 = _mm256_set1_pd(1.04), a5 = _mm256_set1_pd(1.05),
              a6 = _mm256_set1_pd(1.06), a7 = _mm256_set1_pd(1.07), a8 = _mm256_set1_pd(1.08),
              a9 = _mm256_set1_pd(1.09), a10 = _mm256_set1_pd(1.10), a11 = _mm256_set1_pd(1.11);
      const __m256d m = _mm256_set1_pd(0.9999999), c = _mm256_set1_pd(1e-7);
      for (long i = 0; i < kIters / 12; ++i) {
        a0 = _mm256_fmadd_pd(a0, m, c); a1 = _mm256_fmadd_pd(a1, m, c);
        a2 = _mm256_fmadd_pd(a2, m, c); a3 = _mm256_fmadd_pd(a3, m, c);
        a4 = _mm256_fmadd_pd(a4, m, c); a5 = _mm256_fmadd_pd(a5, m, c);
        a6 = _mm256_fmadd_pd(a6, m, c); a7 = _mm256_fmadd_pd(a7, m, c);
        a8 = _mm256_fmadd_pd(a8, m, c); a9 = _mm256_fmadd_pd(a9, m, c);
        a10 = _mm256_fmadd_pd(a10, m, c); a11 = _mm256_fmadd_pd(a11, m, c);
      }
      __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)),
                                _mm256_add_pd(_mm256_add_pd(a4, a5), _mm256_add_pd(a6, a7)));
      s = _mm256_add_pd(s, _mm256_add_pd(_mm256_add_pd(a8, a9), _mm256_add_pd(a10, a11)));
      alignas(32) double out[4];
      _mm256_store_pd(out, s);
      sink += out[0] + out[1] + out[2] + out[3];
    }
    auto t1 = std::chrono::steady_clock::now();
    if (sink == 42.0) std::puts("");  // evita que se elimine el cálculo
    double s = std::chrono::duration<double>(t1 - t0).count();
    best.push_back(double(kIters / 12) * 12 * 4 * 2 * threads / s / 1e9);
  }
  std::sort(best.begin(), best.end());
  return best[best.size() / 2];
}

static double triad_gbs(int threads) {
  const long n = 20'000'000;  // 3 arreglos × 160 MB: muy por encima de los 8 MiB de L3
  double* a = static_cast<double*>(std::aligned_alloc(64, n * 8));
  double* b = static_cast<double*>(std::aligned_alloc(64, n * 8));
  double* c = static_cast<double*>(std::aligned_alloc(64, n * 8));
#pragma omp parallel for num_threads(threads)
  for (long i = 0; i < n; ++i) { a[i] = 0; b[i] = 1; c[i] = 2; }  // first touch por hilo
  std::vector<double> r;
  for (int rep = 0; rep < 10; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for simd num_threads(threads)
    for (long i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];
    auto t1 = std::chrono::steady_clock::now();
    r.push_back(24.0 * n / std::chrono::duration<double>(t1 - t0).count() / 1e9);
  }
  std::printf("# check a[7]=%g\n", a[7]);
  std::free(a); std::free(b); std::free(c);
  std::sort(r.begin(), r.end());
  return r[r.size() / 2];
}

int main(int argc, char** argv) {
  const int t = argc > 1 ? std::atoi(argv[1]) : 1;
  std::printf("metric,threads,value\n");
  std::printf("fma_gflops,%d,%.2f\n", t, fma_gflops(t));
  std::printf("triad_gbs,%d,%.2f\n", t, triad_gbs(t));
}
