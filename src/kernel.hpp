#pragma once
#include <immintrin.h>

namespace gemm::detail {

inline constexpr int MR = 6;
inline constexpr int NR = 8;

/// Acumula C += A·B para un tile de 6×8 con stride ldc.
/// a(r, k) = a[r * rs_a + k * cs_a]; b(k, j) = b[k * rs_b + j] (fila de B contigua).
///
/// 12 acumuladores ymm (6 filas × 2 vectores de 4 doubles) + 2 registros para la fila de B
/// + 1 para el broadcast de A = 15 de los 16 ymm de AVX2. Por cada k: 2 loads de B,
/// 6 broadcasts de A y 12 FMAs (96 FLOP). Está desenrollado a mano para que los 12
/// acumuladores se queden en registros.
inline void kernel_6x8(int kc, const double* a, long rs_a, long cs_a, const double* b, long rs_b,
                       double* c, long ldc) {
  __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
  __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
  __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
  __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
  __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
  __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();
  for (int k = 0; k < kc; ++k) {
    const __m256d b0 = _mm256_loadu_pd(b + k * rs_b);
    const __m256d b1 = _mm256_loadu_pd(b + k * rs_b + 4);
    const double* ak = a + k * cs_a;
    __m256d x;
    x = _mm256_broadcast_sd(ak + 0 * rs_a); c00 = _mm256_fmadd_pd(x, b0, c00); c01 = _mm256_fmadd_pd(x, b1, c01);
    x = _mm256_broadcast_sd(ak + 1 * rs_a); c10 = _mm256_fmadd_pd(x, b0, c10); c11 = _mm256_fmadd_pd(x, b1, c11);
    x = _mm256_broadcast_sd(ak + 2 * rs_a); c20 = _mm256_fmadd_pd(x, b0, c20); c21 = _mm256_fmadd_pd(x, b1, c21);
    x = _mm256_broadcast_sd(ak + 3 * rs_a); c30 = _mm256_fmadd_pd(x, b0, c30); c31 = _mm256_fmadd_pd(x, b1, c31);
    x = _mm256_broadcast_sd(ak + 4 * rs_a); c40 = _mm256_fmadd_pd(x, b0, c40); c41 = _mm256_fmadd_pd(x, b1, c41);
    x = _mm256_broadcast_sd(ak + 5 * rs_a); c50 = _mm256_fmadd_pd(x, b0, c50); c51 = _mm256_fmadd_pd(x, b1, c51);
  }
  auto add_to_row = [&](int r, __m256d lo, __m256d hi) {
    double* row = c + r * ldc;
    _mm256_storeu_pd(row, _mm256_add_pd(_mm256_loadu_pd(row), lo));
    _mm256_storeu_pd(row + 4, _mm256_add_pd(_mm256_loadu_pd(row + 4), hi));
  };
  add_to_row(0, c00, c01); add_to_row(1, c10, c11); add_to_row(2, c20, c21);
  add_to_row(3, c30, c31); add_to_row(4, c40, c41); add_to_row(5, c50, c51);
}

/// Tile de borde (mr < 6 o nr < 8): el kernel escribe en un tile temporal de 6×8 y luego
/// se suma solo la parte válida. `a` y `b` deben tener datos legibles (o ceros) en el tile
/// completo.
inline void kernel_edge(int mr, int nr, int kc, const double* a, long rs_a, long cs_a,
                        const double* b, long rs_b, double* c, long ldc) {
  alignas(32) double tmp[MR * NR] = {};
  kernel_6x8(kc, a, rs_a, cs_a, b, rs_b, tmp, NR);
  for (int r = 0; r < mr; ++r)
    for (int j = 0; j < nr; ++j) c[r * ldc + j] += tmp[r * NR + j];
}

}  // namespace gemm::detail
