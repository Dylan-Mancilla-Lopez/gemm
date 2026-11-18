#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "gemm/gemm.hpp"
#include "kernel.hpp"

namespace gemm {

using detail::kernel_6x8;
using detail::kernel_edge;
using detail::MR;
using detail::NR;

namespace {
struct Free { void operator()(double* p) const { std::free(p); } };
using Buf = std::unique_ptr<double[], Free>;
Buf alloc_buf(std::size_t n) {
  std::size_t bytes = (n * sizeof(double) + 63) / 64 * 64;
  return Buf(static_cast<double*>(std::aligned_alloc(64, bytes)));
}
}  // namespace

// B se lee por filas (bien), pero cada k lee A con salto de K doubles entre filas: 6 líneas
// de caché distintas por k. Para los bordes se copian los datos a tiles con ceros.
void simd_unpacked(int M, int N, int K, const double* A, const double* B, double* C) {
  std::memset(C, 0, sizeof(double) * M * N);
  constexpr int KC = 256;
  std::vector<double> ea(MR * KC), eb(KC * NR);
  for (int pc = 0; pc < K; pc += KC) {
    const int kc = std::min(KC, K - pc);
    for (int j = 0; j < N; j += NR) {
      const int nr = std::min(NR, N - j);
      for (int i = 0; i < M; i += MR) {
        const int mr = std::min(MR, M - i);
        const double* a = A + i * K + pc;
        const double* b = B + pc * N + j;
        double* c = C + i * N + j;
        if (mr == MR && nr == NR) {
          kernel_6x8(kc, a, K, 1, b, N, c, N);
        } else {
          std::fill(ea.begin(), ea.end(), 0.0);
          std::fill(eb.begin(), eb.end(), 0.0);
          for (int r = 0; r < mr; ++r)
            for (int k = 0; k < kc; ++k) ea[r * KC + k] = a[r * K + k];
          for (int k = 0; k < kc; ++k)
            for (int x = 0; x < nr; ++x) eb[k * NR + x] = b[k * N + x];
          kernel_edge(mr, nr, kc, ea.data(), KC, 1, eb.data(), NR, c, N);
        }
      }
    }
  }
}

namespace {

/// Empaca A[mc×kc] en micropaneles de MR filas. Dentro de cada micropanel, para cada k
/// van los MR valores de esa columna seguidos: el kernel lee A de forma 100 % secuencial.
/// Las filas que faltan en el último micropanel se rellenan con ceros.
void pack_A(int mc, int kc, const double* A, int lda, double* Ap) {
  for (int i = 0; i < mc; i += MR) {
    const int mr = std::min(MR, mc - i);
    for (int k = 0; k < kc; ++k) {
      for (int r = 0; r < mr; ++r) Ap[r] = A[(i + r) * lda + k];
      for (int r = mr; r < MR; ++r) Ap[r] = 0.0;
      Ap += MR;
    }
  }
}

/// Empaca B[kc×nc] en micropaneles de NR columnas: para cada k, NR valores contiguos.
void pack_B(int kc, int nc, const double* B, int ldb, double* Bp) {
  for (int j = 0; j < nc; j += NR) {
    const int nr = std::min(NR, nc - j);
    for (int k = 0; k < kc; ++k) {
      const double* src = B + k * ldb + j;
      if (nr == NR) {
        std::memcpy(Bp, src, NR * sizeof(double));
      } else {
        for (int x = 0; x < nr; ++x) Bp[x] = src[x];
        for (int x = nr; x < NR; ++x) Bp[x] = 0.0;
      }
      Bp += NR;
    }
  }
}

}  // namespace

// Los cinco bucles de BLIS: jc (NC) → pc (KC, empaca B) → ic (MC, empaca A) → jr (NR)
// → ir (MR) → microkernel. Con threads > 1 se reparte el bucle ic con OpenMP: B empacado se
// comparte (solo lectura) y cada hilo empaca su propio bloque de A.
//
// El anidamiento es deliberado: es el hot path medido, y extraer jr/ir e ic a funciones
// separadas costó ~6 % de GFLOP/s (n = 1024, 1 hilo, A/B en la misma sesión).
void packed(int M, int N, int K, const double* A, const double* B, double* C, Blocking blk,
            int threads) {
  std::memset(C, 0, sizeof(double) * M * N);
  const int MC = blk.MC - blk.MC % MR, KC = blk.KC, NC = blk.NC - blk.NC % NR;
  Buf Bp = alloc_buf(static_cast<std::size_t>(KC) * (NC + NR));
  const int nthreads = threads < 1 ? 1 : threads;
  std::vector<Buf> Ap(nthreads);
  for (auto& p : Ap) p = alloc_buf(static_cast<std::size_t>(MC + MR) * KC);

  for (int jc = 0; jc < N; jc += NC) {
    const int nc = std::min(NC, N - jc);
    for (int pc = 0; pc < K; pc += KC) {
      const int kc = std::min(KC, K - pc);
      pack_B(kc, nc, B + pc * N + jc, N, Bp.get());
#pragma omp parallel for schedule(dynamic) num_threads(nthreads) if (nthreads > 1)
      for (int ic = 0; ic < M; ic += MC) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double* ap = Ap[tid].get();
        const int mc = std::min(MC, M - ic);
        pack_A(mc, kc, A + ic * K + pc, K, ap);
        for (int jr = 0; jr < nc; jr += NR) {
          const int nr = std::min(NR, nc - jr);
          const double* bp = Bp.get() + jr * kc;
          for (int ir = 0; ir < mc; ir += MR) {
            const int mr = std::min(MR, mc - ir);
            const double* a = ap + ir * kc;
            double* c = C + (ic + ir) * N + jc + jr;
            if (mr == MR && nr == NR) kernel_6x8(kc, a, 1, MR, bp, NR, c, N);
            else kernel_edge(mr, nr, kc, a, 1, MR, bp, NR, c, N);
          }
        }
      }
    }
  }
}

}  // namespace gemm
