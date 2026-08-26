#include <cblas.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "gemm/gemm.hpp"

namespace {

using Mat = std::vector<double>;

Mat random_mat(int r, int c, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> d(-1, 1);
  Mat m(static_cast<std::size_t>(r) * c);
  for (auto& x : m) x = d(rng);
  return m;
}

Mat reference(int M, int N, int K, const Mat& A, const Mat& B) {
  Mat C(static_cast<std::size_t>(M) * N);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A.data(), K, B.data(), N,
              0.0, C.data(), N);
  return C;
}

// Error relativo por elemento con cota proporcional a K·eps: el orden de las sumas cambia
// entre versiones, así que no se espera igualdad bit a bit.
void expect_close(const Mat& got, const Mat& want, int K) {
  const double tol = 4.0 * K * 2.2e-16;
  double worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    double err = std::abs(got[i] - want[i]) / (1.0 + std::abs(want[i]));
    worst = std::max(worst, err);
  }
  EXPECT_LE(worst, tol) << "peor error relativo " << worst;
}

using Fn = void (*)(int, int, int, const double*, const double*, double*);
void packed1(int M, int N, int K, const double* A, const double* B, double* C) {
  gemm::packed(M, N, K, A, B, C, {}, 1);
}
void packed4(int M, int N, int K, const double* A, const double* B, double* C) {
  gemm::packed(M, N, K, A, B, C, {}, 4);
}
// Bloques diminutos para forzar muchos bordes y varias iteraciones de cada bucle externo.
void packed_tiny_blocks(int M, int N, int K, const double* A, const double* B, double* C) {
  gemm::packed(M, N, K, A, B, C, {.MC = 12, .KC = 7, .NC = 16}, 3);
}
void tiled_odd(int M, int N, int K, const double* A, const double* B, double* C) {
  gemm::tiled(M, N, K, A, B, C, 5, 7, 9);
}
void tiled_default(int M, int N, int K, const double* A, const double* B, double* C) {
  gemm::tiled(M, N, K, A, B, C);
}

}  // namespace

class GemmTest : public ::testing::TestWithParam<std::tuple<const char*, Fn>> {};

TEST_P(GemmTest, MatchesOpenBLAS) {
  Fn fn = std::get<1>(GetParam());
  // Tamaños: cuadrados, no múltiplos de 6 ni de 8, degenerados (1) y rectangulares.
  const int shapes[][3] = {{1, 1, 1},   {6, 8, 1},     {7, 9, 5},     {13, 17, 300},
                           {64, 64, 64}, {100, 37, 59}, {37, 100, 513}, {130, 130, 130},
                           {1, 200, 50}, {200, 1, 50}};
  unsigned seed = 1;
  for (const auto& [M, N, K] : shapes) {
    SCOPED_TRACE(::testing::Message() << M << "x" << N << "x" << K);
    Mat A = random_mat(M, K, seed++), B = random_mat(K, N, seed++);
    Mat C(static_cast<std::size_t>(M) * N, 12345.0);  // basura: la función debe sobrescribir
    fn(M, N, K, A.data(), B.data(), C.data());
    expect_close(C, reference(M, N, K, A, B), K);
  }
}

INSTANTIATE_TEST_SUITE_P(
    All, GemmTest,
    ::testing::Values(std::make_tuple("naive", &gemm::naive_ijk),
                      std::make_tuple("ikj", &gemm::loop_ikj),
                      std::make_tuple("tiled", &tiled_default),
                      std::make_tuple("tiled_odd", &tiled_odd),
                      std::make_tuple("ikj_o3", &gemm::o3::loop_ikj),
                      std::make_tuple("naive_o3", &gemm::o3::naive_ijk),
                      std::make_tuple("simd", &gemm::simd_unpacked),
                      std::make_tuple("packed1", &packed1),
                      std::make_tuple("packed4", &packed4),
                      std::make_tuple("packed_tiny", &packed_tiny_blocks)),
    [](const auto& info) { return std::string(std::get<0>(info.param)); });

TEST(Gemm, IdentityIsExact) {
  const int n = 50;
  Mat I(n * n, 0.0), B = random_mat(n, n, 9), C(n * n);
  for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;
  gemm::packed(n, n, n, I.data(), B.data(), C.data());
  EXPECT_EQ(C, B);
}
