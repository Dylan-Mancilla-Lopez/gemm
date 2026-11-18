// Cuerpo compartido por simple.cpp (-O2) y simple_o3.cpp (-O3): se incluye dos veces para
// medir el mismo código con los dos niveles. Con -O2 GCC 16 no vectoriza estos bucles; con
// -O3 sí (ver README).

// v0: el bucle interno recorre B por columna (salto de N doubles): casi cada acceso a B
// es una línea de caché nueva, y el producto punto no se vectoriza.
void naive_ijk(int M, int N, int K, const double* A, const double* B, double* C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

// v1: mismo cálculo, otro orden. El bucle interno recorre una fila de B y una fila de C,
// ambas contiguas: acceso secuencial y vectorizable (a es escalar dentro del bucle).
void loop_ikj(int M, int N, int K, const double* A, const double* B, double* C) {
  std::memset(C, 0, sizeof(double) * M * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      const double a = A[i * K + k];
      const double* b = B + k * N;
      double* c = C + i * N;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];
    }
}

// v2: ikj por bloques. Un tile TK×TJ de B se reutiliza para TI filas de A mientras sigue
// en caché, en vez de recorrer B entera por cada fila. Los seis bucles quedan juntos a
// propósito: extraer el tile a una función bajó tiled_O3 de ~7.0 a ~6.3 GFLOP/s (n = 1024).
void tiled(int M, int N, int K, const double* A, const double* B, double* C, int TI, int TK,
           int TJ) {
  std::memset(C, 0, sizeof(double) * M * N);
  for (int jj = 0; jj < N; jj += TJ) {
    const int je = std::min(jj + TJ, N);
    for (int kk = 0; kk < K; kk += TK) {
      const int ke = std::min(kk + TK, K);
      for (int ii = 0; ii < M; ii += TI) {
        const int ie = std::min(ii + TI, M);
        for (int i = ii; i < ie; ++i)
          for (int k = kk; k < ke; ++k) {
            const double a = A[i * K + k];
            const double* b = B + k * N;
            double* c = C + i * N;
            for (int j = jj; j < je; ++j) c[j] += a * b[j];
          }
      }
    }
  }
}

