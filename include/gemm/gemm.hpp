#pragma once
#include <cstddef>

// Todas las versiones calculan C = A * B con matrices row-major de doble precisión:
// A es M×K, B es K×N, C es M×N, sin leading dimensions extra.
namespace gemm {

void naive_ijk(int M, int N, int K, const double* A, const double* B, double* C);
void loop_ikj(int M, int N, int K, const double* A, const double* B, double* C);
// Bloqueo en caché sobre ikj. Los tamaños de tile se pasan para poder barrerlos.
void tiled(int M, int N, int K, const double* A, const double* B, double* C,
           int TI = 64, int TK = 256, int TJ = 512);
// Microkernel AVX2+FMA 6×8 en registros, leyendo A y B directamente (sin packing).
void simd_unpacked(int M, int N, int K, const double* A, const double* B, double* C);

// Estilo BLIS/Goto: packing de paneles de A y B + el mismo microkernel 6×8.
// Las mismas tres versiones escalares compiladas con -O3, donde GCC sí auto-vectoriza.
namespace o3 {
void naive_ijk(int M, int N, int K, const double* A, const double* B, double* C);
void loop_ikj(int M, int N, int K, const double* A, const double* B, double* C);
void tiled(int M, int N, int K, const double* A, const double* B, double* C,
           int TI = 64, int TK = 256, int TJ = 512);
}  // namespace o3

struct Blocking {
  int MC = 24;    // filas de A por bloque (A empacado: 24×256×8 B = 48 KiB → L2). Elegido
                  // por el barrido de results/sweep_packed.csv; 72 (valor de BLIS) fue ~6 % peor.
  int KC = 256;   // profundidad (micropanel de B: KC×8×8 B = 16 KiB → L1)
  int NC = 4080;  // columnas de B por bloque (B empacado ≈ 8 MiB → L3)
};
void packed(int M, int N, int K, const double* A, const double* B, double* C,
            Blocking blk = {}, int threads = 1);

}  // namespace gemm
