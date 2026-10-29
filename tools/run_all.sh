#!/usr/bin/env bash
# Corre todas las mediciones de P3. Asignación de CPUs:
#   1 hilo  -> CPU 0
#   4 hilos -> CPUs 0-3 (un hilo por núcleo físico; sus hermanas SMT son 4-7)
#   8 hilos -> CPUs 0-7 (SMT)
# Cada librería multihilo corre en su propio proceso (ver gemm_bench.cpp).
set -euo pipefail
B=${1:-build}
R=${2:-results}
taskset -c 0 "$B/peak" 1 > "$R/peak_1t.csv"
taskset -c 0-3 "$B/peak" 4 > "$R/peak_4t.csv"
taskset -c 0 "$B/gemm_bench" sizes 1 > "$R/sizes_1t.csv"
for t in 4 8; do
  cpus=$([ $t = 4 ] && echo 0-3 || echo 0-7)
  { taskset -c $cpus "$B/gemm_bench" sizes $t packed
    taskset -c $cpus "$B/gemm_bench" sizes $t openblas | tail -n +2; } > "$R/sizes_${t}t.csv"
done
taskset -c 0 "$B/gemm_bench" sweep_packed > "$R/sweep_packed.csv"
taskset -c 0 "$B/gemm_bench" sweep_tiled > "$R/sweep_tiled.csv"
