#!/usr/bin/env python3
"""Genera las gráficas de results/ a partir de los CSV de tools/run_all.sh."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def rows(name):
    with open(R / name) as f:
        return [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]


fma1 = float(next(r for r in rows("peak_1t.csv") if r["metric"] == "fma_gflops")["value"])
fma4 = float(next(r for r in rows("peak_4t.csv") if r["metric"] == "fma_gflops")["value"])

# 1) Escalera de optimizaciones, un hilo
order = ["naive_ijk", "loop_ikj", "tiled", "loop_ikj_O3", "tiled_O3", "simd_unpacked", "packed", "openblas"]
labels = {
    "naive_ijk": "v0 ijk ingenuo",
    "loop_ikj": "v1 ikj (-O2, escalar)",
    "tiled": "v2 tiled (-O2, escalar)",
    "loop_ikj_O3": "v1 ikj (-O3, vectorizado)",
    "tiled_O3": "v2 tiled (-O3, vectorizado)",
    "simd_unpacked": "v3 microkernel 6×8 sin packing",
    "packed": "v4 packing + microkernel",
    "openblas": "OpenBLAS 0.3.34",
}
s1 = rows("sizes_1t.csv")
fig, ax = plt.subplots(figsize=(11, 5.5))
for v in order:
    pts = sorted((int(r["n"]), float(r["gflops_median"])) for r in s1 if r["version"] == v)
    if pts:
        ax.plot(*zip(*pts), marker="o", label=labels[v],
                linestyle="--" if v == "openblas" else "-")
ax.axhline(fma1, color="black", linewidth=1, linestyle=":")
ax.text(130, fma1 * 1.08, f"pico FMA medido, 1 hilo: {fma1:.1f} GFLOP/s", fontsize=8)
ax.set_yscale("log")
ax.set_xlabel("n (matrices n×n, double)")
ax.set_ylabel("GFLOP/s (mediana, escala log)")
ax.set_title("DGEMM en i7-8550U, 1 hilo fijado a CPU 0")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout()
fig.savefig(R / "gflops_1thread.png", dpi=130, bbox_inches="tight")

# 2) Multihilo
fig, ax = plt.subplots(figsize=(9, 5))
for name, t in [("sizes_1t.csv", 1), ("sizes_4t.csv", 4), ("sizes_8t.csv", 8)]:
    for v, ls in [("packed", "-"), ("openblas", "--")]:
        pts = sorted((int(r["n"]), float(r["gflops_median"])) for r in rows(name) if r["version"] == v)
        ax.plot(*zip(*pts), marker="o", linestyle=ls, label=f"{v}, {t} hilo(s)")
ax.axhline(fma4, color="black", linewidth=1, linestyle=":")
ax.text(130, fma4 * 1.02, f"pico FMA medido, 4 hilos: {fma4:.0f} GFLOP/s", fontsize=8)
ax.set_xlabel("n")
ax.set_ylabel("GFLOP/s (mediana)")
ax.set_title("v5: packing + OpenMP vs OpenBLAS")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / "gflops_threads.png", dpi=130)

# 3) Barrido de bloques de la versión packed
sw = rows("sweep_packed.csv")
mcs = sorted({int(r["version"].split("_MC")[1].split("_")[0]) for r in sw})
kcs = sorted({int(r["version"].split("_KC")[1]) for r in sw})
grid = [[0.0] * len(kcs) for _ in mcs]
for r in sw:
    mc = int(r["version"].split("_MC")[1].split("_")[0])
    kc = int(r["version"].split("_KC")[1])
    grid[mcs.index(mc)][kcs.index(kc)] = float(r["gflops_median"])
fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(grid, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(kcs)), kcs)
ax.set_yticks(range(len(mcs)), mcs)
ax.set_xlabel("KC")
ax.set_ylabel("MC")
for i in range(len(mcs)):
    for j in range(len(kcs)):
        ax.text(j, i, f"{grid[i][j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if grid[i][j] < (max(map(max, grid)) + min(map(min, grid))) / 2 else "black")
fig.colorbar(im, label="GFLOP/s")
ax.set_title("packed, n=1024, 1 hilo")
fig.tight_layout()
fig.savefig(R / "sweep_packed.png", dpi=130)
print("ok")
