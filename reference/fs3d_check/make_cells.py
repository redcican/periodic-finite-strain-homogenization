"""Analytic test cells for the fs3d checks: physical density fields of shape (n, n, n), axis order (x, y, z).
Usage: python -I make_cells.py <out dir> [n ...]   (default 16 32)
Writes rod<n>.npy (cubic rod lattice, rods of radius 0.18 along the three axes through the cell centre, with a
smooth boundary) and schwarzp<n>.npy (Schwarz P surface thresholded to a shell, smoothed).
"""
import sys
from pathlib import Path

import numpy as np


def rods(n, r=0.18, width=1.0):
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    def d2(a, b):
        return np.minimum(np.abs(a - 0.5), 1 - np.abs(a - 0.5)) ** 2 + np.minimum(np.abs(b - 0.5), 1 - np.abs(b - 0.5)) ** 2
    dist = np.sqrt(np.minimum(np.minimum(d2(Y, Z), d2(X, Z)), d2(X, Y)))
    # smooth step over one voxel so the field has intermediate densities (tests the interpolation)
    rho = 0.5 - 0.5 * np.tanh((dist - r) * n / width * 2.0)
    return np.clip(rho, 1e-3, 1.0)


def schwarzp(n, t=0.45):
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    f = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)
    rho = 0.5 - 0.5 * np.tanh((np.abs(f) - t) * n / 2.0)
    return np.clip(rho, 1e-3, 1.0)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ns = [int(a) for a in sys.argv[2:]] or [16, 32]
    for n in ns:
        for name, f in (("rod", rods), ("schwarzp", schwarzp)):
            rho = f(n).astype(np.float64)
            np.save(out / f"{name}{n}.npy", rho)
            print(f"{name}{n}: volume {rho.mean():.4f}, grey fraction {np.mean((rho > 0.05) & (rho < 0.95)):.3f}")
