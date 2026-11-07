"""Voxelize trusses of the Zheng et al. (2023) library for the finite-strain reference solver.

A truss is a 27-node graph in the octant [0,1]^3 (node-position.npz, adjacency-matrix.npz). The full cell [-1,1]^3
is its 8 mirror images; the lattice has period 2. Struts are capsules of one common radius r. The cell is mapped to
the unit cell of the reference solver by u = (x + 1) / 2 on an n^3 grid; each voxel's density is the fraction of ss^3 sub-samples
inside the struts (with a linear ramp one sub-sample wide at the surface). r is set by bisection so the mean density (solid volume fraction) is --vol (0.15, as in the
dataset). The density is the physical density (stiffness scale), not a SIMP design variable.

Usage (reference env): python -I voxelize_truss.py <dataset data dir> <out dir> <truss index> [...] [--n 48] [--ss 4] [--vol 0.15]
Writes <out dir>/truss_<index>/rho.npy and meta.json (r, n, volume, edges).
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_truss(src, k):
    pos = sp.load_npz(src / "node-position.npz").tocsr()[27 * k: 27 * k + 27].toarray()
    A = sp.load_npz(src / "adjacency-matrix.npz").tocsr()[27 * k: 27 * k + 27].tocoo()
    edges = sorted({(min(i, j), max(i, j)) for i, j in zip(A.row, A.col) if i != j})
    return pos, edges


def cell_segments(pos, edges, rmax):
    """Segments of the full periodic cell [-1,1]^3 (mirror images) plus periodic images within rmax of it."""
    seg = np.array([[pos[i], pos[j]] for i, j in edges])  # (E,2,3) in the octant
    out = []
    for f in range(8):
        sgn = np.array([-1 if f >> a & 1 else 1 for a in range(3)])
        out.append(seg * sgn)
    seg = np.concatenate(out)
    imgs = []
    for t in np.array(np.meshgrid([-2, 0, 2], [-2, 0, 2], [-2, 0, 2])).reshape(3, -1).T:
        s = seg + t
        lo, hi = s.min(axis=1), s.max(axis=1)
        keep = np.all((hi >= -1 - rmax) & (lo <= 1 + rmax), axis=1)
        imgs.append(s[keep])
    seg = np.concatenate(imgs)
    # drop duplicates (mirror images of struts lying in a mirror plane coincide); endpoints in lexicographic order
    s = seg.round(9)
    first_is_smaller = [tuple(a) <= tuple(b) for a, b in s]
    key = np.array([np.concatenate([a, b]) if f else np.concatenate([b, a]) for (a, b), f in zip(s, first_is_smaller)])
    _, u = np.unique(key, axis=0, return_index=True)
    return seg[np.sort(u)]


def distance_field(seg, n, ss, rcap=0.3):
    """Distance from every sub-sample point to the nearest segment, shape (n, n, n, ss^3), in cell units x.
    Only points within rcap of a segment's bounding box are evaluated; farther points get rcap (radii stay below)."""
    m = n * ss
    g = (np.arange(m) + 0.5) / m * 2 - 1  # sub-sample centres in x
    dmin = np.full((m, m, m), rcap)
    for a, b in seg:
        lo = np.minimum(a, b) - rcap
        hi = np.maximum(a, b) + rcap
        i0 = np.clip(np.searchsorted(g, lo), 0, m)
        i1 = np.clip(np.searchsorted(g, hi, side="right"), 0, m)
        if np.any(i1 <= i0):
            continue
        X, Y, Z = np.meshgrid(g[i0[0]:i1[0]], g[i0[1]:i1[1]], g[i0[2]:i1[2]], indexing="ij")
        P = np.stack([X, Y, Z], -1)
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-30), 0, 1)
        d = np.linalg.norm(P - a - t[..., None] * ab, axis=-1)
        sub = dmin[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        np.minimum(sub, d, out=sub)
    return dmin.reshape(n, ss, n, ss, n, ss).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, ss**3)


def voxelize(pos, edges, n=48, ss=4, vol=0.15):
    seg = cell_segments(pos, edges, rmax=0.5)
    d = distance_field(seg, n, ss)
    # covered fraction with a linear ramp of one sub-sample spacing, so the volume is continuous in r
    # (axis-aligned struts put many sub-samples at exactly the same distance)
    delta = 2.0 / (n * ss)

    def frac(r):
        return np.clip(0.5 + (r - d) / delta, 0.0, 1.0).mean(axis=-1)

    lo, hi = 0.0, 0.3
    for _ in range(50):
        r = 0.5 * (lo + hi)
        if frac(r).mean() < vol:
            lo = r
        else:
            hi = r
    r = 0.5 * (lo + hi)
    return frac(r), r, len(seg)


if __name__ == "__main__":
    opt, args, it = {}, [], iter(sys.argv[1:])
    for a in it:
        if a.startswith("--"):
            opt[a] = next(it)
        else:
            args.append(a)
    src, out = Path(args[0]), Path(args[1])
    n, ss, vol = int(opt.get("--n", 48)), int(opt.get("--ss", 4)), float(opt.get("--vol", 0.15))
    for k in map(int, args[2:]):
        pos, edges = load_truss(src, k)
        rho, r, nseg = voxelize(pos, edges, n, ss, vol)
        d = out / f"truss_{k}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "rho.npy", rho)
        h = 2.0 / n
        meta = dict(index=k, n=n, ss=ss, vol=float(rho.mean()), r=float(r), r_voxels=float(r / h), edges=[[int(i), int(j)] for i, j in edges],
                    segments_in_cell=int(nseg), solid_voxels=float((rho > 0.5).mean()))
        json.dump(meta, open(d / "meta.json", "w"), indent=1)
        print(f"truss {k}: {len(edges)} struts, r = {r:.4f} ({r / h:.2f} voxels at n = {n}), volume {rho.mean():.4f}, "
              f"voxels > 0.5: {(rho > 0.5).mean():.4f}", flush=True)
