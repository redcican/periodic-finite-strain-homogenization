"""Cases of the DOLFINx cross-check: density fields and load paths, shared by both sides.

Fields (written once by `python -I cases.py <out dir> <Zheng et al. dataset data dir>` in the reference env, read as
.npy by both sides):
  R8   random density U(0.05, 0.95) per voxel, 8^3, seed 0; plain neo-Hookean
  R16  same at 16^3, seed 1
  T16  Zheng lattice 595492 voxelized at 16^3, volume 0.2 (gray boundary voxels, void 0); WLS void
       interpolation (beta 500, rho0 0.01)
Loads, all from F = I in equal steps, warm-started:
  tension     Fbar = diag(1.10, 1, 1), 5 steps (uniaxial strain)
  compression Fbar = diag(0.90, 1, 1), 5 steps
  shear       Fbar = I + 0.10 e_x (x) e_y, 5 steps
  ustress     F_11 = 0.95, F_22 and F_33 free with P_22 = P_33 = 0 (uniaxial stress, mixed control), 5 steps
Arrays: rho[i, j, k] with (i, j, k) along (x, y, z), as in the JAX reference solver.
"""
import json
import sys
from pathlib import Path

import numpy as np

E, NU = 10.0, 0.3
WLS = (500.0, 0.01)
FIELDS = {"R8": dict(n=8, wls=None), "R16": dict(n=16, wls=None), "T16": dict(n=16, wls=WLS)}
STEPS = 5


def loads():
    I3 = np.eye(3)
    sh = I3.copy()
    sh[0, 1] = 0.10
    return {"tension": dict(F=np.diag([1.10, 1, 1]), free=None), "compression": dict(F=np.diag([0.90, 1, 1]), free=None),
            "shear": dict(F=sh, free=None), "ustress": dict(F=np.diag([0.95, 1, 1]), free=[(1, 1), (2, 2)])}


def path(F):
    """Prescribed Fbar at each step (linear from I)."""
    return [np.eye(3) + (s / STEPS) * (np.asarray(F) - np.eye(3)) for s in range(1, STEPS + 1)]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jax"))
    import voxelize_truss as V

    np.save(out / "R8.npy", np.random.default_rng(0).uniform(0.05, 0.95, (8, 8, 8)))
    np.save(out / "R16.npy", np.random.default_rng(1).uniform(0.05, 0.95, (16, 16, 16)))
    src = Path(sys.argv[2])
    pos, edges = V.load_truss(src, 595492)
    rho, r, _ = V.voxelize(pos, edges, 16, 4, 0.2)
    np.save(out / "T16.npy", rho)
    json.dump(dict(T16=dict(lattice=595492, vol=float(rho.mean()), r=float(r), solid=float((rho > 0.99).mean()),
                            void=float((rho < 0.01).mean()))), open(out / "fields.json", "w"), indent=1)
    print("fields written; T16 volume", rho.mean(), "void fraction", (rho < 0.01).mean())
