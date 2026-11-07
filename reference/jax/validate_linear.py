"""Validation of the JAX reference solver against the linear GPU solver.
1. Patch test: uniform solid -> exact isotropic C and the analytic neo-Hookean stress at finite strain.
2. Small-strain C^H vs homo3d on random anisotropic fields (8^3, 16^3); bar: max relative error < 1e-3
   (relative to the largest entry, entry-wise over the 6x6 matrix).
Usage (reference env): python -I validate_linear.py <results dir> [<dir> ...]
Each results dir holds p3_rho_seed<k>(.npy) and p3_C_seed<k>.txt from the evalfield mode of the GPU driver
(HOMO_CUSTOM=p1 P1_MODE=evalfield).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finite_strain as O  # noqa: E402

VDB2NPY = Path(os.environ.get("VDB2NPY", Path(__file__).resolve().parents[2] / "build" / "vdb2npy"))


def patch_test():
    lam, mu = O.lame()
    C, _ = O.C_linear(np.ones((4, 4, 4)))
    Cex = np.zeros((6, 6))
    Cex[:3, :3] = lam
    Cex[np.arange(3), np.arange(3)] = lam + 2 * mu
    Cex[np.arange(3, 6), np.arange(3, 6)] = mu
    Cex *= 1 + O.RHO_MIN
    e_lin = np.abs(C - Cex).max() / np.abs(Cex).max()
    # neo-Hookean, uniform solid, finite stretch: P = mu (F - F^-T) + lam ln J F^-T
    cell = O.Cell(np.ones((4, 4, 4)), material="neohooke")
    F = np.array([[1.3, 0.2, 0.0], [0.0, 0.9, 0.1], [0.05, 0.0, 1.1]])
    w, log = cell.solve(F)
    P = np.asarray(cell.mean_stress(w, F))
    Fit = np.linalg.inv(F).T
    Pex = (mu * (F - Fit) + lam * np.log(np.linalg.det(F)) * Fit) * (1 + O.RHO_MIN)
    e_nh = np.abs(P - Pex).max() / np.abs(Pex).max()
    print(f"[patch] linear C error {e_lin:.2e}; neo-Hookean P error {e_nh:.2e} (newton {log['newton']}, |w| {float(np.abs(w).max()):.1e})")
    return e_lin < 1e-10 and e_nh < 1e-10


def load_rho(d, seed):
    vdb = d / f"p3_rho_seed{seed:03d}"
    npy = d / f"p3_rho_seed{seed:03d}.npy"
    if not npy.exists():
        subprocess.run([str(VDB2NPY), str(vdb), str(npy)], check=True, capture_output=True)
    return np.load(npy).astype(np.float64)


def compare(d):
    files = sorted(d.glob("p3_C_seed*.txt"))
    if not files:
        print(f"[homo3d] {d}: no reference files")
        return False
    ok = True
    for f in files:
        seed = int(f.stem.split("seed")[1])
        Ch = np.loadtxt(f)
        rho = load_rho(d, seed)
        best = None
        for name, r in (("as stored", rho), ("transposed", rho.transpose(2, 1, 0))):
            t0 = time.time()
            C, logs = O.C_linear(r)
            dt = time.time() - t0
            err = np.abs(C - Ch).max() / np.abs(Ch).max()
            if best is None or err < best[1]:
                best = (name, err, C, dt, sum(l["cg"] for l in logs))
        name, err, C, dt, cg = best
        ok &= err < 1e-3
        print(f"[homo3d] {d.name} seed {seed}: n={rho.shape[0]}  axis order '{name}'  max rel error {err:.2e}  "
              f"(C11 {C[0,0]:.5f} vs {Ch[0,0]:.5f}; C44 {C[3,3]:.5f} vs {Ch[3,3]:.5f})  {dt:.1f}s, {cg} CG its")
    return ok


if __name__ == "__main__":
    import jax
    print("devices:", jax.devices())
    p = patch_test()
    c = all(compare(Path(a)) for a in sys.argv[1:])
    print("PASS" if p and c else "FAIL", "(patch test", "ok" if p else "failed", "; homo3d < 1e-3", "ok" if c else "failed", ")")
