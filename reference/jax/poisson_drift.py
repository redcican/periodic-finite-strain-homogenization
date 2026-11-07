"""Poisson's ratio of a cell along uniaxial-stress paths at finite strain (tension and compression).

For a homo3d result dir (rho VDB from -obj npr), physical stiffness 1e-9 + rho^3 as in homo3d's runInstance.
  nu0     : linear, uniaxial stress along x, from the compliance S = C^-1: nu0 = -S21 / S11
  nu(e)   : secant, -(Fbar_22 - 1) / (Fbar_11 - 1), neo-Hookean with void interpolation, uniaxial stress
            (Fbar_11 = 1 + e prescribed, Pbar_22 = Pbar_33 = 0, shears fixed at 0), e = +-0.01 .. +-0.20
Drift metrics: max |nu(e) - nu0| over the converged path, and the first |e| with |nu(e) - nu0| > 0.05.
Usage (reference env): python -I poisson_drift.py <homo3d result dir> [--emax 0.2] [--de 0.01] [--paths tension,compression] [--tile 2] [--perturb 1e-3] [--phys] [--fast] [--tag x]
--fast: stored Gauss-point tangent and Eisenstat-Walker CG tolerances (same solution, about 20x faster).
Writes <dir>/drift{_paths}{_tileN}{_tag}.csv and prints a summary.
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finite_strain as O  # noqa: E402

VDB2NPY = Path(os.environ.get("VDB2NPY", Path(__file__).resolve().parents[2] / "build" / "vdb2npy"))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict(zip(sys.argv[1:], sys.argv[2:]))
emax = float(opt.get("--emax", 0.2))
de = float(opt.get("--de", 0.01))
d = Path(args[0])

npy = d / "rho.npy"
if not npy.exists():
    subprocess.run([str(VDB2NPY), str(d / "rho"), str(npy)], check=True, capture_output=True)
rho = np.load(npy).astype(np.float64)
phys = "--phys" in sys.argv  # rho is already the physical density (e.g. voxelized trusses); default: homo3d rho^3
rho_phys = np.clip(rho, 0, 1) if phys else np.clip(rho, 0, 1) ** 3
tile = int(opt.get("--tile", 1))  # N x N x N copies of the cell: admits buckling modes with N times the period
perturb = float(opt.get("--perturb", 0))  # random kick to the warm start (times the |w| scale) to leave symmetric paths
if tile > 1:
    rho = np.tile(rho, (tile,) * 3)
    rho_phys = np.tile(rho_phys, (tile,) * 3)
n = rho.shape[0]
print(f"{d.name}: n={n}, volume {rho.mean():.3f} (design), solid fraction (rho>0.5) {np.mean(rho > 0.5):.3f}")

t0 = time.time()
C, _ = O.C_linear(rho_phys)
S = np.linalg.inv(C)
nu0 = -S[1, 0] / S[0, 0]
nu0_z = -S[2, 0] / S[0, 0]
print(f"linear: C11 {C[0,0]:.5f} C12 {C[0,1]:.5f} C44 {C[3,3]:.5f}; nu0 (x->y) {nu0:+.4f}, (x->z) {nu0_z:+.4f}  [{time.time()-t0:.1f}s]")

cell = O.Cell(rho_phys, material="neohooke", wls=(500.0, 0.01))
fast = "--fast" in sys.argv
mix = O.Mixed(cell, [(1, 1), (2, 2)], **(dict(tangent="stored", forcing="ew") if fast else {}))
rows = []
paths = [(1, "tension"), (-1, "compression")]
if "--paths" in opt:
    paths = [p for p in paths if p[1] in opt["--paths"].split(",")]
def solve_step(e_from, e_to, z, depth=0):
    """Solve at e_to, warm-started from the state z at e_from. If Newton runs out of iterations (as when the
    path leaves an unstable branch), continue it up to 3 times with the same reference gradient; if it still
    fails, split the step in two (at most 3 levels, i.e. de / 8)."""
    Fb = np.diag([1 + e_to, 1.0, 1.0])  # Fbar_11 prescribed; the free Fbar_22, Fbar_33 come from z
    w, F, log = mix.solve(Fb, z0=z)
    tot = dict(newton=log["newton"], cg=log["cg"], restarts=0, splits=0)
    while not log["converged"] and not log.get("linesearch_failed") and tot["restarts"] < 3:
        w, F, log = mix.solve(Fb, z0=log["z"], g_ref=log["g0"])
        tot["newton"] += log["newton"]
        tot["cg"] += log["cg"]
        tot["restarts"] += 1
    if log["converged"] or depth >= 3:
        return w, F, log, tot
    em = 0.5 * (e_from + e_to)
    _, _, l1, t1 = solve_step(e_from, em, z, depth + 1)
    for key in tot:
        tot[key] += t1[key]
    if not l1["converged"]:
        return w, F, log, tot
    w, F, log, t2 = solve_step(em, e_to, l1["z"], depth + 1)
    for key in tot:
        tot[key] += t2[key]
    tot["splits"] += 1
    return w, F, log, tot


def kicked(z, k):
    """Random kick to the warm start after step k (deterministic in k)."""
    if perturb <= 0:
        return z
    nw = mix.nw
    wscale = float(np.abs(np.asarray(z[:nw])).max()) + 1e-12
    kick = np.random.default_rng(k).standard_normal(nw) * perturb * wscale
    return z.at[:nw].add(kick)


for sign, name in paths:
    z = None
    nsteps = int(round(emax / de))
    # state after every converged step, so a run killed by a GPU fault resumes where it stopped
    state = d / f"drift_state_{name}_de{de:g}{'_tile%d' % tile if tile > 1 else ''}{'_' + opt['--tag'] if '--tag' in opt else ''}.npz"
    k0 = 1
    if state.exists():
        st = np.load(state, allow_pickle=False)
        if float(st["perturb"]) == perturb and int(st["n"]) == n:
            rows.extend(json.loads(str(st["rows"])))
            k0 = int(st["k"]) + 1
            z = kicked(jnp.asarray(st["z"]), int(st["k"]))
            print(f"  resuming {name} after step {k0 - 1} (e {sign * de * (k0 - 1):+.2f}) from {state.name}", flush=True)
    for k in range(k0, nsteps + 1):
        e = sign * de * k
        t1 = time.time()
        w, F, log, tot = solve_step(e - sign * de, e, z)  # warm start from the previous step (w and lateral stretch)
        F = np.asarray(F)
        P = np.asarray(cell.mean_stress(w, F))
        nu_y = -(F[1, 1] - 1) / e
        nu_z = -(F[2, 2] - 1) / e
        ok = bool(log["converged"])
        rows.append(dict(path=name, e=e, nu_y=nu_y, nu_z=nu_z, P11=P[0, 0], P22=P[1, 1], P33=P[2, 2],
                         newton=tot["newton"], cg=tot["cg"], restarts=tot["restarts"], splits=tot["splits"],
                         seconds=time.time() - t1, converged=ok))
        print(f"  {name:11s} e {e:+.2f}  nu_y {nu_y:+.4f}  nu_z {nu_z:+.4f}  P11 {P[0,0]:+.5f}  "
              f"|P22|,|P33| {abs(P[1,1]):.0e},{abs(P[2,2]):.0e}  newton {tot['newton']:2d} cg {tot['cg']:5d}  "
              f"{'restarts %d splits %d  ' % (tot['restarts'], tot['splits']) if tot['restarts'] or tot['splits'] else ''}"
              f"{time.time()-t1:5.1f}s {'ok' if ok else 'NOT CONVERGED'}{' (round-off floor)' if log.get('roundoff_floor') else ''}",
              flush=True)
        if not ok:
            print(f"    |g| history: {' '.join(f'{x:.1e}' for x in log['gnorm'])}"
                  f"{'  (line search failed)' if log.get('linesearch_failed') else ''}")
            break
        z = log["z"]
        np.savez(state, z=np.asarray(z), k=k, perturb=perturb, n=n,
                 rows=json.dumps([{kk: (v.item() if hasattr(v, "item") else v) for kk, v in r.items()} for r in rows]))
        z = kicked(z, k)

tag = "_" + opt["--paths"].replace(",", "_") if "--paths" in opt else ""
tag += f"_tile{tile}" if tile > 1 else ""
tag += f"_{opt['--tag']}" if "--tag" in opt else ""
with open(d / f"drift{tag}.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

good = [r for r in rows if r["converged"]]
dev = [(abs(r["e"]), max(abs(r["nu_y"] - nu0), abs(r["nu_z"] - nu0_z)), r["path"]) for r in good]
maxdev = max((dv for _, dv, _ in dev), default=float("nan"))
first = {p: next((ae for ae, dv, pp in sorted(dev) if pp == p and dv > 0.05), None) for p in ("tension", "compression")}  # noqa: E501
reach = {p: max((abs(r["e"]) for r in good if r["path"] == p), default=0) for p in ("tension", "compression")}
print(f"[drift] {d.name}: nu0 {nu0:+.4f}; max |nu - nu0| {maxdev:.4f} over the converged path "
      f"(tension to {reach['tension']:.2f}, compression to {reach['compression']:.2f}); "
      f"first |e| with deviation > 0.05: tension {first['tension']}, compression {first['compression']}")
