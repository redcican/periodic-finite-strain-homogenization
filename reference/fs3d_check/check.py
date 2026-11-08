"""fs3d against the FP64 JAX reference solver (reference/jax/finite_strain.py).

Items (--items):
  1  small-strain moduli C^H (fs3d -mode toec) against C_linear
  2  uniaxial-stress paths L1 (tension) and L2 (compression): Pbar_11(eps) and nu_12(eps) against Mixed
  3  adjoint gradient of Pbar_11 at the last load point of L1: fs3d adjoint against fs3d central differences
     (-mode fdcheck) and against the oracle's mixed adjoint (Mixed.grad_rho), both w.r.t. the physical density
  4  third-order constants: T[h,h,h] + 3 C[h, H^T H] from moduli.txt against the second derivative of
     g(t) = <Pbar(I + t H), H> (fully prescribed Fbar) by Richardson-extrapolated central differences
  5  stability: lambda_min of the scaled tangent along L2 (criterion 1) against LOBPCG on the oracle Hessian with
     the same scaling; the first load point with lambda_min <= 0 must agree within one step

Both solvers use E = 10, nu = 0.3, the neo-Hookean energy with the void interpolation (beta 500, rho0 0.01) and the
stiffness scale 1e-9 + rho^3 of the physical density rho in the .npy cell (shape (n, n, n), axis order x, y, z).

Usage (homo3d-jax env, GPU 1):
  CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python -I reference/fs3d_check/check.py \\
      --cells runs/cells/rod16.npy runs/cells/bulk32.npy --out runs/check --items 1 2 3 4 [--prec adaptive]
  Item 5 only makes sense for a cell that becomes unstable in compression: --items 5 --emin5 -0.14 --de5 0.01.
Oracle results are cached in <out>/<cell>/oracle_*.{npy,csv,json}; delete them to recompute. A summary table is
printed and appended to <out>/summary.csv.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "jax"))
import finite_strain as fs  # noqa: E402

E, NU, WLS = 10.0, 0.3, (500.0, 0.01)
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt(X):
    """Voigt vector of the symmetric part of X with engineering shears."""
    X = 0.5 * (X + X.T)
    return np.array([X[0, 0], X[1, 1], X[2, 2], 2 * X[1, 2], 2 * X[0, 2], 2 * X[0, 1]])


def stiff(rho):
    return np.clip(np.asarray(rho, dtype=np.float64), 0, 1) ** 3


# --------------------------------------------------------------------------------------------- fs3d
class Fs3d:
    def __init__(self, exe, prec, verbose):
        self.exe, self.prec, self.verbose = str(exe), prec, verbose

    def run(self, args, log):
        cmd = [self.exe] + [str(a) for a in args]
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t
        Path(log).write_text(" ".join(cmd) + "\n\n" + r.stdout + r.stderr)
        if self.verbose:
            print("    $ " + " ".join(cmd[1:]))
            for line in r.stdout.strip().splitlines()[-3:]:
                print("      " + line)
        return r.stdout + r.stderr, dt, r.returncode


def read_path_csv(f):
    with open(f, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0].keys():
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def read_moduli(f):
    lines = Path(f).read_text().splitlines()
    C = np.array([[float(v) for v in lines[i].split()] for i in range(1, 7)])
    T = np.zeros((6, 6, 6))
    for line in lines[8:]:
        name, val = line.split()
        I, J, K = (int(c) - 1 for c in name[1:])
        for a, b, c in {(I, J, K), (I, K, J), (J, I, K), (J, K, I), (K, I, J), (K, J, I)}:
            T[a, b, c] = float(val)
    return C, T


# --------------------------------------------------------------------------------------------- oracle
class Oracle:
    def __init__(self, rho, cache):
        self.rho = np.asarray(rho, dtype=np.float64)
        self.rs = stiff(rho)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._cell = self._mix = None

    @property
    def cell(self):
        if self._cell is None:
            self._cell = fs.Cell(self.rs, E, NU, material="neohooke", wls=WLS)
        return self._cell

    @property
    def mix(self):
        if self._mix is None:
            self._mix = fs.Mixed(self.cell, [(1, 1), (2, 2)], tangent="stored", forcing="ew")
        return self._mix

    def C(self):
        f = self.cache / "oracle_C.npy"
        if f.exists():
            return np.load(f)
        C, logs = fs.C_linear(self.rs, E, NU)
        if not all(l["converged"] for l in logs):
            print("    warning: oracle C_linear not converged")
        np.save(f, C)
        return C

    def path(self, name, eps):
        """Uniaxial stress along x at the strains eps (F11 = 1 + eps, Pbar_22 = Pbar_33 = 0); cached CSV with the
        rows eps, P11, F22, F33, P22, P33, newton, cg, seconds. Also keeps the states for item 5."""
        f = self.cache / f"oracle_{name}.csv"
        rows = []
        if f.exists():
            with open(f, newline="") as fh:
                rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        have = {round(r["eps"], 10) for r in rows}
        todo = [e for e in eps if round(e, 10) not in have]
        if todo:
            z, g0 = None, None
            for e in sorted(todo, key=abs):
                Fb = np.diag([1 + e, 1.0, 1.0])
                t = time.time()
                w, F, log = self.mix.solve(Fb, z0=z, rtol=1e-10)
                z = log["z"]
                P = np.asarray(self.cell.mean_stress(w, F))
                F = np.asarray(F)
                rows.append(dict(eps=e, P11=P[0, 0], F22=F[1, 1], F33=F[2, 2], P22=P[1, 1], P33=P[2, 2], newton=log["newton"],
                                 cg=log["cg"], seconds=time.time() - t, converged=float(log["converged"])))
                print(f"    oracle {name} eps {e:+.4f}: P11 {P[0, 0]:+.6e} nu {-(F[1, 1] - 1) / e:+.5f} newton {log['newton']} "
                      f"cg {log['cg']} {time.time() - t:.1f}s{'' if log['converged'] else '  NOT CONVERGED'}", flush=True)
            rows.sort(key=lambda r: r["eps"])
            with open(f, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
        return {k: np.array([r[k] for r in rows if round(r["eps"], 10) in {round(e, 10) for e in eps}]) for k in rows[0]}

    def grad_P11(self, eps):
        """d Pbar_11 / d rho (physical density) under uniaxial stress at F11 = 1 + eps."""
        f = self.cache / f"oracle_grad_P11_{eps:.4f}.npy"
        if f.exists():
            return np.load(f)
        z = None
        for e in np.arange(1, 6) * eps / 5:  # five steps to the load point, as fs3d
            w, F, log = self.mix.solve(np.diag([1 + e, 1.0, 1.0]), z0=z, rtol=1e-10)
            z = log["z"]
        G = np.zeros((3, 3))
        G[0, 0] = 1.0
        g, log = self.mix.grad_rho(z, F, G)
        g = np.asarray(g) * 3 * self.rho**2  # chain rule: stiffness density rho^3
        print(f"    oracle adjoint: cg {log['cg']}{'' if log['converged'] else '  NOT CONVERGED'}")
        np.save(f, g)
        return g

    def energy_curvature(self, H, delta):
        """g''(0) of g(t) = <Pbar(I + t H), H> by Richardson-extrapolated central differences (steps delta, 2 delta)."""
        key = "oracle_toec.json"
        f = self.cache / key
        data = json.loads(f.read_text()) if f.exists() else {}
        k = json.dumps([list(np.round(H, 12).ravel()), delta])
        if k in data:
            return data[k]
        H = np.asarray(H, dtype=np.float64)

        def g(t):
            Fb = np.eye(3) + t * H
            w, log = self.cell.solve(Fb, rtol=1e-11)
            if not log["converged"]:
                print("    warning: oracle solve not converged at t =", t)
            return float(np.sum(np.asarray(self.cell.mean_stress(w, Fb)) * H))

        D1 = (g(delta) + g(-delta)) / delta**2  # g(0) = 0 at F = I
        D2 = (g(2 * delta) + g(-2 * delta)) / (2 * delta) ** 2
        val = (4 * D1 - D2) / 3
        data[k] = val
        f.write_text(json.dumps(data, indent=1))
        return val

    def stability(self, eps):
        """lambda_min of D^-1/2 H D^-1/2 (D: Jacobi diagonal at the undeformed state, translations excluded)
        along the uniaxial-stress path at the strains eps, as reference/jax/stability_path.py."""
        f = self.cache / "oracle_stability.csv"
        rows = []
        if f.exists():
            with open(f, newline="") as fh:
                rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        have = {round(r["eps"], 10) for r in rows}
        todo = [e for e in [0.0] + list(eps) if round(e, 10) not in have]
        if todo:
            import jax
            import jax.numpy as jnp
            from scipy.sparse.linalg import LinearOperator, lobpcg

            mix = self.mix
            nz = mix.nw + 2
            hvp_block = jax.jit(lambda z, Fb, V: jax.lax.map(lambda v: mix._hvp_raw(z, Fb, v), V.T, batch_size=4).T)
            z0 = jnp.concatenate([jnp.zeros(mix.nw), jnp.ones(2)])
            Dm12 = jnp.sqrt(mix._minv(z0, jnp.eye(3)))
            T = np.zeros((nz, 3))
            for i in range(3):
                T[i:mix.nw:3, i] = 1.0
            Y, _ = np.linalg.qr(T / np.asarray(Dm12)[:, None])
            X = np.random.default_rng(0).standard_normal((nz, 4))

            def lam_min(z, Fb):
                nonlocal X
                zj, Fj = jnp.asarray(z), jnp.asarray(Fb)

                def mm(v):
                    V = jnp.asarray(np.asarray(v, dtype=np.float64).reshape(nz, -1)) * Dm12[:, None]
                    return np.asarray(hvp_block(zj, Fj, V) * Dm12[:, None]).reshape(v.shape)

                op = LinearOperator((nz, nz), matvec=mm, matmat=mm, dtype=np.float64)
                vals, vecs = lobpcg(op, X, Y=Y, largest=False, tol=1e-5, maxiter=2000)
                X = vecs
                return float(np.min(vals))

            z = None
            for e in sorted(todo, key=abs):
                t = time.time()
                if e == 0.0:
                    lam, P11, F = lam_min(z0, np.eye(3)), 0.0, np.eye(3)
                else:
                    w, F, log = mix.solve(np.diag([1 + e, 1.0, 1.0]), z0=z, rtol=1e-10)
                    z = log["z"]
                    F = np.asarray(F)
                    P11 = float(np.asarray(self.cell.mean_stress(w, F))[0, 0])
                    lam = lam_min(z, F)
                rows.append(dict(eps=e, lam1=lam, P11=P11, F22=F[1, 1], seconds=time.time() - t))
                print(f"    oracle stability eps {e:+.3f}: lambda_min {lam:+.4e} ({time.time() - t:.1f}s)", flush=True)
            rows.sort(key=lambda r: r["eps"], reverse=eps[0] < 0)
            with open(f, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
        byeps = {round(r["eps"], 10): r for r in rows}
        return byeps[0.0]["lam1"], np.array([byeps[round(e, 10)]["lam1"] for e in eps])


# --------------------------------------------------------------------------------------------- items
def item1(fsd, cell, npy, out, orc, res):
    o = out / "toec"
    o.mkdir(exist_ok=True)
    txt, dt, rc = fsd.run(["-mode", "toec", "-in", npy, "-prefix", f"{o}/", "-prec", fsd.prec], o / "log.txt")
    if rc != 0:
        res.append((cell, 1, "C^H", "fs3d failed", "FAIL"))
        return None
    C, T = read_moduli(o / "moduli.txt")
    Co = orc.C()
    rel = np.linalg.norm(C - Co) / np.linalg.norm(Co)
    cmax = np.abs(C - Co).max() / np.abs(Co).max()
    print(f"  [1] C^H: fs3d C11 {C[0, 0]:.6f} C12 {C[0, 1]:.6f} C44 {C[3, 3]:.6f} | oracle {Co[0, 0]:.6f} {Co[0, 1]:.6f} "
          f"{Co[3, 3]:.6f} | rel Frobenius {rel:.2e}, max entry {cmax:.2e} ({dt:.1f}s)")
    res.append((cell, 1, "C^H rel. Frobenius error", f"{rel:.2e}", "pass" if rel <= 1e-4 else "FAIL"))
    return C, T


def item2(fsd, cell, npy, out, orc, res, emax, de, crit):
    nsteps = int(round(emax / de))
    for name, sign in (("L1", 1), ("L2", -1)):
        o = out / f"path_{name}"
        o.mkdir(exist_ok=True)
        txt, dt, rc = fsd.run(["-mode", "path", "-in", npy, "-prefix", f"{o}/", "-path", name, "-steps", nsteps, "-inc", de,
                               "-prec", fsd.prec, "-crit", crit, "-nostop_distortion"], o / "log.txt")
        p = read_path_csv(o / f"path_{name}.csv")
        eps = sign * de * np.arange(1, len(p["eps"]) + 1)
        orow = orc.path(name, list(eps))
        nu_fs = -(p["F22"] - 1) / p["eps"]
        nu_or = -(orow["F22"] - 1) / orow["eps"]
        relP = np.abs(p["P11"] - orow["P11"]) / np.abs(orow["P11"])
        dnu = np.abs(nu_fs - nu_or)
        sec = p["seconds"] if "seconds" in p else np.full(len(eps), np.nan)
        status = "completed" if len(p["eps"]) == nsteps else f"{len(p['eps'])}/{nsteps} points"
        print(f"  [2] {name}: {status}, eps to {eps[-1]:+.3f}; max rel |dP11| {relP.max():.2e}, max |dnu| {dnu.max():.2e}; "
              f"newton/step {p['newton'].mean():.1f} (max {int(p['newton'].max())}), krylov/step {p['krylov'].mean():.0f}, "
              f"{np.nansum(sec):.2f}s fs3d ({dt:.1f}s wall) vs {orow['seconds'].sum():.0f}s oracle; "
              f"prec {','.join(sorted(set(p['prec'])))}, minres steps {int(p['minres'].sum())}")
        for i in range(len(eps)):
            print(f"      eps {eps[i]:+.3f}  P11 {p['P11'][i]:+.6e} / {orow['P11'][i]:+.6e}  nu {nu_fs[i]:+.5f} / {nu_or[i]:+.5f}  "
                  f"newton {int(p['newton'][i])} krylov {int(p['krylov'][i])} {sec[i]:.3f}s"
                  + (f"  lambda {p['lambda_cell'][i]:+.3e}" if np.isfinite(p["lambda_cell"][i]) else ""))
        ok = relP.max() <= 1e-3 and dnu.max() <= 1e-3 and len(p["eps"]) == nsteps
        res.append((cell, 2, f"{name} to eps={eps[-1]:+.2f}: max rel |dP11| / max |dnu|", f"{relP.max():.1e} / {dnu.max():.1e}",
                    "pass" if ok else "FAIL"))
        res.append((cell, 2, f"{name} Newton per step (mean/max), fs3d seconds per path", f"{p['newton'].mean():.1f}/{int(p['newton'].max())}, {np.nansum(sec):.2f}s", ""))


def item3(fsd, cell, npy, out, orc, res, fd_h, fd_n):
    o = out / "fdcheck"
    o.mkdir(exist_ok=True)
    txt, dt, rc = fsd.run(["-mode", "fdcheck", "-in", npy, "-path", "L1", "-steps", 5, "-inc", 0.02, "-obj", "P11", "-prec",
                           "fp32", "-fd_n", fd_n, "-fd_h", fd_h, "-fd_out", o / "grad_fs3d.npy", "-newton_tol", 1e-9],
                          o / "log.txt")
    worst = np.nan
    for line in txt.splitlines():
        if line.startswith("largest relative error"):
            worst = float(line.split()[3])
        if line.startswith("voxel"):
            print("      " + line.strip())
    g = np.load(o / "grad_fs3d.npy")
    go = orc.grad_P11(0.1)
    rel = np.linalg.norm(g - go) / np.linalg.norm(go)
    cos = float(np.sum(g * go) / np.linalg.norm(g) / np.linalg.norm(go))
    print(f"  [3] dP11/drho at eps = 0.1 (L1, 5 steps): fs3d adjoint vs fs3d central differences (h = {fd_h}, {fd_n} voxels): "
          f"worst rel {worst:.2e}; vs oracle adjoint: rel L2 {rel:.2e}, cosine {cos:.8f} ({dt:.1f}s)")
    res.append((cell, 3, f"adjoint vs fs3d central differences (h={fd_h})", f"{worst:.1e}", "pass" if worst <= 1e-4 else "FAIL"))
    res.append((cell, 3, "adjoint vs oracle adjoint, rel. L2", f"{rel:.1e}", "pass" if rel <= 1e-3 else "FAIL"))


def item4(cell, CT, orc, res, delta):
    if CT is None:
        return
    C, T = CT
    dirs = {
        "e1 x e1 (T111 + 3 C11)": np.diag([1.0, 0.0, 0.0]),
        "e1 x e1 + e2 x e2": np.diag([1.0, 1.0, 0.0]),
        "e2 x e3 + e3 x e2 (8 T444 + ...)": np.array([[0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]]),
        "general symmetric": np.array([[0.6, 0.2, 0.15], [0.2, -0.3, 0.1], [0.15, 0.1, 0.2]]),
    }
    worst = 0.0
    for name, H in dirs.items():
        h = voigt(H)
        x = voigt(H.T @ H)
        Gfs = np.einsum("ijk,i,j,k->", T, h, h, h) + 3 * h @ C @ x
        Gor = orc.energy_curvature(H, delta)
        rel = abs(Gfs - Gor) / max(abs(Gor), 1e-30)
        worst = max(worst, rel)
        print(f"  [4] W''' along {name}: fs3d {Gfs:+.6e}  oracle {Gor:+.6e}  rel {rel:.2e}")
    res.append((cell, 4, f"TOEC: W''' along 4 strain directions, worst rel (oracle FD delta={delta})", f"{worst:.1e}",
                "pass" if worst <= 5e-3 else "FAIL"))


def item5(fsd, cell, npy, out, orc, res, emin, de):
    nsteps = int(round(abs(emin) / de))
    o = out / "stability_L2"
    o.mkdir(exist_ok=True)
    txt, dt, rc = fsd.run(["-mode", "path", "-in", npy, "-prefix", f"{o}/", "-path", "L2", "-steps", nsteps, "-inc", de, "-prec",
                           fsd.prec, "-crit", "1", "-nostop_distortion"], o / "log.txt")
    p = read_path_csv(o / "path_L2.csv")
    lam0 = np.nan
    for line in txt.splitlines():
        if "lambda_min(0)" in line:
            lam0 = float(line.split("=")[1])
    eps = -de * np.arange(1, len(p["eps"]) + 1)
    lam0o, lamo = orc.stability(list(eps))
    r_fs, r_or = p["lambda_cell"] / lam0, lamo / lam0o
    first = lambda r, e: (e[np.argmax(r <= 0)] if np.any(r <= 0) else np.nan)  # noqa: E731
    e_fs, e_or = first(r_fs, eps), first(r_or, eps)
    print(f"  [5] lambda_min(0): fs3d {lam0:.4e}, oracle {lam0o:.4e} (rel {abs(lam0 - lam0o) / lam0o:.1e})")
    for i in range(len(eps)):
        print(f"      eps {eps[i]:+.3f}  lambda/lambda0 fs3d {r_fs[i]:+.4f}  oracle {r_or[i]:+.4f}  (fs3d lambda {p['lambda_cell'][i]:+.3e}, "
              f"oracle {lamo[i]:+.3e})  newton {int(p['newton'][i])} {p['seconds'][i]:.2f}s")
    dmax = np.nanmax(np.abs(r_fs - r_or))
    ok = (np.isnan(e_fs) and np.isnan(e_or)) or abs(e_fs - e_or) <= de + 1e-12
    print(f"  [5] first lambda_min <= 0: fs3d eps {e_fs}, oracle eps {e_or}; max |d(lambda/lambda0)| {dmax:.2e} ({dt:.1f}s fs3d)")
    res.append((cell, 5, "first-instability strain fs3d / oracle", f"{e_fs} / {e_or}", "pass" if ok else "FAIL"))
    res.append((cell, 5, "max |d(lambda/lambda0)| along L2", f"{dmax:.1e}", "pass" if dmax <= 0.05 else "FAIL"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True, help="density .npy files (physical density, shape (n, n, n))")
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--fs3d", default=str(ROOT / "build" / "fs3d"))
    ap.add_argument("--prec", default="adaptive", help="fs3d coarse stencil precision for items 1, 2, 5")
    ap.add_argument("--emax", type=float, default=0.2, help="item 2: largest strain")
    ap.add_argument("--de", type=float, default=0.02, help="item 2: load increment")
    ap.add_argument("--crit", default="1", help="item 2: fs3d stability criteria to evaluate along the paths")
    ap.add_argument("--fd_h", type=float, default=1e-3)
    ap.add_argument("--fd_n", type=int, default=6)
    ap.add_argument("--delta", type=float, default=0.01, help="item 4: oracle finite-difference step")
    ap.add_argument("--emin5", type=float, default=-0.14)
    ap.add_argument("--de5", type=float, default=0.01)
    ap.add_argument("-v", "--verbose", action="store_true", help="print the fs3d command lines")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fsd = Fs3d(a.fs3d, a.prec, a.verbose)
    res = []
    for npy in a.cells:
        npy = Path(npy).resolve()
        cell = npy.stem
        rho = np.load(npy, allow_pickle=False).astype(np.float64)
        o = out / cell
        o.mkdir(exist_ok=True)
        orc = Oracle(rho, o)
        print(f"== {cell}: n = {rho.shape[0]}, volume {rho.mean():.4f}, prec {a.prec}")
        CT = None
        if 1 in a.items or 4 in a.items:
            CT = item1(fsd, cell, npy, o, orc, res)
        if 2 in a.items:
            item2(fsd, cell, npy, o, orc, res, a.emax, a.de, a.crit)
        if 3 in a.items:
            item3(fsd, cell, npy, o, orc, res, a.fd_h, a.fd_n)
        if 4 in a.items:
            item4(cell, CT, orc, res, a.delta)
        if 5 in a.items:
            item5(fsd, cell, npy, o, orc, res, a.emin5, a.de5)
    print("\nsummary")
    with open(out / "summary.csv", "a", newline="") as fh:
        wr = csv.writer(fh)
        for r in res:
            print(f"  {r[0]:>10}  item {r[1]}  {r[2]:<70} {r[3]:>22}  {r[4]}")
            wr.writerow([time.strftime("%Y-%m-%d %H:%M"), a.prec] + list(r))
    return 0 if all(r[4] != "FAIL" for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
