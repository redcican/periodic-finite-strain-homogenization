"""Bloch-wave stability along a uniaxial-stress path (all wavelengths, not only the cell period).

A perturbation of the infinite lattice u(x) = u~(x) exp(2 pi i kappa . x), u~ cell-periodic, kappa in the
Brillouin zone [-1/2, 1/2]^3 (reciprocal cell units). On element e with corner a at offset h o_a:
    G = sum_a dN_a (x) u~_{n(a)} exp(2 pi i kappa . o_a / n)        (the common factor exp(i k . x_e) cancels)
and the second variation is sum_{e,g} s_e w_q conj(G) : A : G with A = d2 psi / dF2 at the converged state.
H(kappa) is Hermitian; it is applied in real form on (Re u~, Im u~), where each eigenvalue appears twice.
As in stability_path.py the operator is scaled by D^-1/2 (Jacobi diagonal of the linear stiffness; the diagonal
of H(kappa) does not depend on kappa), which keeps the inertia, and LOBPCG gives the lowest eigenvalues.
kappa = 0 drops the macroscopic unknowns (see stability_path.py for that case) and is skipped.
The state is the cell-periodic (symmetric) branch, followed in steps of 0.01 from e = 0.

Usage (reference env):
  python -I bloch_scan.py <homo3d result dir> --strains 0,-0.06 --kappas 0.5,0,0/0,0.5,0  [--k 4]
  python -I bloch_scan.py <dir> --strains 0,-0.04 --grid 5 [--part 0/2]    # wedge kappa_x>=0, kappa_y>=kappa_z>=0
Writes <dir>/bloch_<tag>.csv (tag from --tag, default "run").
"""
import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finite_strain as O  # noqa: E402

opt = dict(zip(sys.argv[1:], sys.argv[2:]))
d = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
strains = [float(x) for x in opt.get("--strains", "0").split(",")]
k = int(opt.get("--k", 4))
tag = opt.get("--tag", "run")
if "--kappas" in opt:
    kappas = [tuple(float(c) for c in s.split(",")) for s in opt["--kappas"].split("/")]
else:
    m = int(opt.get("--grid", 5))
    vals = np.linspace(0, 0.5, m)
    kappas = [(a, b, c) for a in vals for b in vals for c in vals if b >= c and (a, b, c) != (0, 0, 0)]
if "--part" in opt:
    i, p = (int(x) for x in opt["--part"].split("/"))
    kappas = kappas[i::p]

rho = np.load(d / "rho.npy").astype(np.float64)
cell = O.Cell(np.clip(rho, 0, 1) ** 3, material="neohooke", wls=(500.0, 0.01))
mix = O.Mixed(cell, [(1, 1), (2, 2)])
n = cell.n
N = 3 * n**3
OFF = jnp.asarray(O._NODES, dtype=jnp.float64)  # (8,3) corner offsets
Dm12 = 1.0 / jnp.sqrt(cell._diag)  # (n,n,n,3)


def psi_sum(F):
    return cell.wq * jnp.sum(cell.s[None] * cell.psi_eff(F, cell.gamma))


grad_psi = jax.grad(psi_sum)


def bloch_grad(x, kap):
    """x: (2, n, n, n, 3) real and imaginary parts of u~ (scaled coordinates). Returns (Gr, Gi)."""
    ur, ui = x[0] * Dm12, x[1] * Dm12
    th = 2 * jnp.pi * (OFF @ kap) / n  # (8,)
    c, s = jnp.cos(th), jnp.sin(th)

    def nodal(u):
        return jnp.stack([jnp.roll(u, shift=tuple(-int(o) for o in off), axis=(0, 1, 2)) for off in O._NODES])

    Ur, Ui = nodal(ur), nodal(ui)  # (8,n,n,n,3)
    cc, ss = c[:, None, None, None, None], s[:, None, None, None, None]
    Gr = jnp.einsum("aijkc,gad->gijkcd", Ur * cc - Ui * ss, cell.dN)
    Gi = jnp.einsum("aijkc,gad->gijkcd", Ur * ss + Ui * cc, cell.dN)
    return Gr, Gi


def _matvec(F, kap, x):
    (Gr, Gi), back = jax.vjp(lambda y: bloch_grad(y, kap), x)
    AGr = jax.jvp(grad_psi, (F,), (Gr,))[1]
    AGi = jax.jvp(grad_psi, (F,), (Gi,))[1]
    return back((AGr, AGi))[0]


matvec = jax.jit(_matvec)


def lam_min(F, kap, X):
    kj = jnp.asarray(kap, dtype=jnp.float64)

    def mm(v):
        v = np.asarray(v, dtype=np.float64)
        cols = v.reshape(2 * N, -1)
        out = [np.asarray(matvec(F, kj, jnp.asarray(cc.reshape(2, n, n, n, 3)))).ravel() for cc in cols.T]
        return np.stack(out, axis=1).reshape(v.shape)

    A = LinearOperator((2 * N, 2 * N), matvec=mm, matmat=mm, dtype=np.float64)
    vals, vecs, hist = lobpcg(A, X, largest=False, tol=1e-5, maxiter=3000, retLambdaHistory=True)
    o = np.argsort(vals)
    return vals[o], vecs[:, o], len(hist)


# follow the symmetric branch to each requested strain
states = {}
z = None
targets = sorted(set(strains), key=abs)
for e_t in targets:
    if e_t == 0:
        states[0.0] = np.eye(3)[None, None, None, None] + np.zeros((8, n, n, n, 3, 3))
        continue
    sign = 1 if e_t > 0 else -1
    steps = int(round(abs(e_t) / 0.01))
    z = None
    for s_ in range(1, steps + 1):
        e = sign * 0.01 * s_
        w, Fb, log = mix.solve(np.diag([1 + e, 1.0, 1.0]), z0=z)
        assert log["converged"], f"Newton failed at e = {e}"
        z = log["z"]
    states[e_t] = np.asarray(Fb + cell.grad_w(w))
    print(f"state e {e_t:+.3f}: nu {-(float(Fb[1, 1]) - 1) / e_t:+.4f}", flush=True)

rng = np.random.default_rng(0)
rows = []
out = d / f"bloch_{tag}.csv"
for e_t in strains:
    F = jnp.asarray(states[e_t])
    X = rng.standard_normal((2 * N, k))
    for kap in kappas:
        t0 = time.time()
        vals, X, its = lam_min(F, kap, X)
        rows.append(dict(e=e_t, kx=kap[0], ky=kap[1], kz=kap[2], lam1=vals[0], lam2=vals[2] if k > 2 else np.nan, its=its))
        print(f"e {e_t:+.3f}  kappa ({kap[0]:.3f},{kap[1]:.3f},{kap[2]:.3f})  lam {' '.join(f'{v:+.4e}' for v in vals)}"
              f"  ({its} its, {time.time()-t0:.1f}s)", flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
