"""Stability criterion (scaled lambda_min) along a uniaxial-stress path.
At each converged state z(e) of the mixed problem (Fbar_11 = 1 + e prescribed, Fbar_22, Fbar_33 free), the lowest
eigenvalues of the diagonally scaled Hessian S = D^-1/2 H(z(e)) D^-1/2 are computed with LOBPCG, with D the Jacobi
diagonal at the undeformed state and rigid translations excluded as constraints. By Sylvester's law S and H have
the same inertia, so lam_min(S) <= 0 marks loss of stability of the cell-periodic solution; the scaling keeps
near-void modes from swamping the spectrum. Report lam_min(e) / lam_min(0).
Usage (reference env): python -I stability_path.py <homo3d result dir> [--emin -0.14] [--de 0.01] [--k 4] [--tile 1] [--chunk 4]
Writes <dir>/stability_{compression|tension}[_tileN].csv.
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

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict(zip(sys.argv[1:], sys.argv[2:]))
emin = float(opt.get("--emin", -0.14))
de = float(opt.get("--de", 0.01))
k = int(opt.get("--k", 4))
tile = int(opt.get("--tile", 1))
d = Path(args[0])

rho = np.load(d / "rho.npy").astype(np.float64)
rho_phys = np.clip(rho, 0, 1) ** 3
if tile > 1:
    rho_phys = np.tile(rho_phys, (tile,) * 3)
cell = O.Cell(rho_phys, material="neohooke", wls=(500.0, 0.01))
mix = O.Mixed(cell, [(1, 1), (2, 2)])
nz = mix.nw + 2
hvp = jax.jit(mix._hvp_raw)
proj = jax.jit(mix._proj)

chunk = int(opt.get("--chunk", 4 if tile == 1 else 1))  # Hessian-vector products evaluated together (memory)
hvp_block = jax.jit(lambda z, Fb, V: jax.lax.map(lambda v: mix._hvp_raw(z, Fb, v), V.T, batch_size=chunk).T)
I3 = np.eye(3)
z0 = jnp.concatenate([jnp.zeros(mix.nw), jnp.ones(2)])
Dm12 = jnp.sqrt(mix._minv(z0, jnp.asarray(I3)))  # D^-1/2

# rigid translations t_i of w; in scaled coordinates the null vectors are D^1/2 t_i
T = np.zeros((nz, 3))
for i in range(3):
    T[i:mix.nw:3, i] = 1.0
Y, _ = np.linalg.qr(T / np.asarray(Dm12)[:, None])


def op(z, Fb):
    zj, Fj = jnp.asarray(z), jnp.asarray(Fb)

    def mm(v):
        v = np.asarray(v, dtype=np.float64)
        V = jnp.asarray(v.reshape(nz, -1)) * Dm12[:, None]
        out = np.asarray(hvp_block(zj, Fj, V) * Dm12[:, None])
        return out.reshape(v.shape)
    return LinearOperator((nz, nz), matvec=mm, matmat=mm, dtype=np.float64)


rng = np.random.default_rng(0)
X = rng.standard_normal((nz, k))


def lam_min(z, Fb):
    global X
    t0 = time.time()
    vals, vecs, hist = lobpcg(op(z, Fb), X, Y=Y, largest=False, tol=1e-5, maxiter=2000,
                              retLambdaHistory=True)
    X = vecs  # warm start for the next state
    order = np.argsort(vals)
    return vals[order], len(hist), time.time() - t0


rows = []
vals, its, dt = lam_min(z0, I3)
lam0 = vals[0]
print(f"e +0.00  lam {' '.join(f'{v:+.3e}' for v in vals)}  (lobpcg {its} its, {dt:.1f}s)", flush=True)
rows.append(dict(e=0.0, lam1=vals[0], lam2=vals[1], lam3=vals[2] if k > 2 else np.nan, P11=0.0, nu=np.nan, its=its))
z = None
nsteps = int(round(abs(emin) / de))
sign = -1 if emin < 0 else 1
for s in range(1, nsteps + 1):
    e = sign * de * s
    Fb = np.diag([1 + e, 1.0, 1.0])
    w, F, log = mix.solve(Fb, z0=z)
    if not log["converged"]:
        print(f"e {e:+.2f}  Newton did not converge; stop")
        break
    z = log["z"]
    F = np.asarray(F)
    P11 = float(np.asarray(cell.mean_stress(w, F))[0, 0])
    nu = -(F[1, 1] - 1) / e
    vals, its, dt = lam_min(z, F)
    rows.append(dict(e=e, lam1=vals[0], lam2=vals[1], lam3=vals[2] if k > 2 else np.nan, P11=P11, nu=nu, its=its))
    print(f"e {e:+.2f}  nu {nu:+.4f}  P11 {P11:+.5f}  newton {log['newton']:2d}  lam1/lam1(0) {vals[0]/lam0:+.4f}  lam {' '.join(f'{v:+.3e}' for v in vals)}"
          f"  (lobpcg {its} its, {dt:.1f}s)", flush=True)

tag = ("_compression" if sign < 0 else "_tension") + (f"_tile{tile}" if tile > 1 else "")
with open(d / f"stability{tag}.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
