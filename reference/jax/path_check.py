"""Finite-strain checks on one density field (e.g. a near-binary 16^3 cell).
1. Small-strain consistency: neo-Hookean Pbar at 1e-6 strain vs the linear C^H columns.
2. Uniaxial-strain paths Fbar = diag(1+e, 1, 1), e = +-0.02..0.20: Newton/CG iterations and time per step.
3. Adjoint gradient of q = Pbar_11 w.r.t. rho vs central finite differences at e = 0.1.
Usage (reference env): python -I path_check.py <density .npy> [--wls]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finite_strain as O  # noqa: E402

src = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
rho = np.load(src, allow_pickle=False).astype(np.float64)
print(f"cell n={rho.shape[0]}, rho in [{rho.min():.3g}, {rho.max():.3g}], mean {rho.mean():.3f}")

# 1. small-strain consistency
Clin, _ = O.C_linear(rho)
WLS = (500.0, 0.01) if "--wls" in sys.argv else None
nh = O.Cell(rho, material="neohooke", wls=WLS)
print(f"void interpolation: {WLS}")
delta = 1e-6
worst = 0.0
for I in range(6):
    Fb = np.eye(3) + delta * O.unit_strain(I)
    w, _ = nh.solve(Fb, rtol=1e-9)
    P = np.asarray(nh.mean_stress(w, Fb)) / delta
    col = np.array([0.5 * (P[a, b] + P[b, a]) for a, b in O.VOIGT])
    worst = max(worst, np.abs(col - Clin[:, I]).max() / np.abs(Clin).max())
print(f"[small strain] neo-Hookean vs linear C^H at strain 1e-6: max rel diff {worst:.2e}")

# 2. load paths
for sign, name in ((1, "tension"), (-1, "compression")):
    w = None
    t_path = time.time()
    print(f"[path] {name}:  e      P11        newton  cg    s    converged")
    for k in range(1, 16 if sign < 0 else 11):
        e = sign * 0.02 * k
        Fb = np.diag([1 + e, 1.0, 1.0])
        t0 = time.time()
        w, log = nh.solve(Fb, w0=w, rtol=1e-9, maxnewton=25)
        P = np.asarray(nh.mean_stress(w, Fb))
        print(f"[path] {name}: {e:+.2f}  {P[0,0]:+.5f}  {log['newton']:6d} {log['cg']:5d} {time.time()-t0:5.1f}  {log['converged']}")
        if not log["converged"]:
            print(f"[path] {name}: stopped (Newton did not converge at e = {e:+.2f})")
            break
    print(f"[path] {name}: {time.time()-t_path:.1f} s for the path")

# 3. adjoint vs finite differences
e = 0.1
Fb = np.diag([1 + e, 1.0, 1.0])
G = np.zeros((3, 3)); G[0, 0] = 1.0
w0, _ = nh.solve(np.diag([1.05, 1, 1]), rtol=1e-11)
w, _ = nh.solve(Fb, w0=w0, rtol=1e-11)
g, info = nh.grad_rho(w, Fb, G)
g = np.asarray(g)
rng = np.random.default_rng(0)
v = rng.standard_normal(rho.shape)
lin = float(np.sum(g * v))
for eps in (1e-4, 1e-5):
    q = []
    for sgn in (1, -1):
        c = nh.with_rho(rho + sgn * eps * v)
        ws, _ = c.solve(Fb, w0=w, rtol=1e-11)
        q.append(float(np.asarray(c.mean_stress(ws, Fb))[0, 0]))
    fd = (q[0] - q[1]) / (2 * eps)
    print(f"[adjoint] dP11/drho . v: adjoint {lin:.8e}  fd(eps={eps:g}) {fd:.8e}  rel diff {abs(lin-fd)/abs(fd):.2e}  (adjoint CG {info['cg']})")
