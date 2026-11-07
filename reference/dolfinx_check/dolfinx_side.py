"""DOLFINx side of the cross-check: periodic finite-strain homogenization with dolfinx + dolfinx_mpc.

Same discrete model as the JAX reference solver, written independently in UFL: unit cube, n^3 Q1 hexahedra, 2x2x2 Gauss
points (quadrature degree 2), periodic fluctuation w (one periodic constraint maps the faces x=1, y=1, z=1 and
their edges and corners onto x=0, y=0, z=0), translation fixed at the origin node, F = Fbar + grad w.
Energy per cell: s * psi_eff(F), s = 1e-9 + rho; psi_eff = neo-Hookean, or with WLS void interpolation
psi_nh(F_g) - psi_lin(F_g) + psi_lin(F), F_g = I + gamma (F - I).
Newton with a direct solve (MUMPS) and a backtracking line search on the energy; Pbar = int s dpsi/dF.
Mixed control: Newton on the free Fbar components with a finite-difference 2x2 Jacobian.
Usage (dolfinx env): python -I dolfinx_side.py <dir> [field/load ...]. Writes <dir>/dolfinx.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import set_bc
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx_mpc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases as C  # noqa: E402

RHO_MIN = 1e-9


class PeriodicCell:
    def __init__(self, rho, wls=None):
        n = rho.shape[0]
        self.msh = msh = mesh.create_unit_cube(MPI.COMM_SELF, n, n, n, cell_type=mesh.CellType.hexahedron)
        V = fem.functionspace(msh, ("Lagrange", 1, (3,)))
        Q = fem.functionspace(msh, ("DG", 0))
        ncell = msh.topology.index_map(3).size_local
        mid = mesh.compute_midpoints(msh, 3, np.arange(ncell, dtype=np.int32))
        ijk = np.clip(np.floor(mid * n).astype(int), 0, n - 1)
        r = rho[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        dofs = np.array([Q.dofmap.cell_dofs(c)[0] for c in range(ncell)])
        s, gam = fem.Function(Q), fem.Function(Q)
        s.x.array[dofs] = RHO_MIN + r
        lam, mu = C.E * C.NU / ((1 + C.NU) * (1 - 2 * C.NU)), C.E / (2 * (1 + C.NU))
        if wls is not None:
            b, r0 = wls
            gam.x.array[dofs] = (np.tanh(b * r0) + np.tanh(b * (r - r0))) / (np.tanh(b * r0) + np.tanh(b * (1 - r0)))
        # translation: fix the origin node; periodicity: faces at 1 -> faces at 0 (edges and corners included)
        origin = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x, 0.0).all(axis=0))
        self.bc = fem.dirichletbc(np.zeros(3), origin, V)
        tol = 1e-8

        def on_far(x):
            return np.isclose(x[0], 1, atol=tol) | np.isclose(x[1], 1, atol=tol) | np.isclose(x[2], 1, atol=tol)

        def relation(x):
            y = x.copy()
            for i in range(3):
                y[i] = np.where(np.isclose(x[i], 1, atol=tol), x[i] - 1, x[i])
            return y

        self.mpc = mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_geometrical(V, on_far, relation, [self.bc])
        mpc.finalize()
        Vm = mpc.function_space
        # the pin must live on the constrained space, or the assembler ignores it (translation stays free)
        self.bc = fem.dirichletbc(np.zeros(3), fem.locate_dofs_geometrical(Vm, lambda x: np.isclose(x, 0.0).all(axis=0)), Vm)
        self.pin = self.bc.dof_indices()[0].astype(PETSc.IntType)
        self.w = w = fem.Function(Vm)
        self.Fbar = fem.Constant(msh, np.eye(3))
        I = ufl.Identity(3)

        def psi_nh(F):
            J = ufl.det(F)
            return 0.5 * mu * (ufl.tr(F.T * F) - 3) - mu * ufl.ln(J) + 0.5 * lam * ufl.ln(J) ** 2

        def psi_lin(F):
            e = ufl.sym(F - I)
            return 0.5 * lam * ufl.tr(e) ** 2 + mu * ufl.inner(e, e)

        def psi_eff(F):
            if wls is None:
                return psi_nh(F)
            Fg = I + gam * (F - I)
            return psi_nh(Fg) - psi_lin(Fg) + psi_lin(F)

        dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 2})
        Fv = ufl.variable(self.Fbar + ufl.grad(w))
        psi = s * psi_eff(Fv)
        P = ufl.diff(psi, Fv)
        v, du = ufl.TestFunction(Vm), ufl.TrialFunction(Vm)
        self.Pi = fem.form(psi * dx)
        R = ufl.derivative(psi * dx, w, v)
        self.R = fem.form(R)
        self.J = fem.form(ufl.derivative(R, w, du))
        self.Pf = [[fem.form(P[i, j] * dx) for j in range(3)] for i in range(3)]
        self.du = fem.Function(Vm)
        self.ksp = PETSc.KSP().create(MPI.COMM_SELF)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

    def energy(self):
        return self.msh.comm.allreduce(fem.assemble_scalar(self.Pi), op=MPI.SUM)

    def residual(self):
        b = dolfinx_mpc.assemble_vector(self.R, self.mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [self.bc])  # zero residual at the fixed origin dofs (g = 0)
        return b

    def solve(self, Fbar, rtol=1e-11, maxit=50):
        self.Fbar.value[:] = Fbar
        b = self.residual()
        r0 = b.norm()
        it = 0
        for it in range(maxit):
            if b.norm() <= max(rtol * max(r0, 1e-30), 1e-14):
                break
            A = dolfinx_mpc.assemble_matrix(self.J, self.mpc, bcs=[])
            A.assemble()
            # pin: identity rows and columns at the origin dofs. Passing the bc to the MPC assembler is not enough,
            # since the origin is the master of the corner slaves and their couplings are added after the bc.
            A.zeroRowsColumns(self.pin, 1.0)
            self.ksp.setOperators(A)
            b.scale(-1.0)
            self.ksp.solve(b, self.du.x.petsc_vec)
            self.du.x.scatter_forward()
            self.mpc.backsubstitution(self.du)
            w0 = self.w.x.array.copy()
            E0, t = self.energy(), 1.0
            for _ in range(20):  # backtracking on the energy (Newton direction of a convex model)
                self.w.x.array[:] = w0 + t * self.du.x.array
                if self.energy() <= E0 + 1e-14 * abs(E0):
                    break
                t *= 0.5
            A.destroy()
            b.destroy()
            b = self.residual()
        ok = b.norm() <= max(rtol * max(r0, 1e-30), 1e-14)
        b.destroy()
        return it, ok

    def Pbar(self):
        return np.array([[fem.assemble_scalar(self.Pf[i][j]) for j in range(3)] for i in range(3)])


def mixed_solve(cell, Fb, free, tol=1e-12, h=1e-6):
    """Free components of Fbar adjusted until their Pbar components vanish (Newton, FD Jacobian)."""
    F = np.array(Fb, dtype=float)
    w_keep = cell.w.x.array.copy()
    its = 0
    for k in range(30):
        its += cell.solve(F)[0]
        P = cell.Pbar()
        g = np.array([P[i, j] for i, j in free])
        if np.abs(g).max() <= tol * max(abs(P[0, 0]), 1e-30):
            break
        w_keep = cell.w.x.array.copy()
        Jm = np.zeros((len(free), len(free)))
        for c, (i, j) in enumerate(free):
            Fp = F.copy()
            Fp[i, j] += h
            cell.solve(Fp)
            Pp = cell.Pbar()
            Jm[:, c] = (np.array([Pp[a, b] for a, b in free]) - g) / h
            cell.w.x.array[:] = w_keep
        dF = np.linalg.solve(Jm, -g)
        for c, (i, j) in enumerate(free):
            F[i, j] += dF[c]
    # end on a solve at the returned Fbar: after the last FD probe, the cell would still hold the probe's Fbar
    cell.w.x.array[:] = w_keep
    its += cell.solve(F)[0]
    P = cell.Pbar()
    ok = max(abs(P[i, j]) for i, j in free) <= 1e-9 * max(abs(P[0, 0]), 1e-30)
    return F, its, ok


if __name__ == "__main__":
    d = Path(sys.argv[1])
    res = {}
    only = sys.argv[2:]  # optional "field/load" names to rerun; others are kept from an earlier dolfinx.json
    if only and (d / "dolfinx.json").exists():
        res = json.load(open(d / "dolfinx.json"))
    for name, f in C.FIELDS.items():
        rho = np.load(d / f"{name}.npy", allow_pickle=False)
        for ln, L in C.loads().items():
            if only and f"{name}/{ln}" not in only:
                continue
            t0 = time.time()
            cell = PeriodicCell(rho, f["wls"])
            rows = []
            Fcur = np.eye(3)
            for Fb in C.path(L["F"]):
                if L["free"] is None:
                    it, ok = cell.solve(Fb)
                    F = Fb
                else:
                    Fstart = np.array(Fb)
                    for i, j in L["free"]:
                        Fstart[i, j] = Fcur[i, j]
                    F, it, ok = mixed_solve(cell, Fstart, L["free"])
                Fcur = np.array(F)
                rows.append(dict(F=np.asarray(F).tolist(), P=cell.Pbar().tolist(), newton=int(it), converged=bool(ok)))
            res[f"{name}/{ln}"] = rows
            print(f"{name}/{ln}: P11 {rows[-1]['P'][0][0]:+.6e}  F22 {rows[-1]['F'][1][1]:.6f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(d / "dolfinx.json", "w"), indent=1)
