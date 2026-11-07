"""Periodic finite-strain homogenization on a voxel grid (JAX). FP64 reference solver.

Unit cell [0,1]^3 on an n^3 periodic grid of trilinear (Q1) hexahedra, 2x2x2 Gauss points, as in homo3d.
Unknown: periodic fluctuation w at the n^3 nodes; deformation F = Fbar + grad w.
Element stiffness scale s_e = 1e-9 + rho_e (homo3d's convention; pass the physical density).

Materials (Lame constants lam, mu from E, nu):
  linear:      psi = lam/2 tr(eps)^2 + mu eps:eps,  eps = sym(H),  H = F - I
  neo-Hookean: psi = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2   (same small-strain limit)

Solver: minimize the cell energy over w by Newton-CG. The gradient comes from jax.grad of the energy,
Hessian-vector products from jax.jvp, with a Jacobi preconditioner from the linear element stiffness.
Average stress Pbar = dWbar/dFbar at equilibrium (envelope theorem). Everything runs in float64.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

RHO_MIN = 1e-9
VERBOSE = bool(int(__import__("os").environ.get("ORACLE_VERBOSE", "0")))  # per-Newton progress of Mixed.solve

# reference shape-function gradients of the 8 Q1 nodes at the 8 Gauss points, cube [0,1]^3
_g = 0.5 + np.array([-1.0, 1.0]) / (2 * np.sqrt(3.0))
_NODES = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])  # (8,3): x fastest
_GPTS = np.array([[x, y, z] for z in _g for y in _g for x in _g])  # (8,3)


def _dN_ref():
    d = np.zeros((8, 8, 3))  # [gauss, node, dim]
    for gi, (x, y, z) in enumerate(_GPTS):
        for a, (i, j, k) in enumerate(_NODES):
            fx = x if i else 1 - x
            fy = y if j else 1 - y
            fz = z if k else 1 - z
            sx = 1 if i else -1
            sy = 1 if j else -1
            sz = 1 if k else -1
            d[gi, a] = [sx * fy * fz, fx * sy * fz, fx * fy * sz]
    return d


DN_REF = jnp.asarray(_dN_ref())


def lame(E=10.0, nu=0.3):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def psi_linear(F, lam, mu):
    H = F - jnp.eye(3)
    eps = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    tr = jnp.trace(eps, axis1=-2, axis2=-1)
    return 0.5 * lam * tr**2 + mu * jnp.sum(eps * eps, axis=(-2, -1))


def psi_neohooke(F, lam, mu):
    J = jnp.linalg.det(F)
    I1 = jnp.sum(F * F, axis=(-2, -1))
    lnJ = jnp.log(J)
    return 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ**2


MATERIALS = {"linear": psi_linear, "neohooke": psi_neohooke}


# A solve whose line search can no longer improve a gradient already this far below its start has hit the
# float64 round-off floor (which grows with problem size); it is reported as converged.
STALL_RTOL = 1e-6


def _linesearch(E, gnorm, x, dx, E0, slope, gn):
    """Backtracking Armijo search on E(x + t dx). Near convergence the energy decrease falls below
    float64 resolution of E, so a step whose energy change is within round-off is also accepted when
    it lowers the gradient norm. Returns the step length, or None if no step is acceptable."""
    tol = 64 * np.finfo(np.float64).eps * max(abs(E0), 1e-300)
    t = 1.0
    while t > 1e-6:
        Et = E(x + t * dx)
        if jnp.isfinite(Et):
            Et = float(Et)
            if Et <= E0 + 1e-4 * t * slope:
                return t
            if Et <= E0 + tol and gnorm(x + t * dx) < gn:
                return t
        t *= 0.5
    return None


class Cell:
    """A periodic voxel cell with density rho (n,n,n), axis order (x, y, z) = array axes (0, 1, 2)."""

    def __init__(self, rho, E=10.0, nu=0.3, material="neohooke", wls=None):
        """wls: None, or (beta, rho0) for the Wang-Lazarov-Sigmund void interpolation, e.g. (500, 0.01).
        gamma(rho) is a smooth step: about 0 for rho << rho0 (void, linear strain) and 1 for solid."""
        rho = jnp.asarray(rho, dtype=jnp.float64)
        if wls is None:
            self.gamma = None
        else:
            self._wls = wls
            self.gamma = self._gamma_of(rho)
        self._wls = wls
        self.rho = rho
        self.n = rho.shape[0]
        assert rho.shape == (self.n,) * 3
        self.h = 1.0 / self.n
        self.s = RHO_MIN + rho
        self.lam, self.mu = lame(E, nu)
        self.psi = MATERIALS[material]
        self._Enu, self._mat = (E, nu), material
        self.dN = DN_REF / self.h  # physical gradients
        self.wq = self.h**3 / 8.0  # Gauss weight times element volume (cell volume is 1)
        self._jit()

    # --- discretization -------------------------------------------------------------
    def grad_w(self, w):
        """Displacement gradient at the Gauss points: (8, n, n, n, 3, 3), H[..., i, j] = d w_i / d x_j."""
        # nodal values of element (i,j,k): node a at offset _NODES[a]
        Wa = jnp.stack([jnp.roll(w, shift=tuple(-o for o in off), axis=(0, 1, 2)) for off in _NODES])  # (8,n,n,n,3)
        return jnp.einsum("aijkc,gad->gijkcd", Wa, self.dN)

    def grad_w_g(self, w, g):
        """Displacement gradient at Gauss point g only: (n, n, n, 3, 3). g may be traced."""
        Wa = jnp.stack([jnp.roll(w, shift=tuple(-o for o in off), axis=(0, 1, 2)) for off in _NODES])  # (8,n,n,n,3)
        return jnp.einsum("aijkc,ad->ijkcd", Wa, self.dN[g])

    def _gamma_of(self, rho):
        b, r0 = self._wls
        return (jnp.tanh(b * r0) + jnp.tanh(b * (rho - r0))) / (jnp.tanh(b * r0) + jnp.tanh(b * (1 - r0)))

    def energy(self, w, Fbar, rho=None):
        """Cell energy. rho=None uses the cell's density; passing rho makes both the stiffness scale and
        the void interpolation depend on it (used by the adjoint)."""
        if rho is None:
            s, gamma = self.s, self.gamma
        else:
            s = RHO_MIN + rho
            gamma = None if self._wls is None else self._gamma_of(rho)
        # sum over the 8 Gauss points, each term rematerialized in the backward pass: the AD temporaries of all
        # points at once exceed 4 GB at 96^3, the largest single allocation WSL allows
        def eg(w_, Fbar_, s_, gamma_, g):
            return jnp.sum(s_ * self.psi_eff(Fbar_ + self.grad_w_g(w_, g), gamma_))

        eg_c = jax.checkpoint(eg)
        tot, _ = jax.lax.scan(lambda acc, g: (acc + eg_c(w, Fbar, s, gamma, g), None), jnp.zeros(()), jnp.arange(8))
        return self.wq * tot

    def psi_eff(self, F, gamma):
        """Strain energy density at the Gauss points (before the stiffness scale s), with the void
        interpolation when gamma is given."""
        if gamma is None:
            return self.psi(F, self.lam, self.mu)
        # Wang, Lazarov & Sigmund (2014): void elements see the linear strain measure, so they cannot
        # invert; psi = psi_nl(F_g) - psi_lin(F_g) + psi_lin(F), F_g = I + gamma (F - I).
        g = gamma[..., None, None]  # broadcasts against F of shape (..., n, n, n, 3, 3)
        Fg = jnp.eye(3) + g * (F - jnp.eye(3))
        return self.psi(Fg, self.lam, self.mu) - psi_linear(Fg, self.lam, self.mu) + psi_linear(F, self.lam, self.mu)

    def tangent_g(self, w, Fbar, g):
        """Material tangent at Gauss point g, scaled by s_e and the Gauss weight:
        A[i, j, k, a, b, c, d] = wq s_e d^2 psi_eff / dF_ab dF_cd, shape (n, n, n, 3, 3, 3, 3). g may be traced.
        The energy Hessian is H = sum_g L_g^T A_g L_g with L_g the linear map from (w, Fbar) to F at point g."""
        F = Fbar + self.grad_w_g(w, g)
        gam = jnp.ones(self.s.shape) if self.gamma is None else self.gamma
        if self.gamma is None:
            f = lambda F1, g1: self.psi(F1, self.lam, self.mu)  # noqa: E731
        else:
            f = lambda F1, g1: self.psi_eff(F1[None, None, None], g1[None, None, None])[0, 0, 0]  # noqa: E731
        hf = jax.vmap(jax.hessian(f, argnums=0))
        sc = (self.wq * self.s).reshape(-1, 1, 1, 1, 1)
        # each point's 3x3x3x3 block flattened to 81, so XLA does not transpose the whole array
        return (sc * hf(F.reshape(-1, 3, 3), gam.reshape(-1))).reshape(-1, 81).reshape(F.shape + (3, 3))

    def tangent(self, w, Fbar):
        """All 8 Gauss points stacked: (8, n, n, n, 3, 3, 3, 3) (small grids only; Mixed keeps them apart)."""
        return jnp.stack([self.tangent_g(w, Fbar, g) for g in range(8)])

    # --- solver ---------------------------------------------------------------------
    def _jit(self):
        self._E = jax.jit(self.energy)
        self._g = jax.jit(jax.grad(self.energy, argnums=0))
        self._hvp_raw = lambda w, Fb, v: jax.jvp(lambda x: jax.grad(self.energy, argnums=0)(x, Fb), (w,), (v,))[1]
        self._hvp = jax.jit(self._hvp_raw)
        self._P = jax.jit(jax.grad(self.energy, argnums=1))
        self._diag = self._jacobi()

    def _jacobi(self):
        """Diagonal of the linear-elastic stiffness: sum over the 8 elements at each node of s_e * Ke_diag."""
        lam, mu = self.lam, self.mu

        def e_one(u):  # one unit-scale element, u (8,3)
            H = jnp.einsum("ac,gad->gcd", u, self.dN)
            return self.wq * jnp.sum(psi_linear(jnp.eye(3) + H, lam, mu))

        Kd = jnp.diag(jax.hessian(lambda x: e_one(x.reshape(8, 3)))(jnp.zeros(24))).reshape(8, 3)
        d = jnp.zeros((self.n,) * 3 + (3,))
        for a, off in enumerate(_NODES):
            # element at node - off has this node as its local node a
            d = d + jnp.roll(self.s, shift=tuple(off), axis=(0, 1, 2))[..., None] * Kd[a]
        return d

    def _cg(self, w, Fbar, b, tol, maxit):
        """Jacobi-preconditioned CG for H x = b on the mean-free subspace, as one compiled loop.
        Stops early on negative curvature (then returns the iterate so far, or the preconditioned
        residual on the first iteration). Returns (x, iterations, converged)."""
        if not hasattr(self, "_cg_jit"):
            self._cg_jit = jax.jit(self._cg_impl, static_argnums=(4,))
        x, it, conv = self._cg_jit(w, jnp.asarray(Fbar, dtype=jnp.float64), b, tol, int(maxit))
        return x, int(it), bool(conv)

    def _cg_impl(self, w, Fbar, b, tol, maxit):
        Minv = 1.0 / self._diag

        def proj(v):
            return v - jnp.mean(v, axis=(0, 1, 2))

        b = proj(b)
        bn = jnp.linalg.norm(b) + 1e-300
        z0 = proj(Minv * b)
        # state: x, r, z, p, rz, it, done, conv
        init = (jnp.zeros_like(b), b, z0, z0, jnp.vdot(b, z0), 0, False, False)

        def cond(st):
            return (~st[6]) & (st[5] < maxit)

        def body(st):
            x, r, z, p, rz, it, done, conv = st
            Ap = proj(self._hvp_raw(w, Fbar, p))
            pAp = jnp.vdot(p, Ap)
            neg = pAp <= 0
            a = jnp.where(neg, 0.0, rz / jnp.where(neg, 1.0, pAp))
            x1 = x + a * p
            r1 = r - a * Ap
            conv1 = jnp.linalg.norm(r1) <= tol * bn
            z1 = proj(Minv * r1)
            rz1 = jnp.vdot(r1, z1)
            p1 = z1 + (rz1 / jnp.where(rz == 0, 1.0, rz)) * p
            # negative curvature on the first step: return the preconditioned residual direction
            x1 = jnp.where(neg & (it == 0), z0, jnp.where(neg, x, x1))
            return (x1, r1, z1, p1, rz1, it + 1, neg | conv1, conv1 & ~neg)

        x, r, z, p, rz, it, done, conv = jax.lax.while_loop(cond, body, init)
        return x, it, conv

    def solve(self, Fbar, w0=None, rtol=1e-10, maxnewton=30, cg_tol=1e-6, cg_maxit=5000, verbose=False):
        """Newton-CG with backtracking line search. Returns w and a log dict."""
        Fbar = jnp.asarray(Fbar, dtype=jnp.float64)
        w = jnp.zeros((self.n,) * 3 + (3,)) if w0 is None else jnp.asarray(w0)
        g0 = None
        log = dict(newton=0, cg=0, converged=False, gnorm=[])
        for k in range(maxnewton):
            g = self._g(w, Fbar)
            g = g - jnp.mean(g, axis=(0, 1, 2))
            gn = float(jnp.linalg.norm(g))
            if g0 is None:
                g0 = max(gn, 1e-300)
                # scale reference: the gradient of the affine (w = 0) state at this Fbar
            log["gnorm"].append(gn)
            if verbose:
                print(f"  newton {k}: |g| {gn:.3e}")
            if gn <= rtol * g0 or gn < 1e-14:
                log["converged"] = True
                break
            dw, its, ok = self._cg(w, Fbar, -g, cg_tol, cg_maxit)
            log["cg"] += its
            E0 = float(self._E(w, Fbar))
            slope = float(jnp.vdot(g, dw))
            if slope >= 0:  # not a descent direction
                dw = -g / self._diag
                slope = float(jnp.vdot(g, dw))
            t = _linesearch(lambda x: self._E(x, Fbar), self._gnorm_fn(Fbar), w, dw, E0, slope, gn)
            log["newton"] = k + 1
            if t is None:  # keep the last good state
                log["linesearch_failed"] = True
                if gn <= STALL_RTOL * g0:  # stalled at the round-off floor: accept as converged
                    log["converged"] = log["roundoff_floor"] = True
                break
            w = w + t * dw
        return w, log

    def _gnorm_fn(self, Fbar):
        def f(w):
            g = self._g(w, Fbar)
            return float(jnp.linalg.norm(g - jnp.mean(g, axis=(0, 1, 2))))
        return f

    def mean_stress(self, w, Fbar):
        """First Piola-Kirchhoff average Pbar = dWbar/dFbar (cell volume 1)."""
        return self._P(w, jnp.asarray(Fbar, dtype=jnp.float64))

    def grad_rho(self, w, Fbar, G, cg_tol=1e-10, cg_maxit=10000):
        """Adjoint gradient of q = <G, Pbar> (G a 3x3 weight) with respect to rho, at the equilibrium w.

        r(w, rho) = dE/dw = 0 defines w(rho). q(w, rho) = <G, dE/dFbar>. With H lam = dq/dw:
        dq/drho = dq/drho|_w - d/drho <lam, r(w, rho)>, through both the stiffness scale and the void
        interpolation. One adjoint CG solve per q.
        """
        Fbar = jnp.asarray(Fbar, dtype=jnp.float64)
        G = jnp.asarray(G, dtype=jnp.float64)

        def q(w_, r_):
            return jnp.sum(G * jax.grad(self.energy, argnums=1)(w_, Fbar, r_))

        dq_dw = jax.grad(q, argnums=0)(w, self.rho)
        lam, its, ok = self._cg(w, Fbar, dq_dw, cg_tol, cg_maxit)
        dq_dr = jax.grad(q, argnums=1)(w, self.rho)
        coupling = jax.grad(lambda r_: jnp.vdot(lam, jax.grad(self.energy, argnums=0)(w, Fbar, r_)))(self.rho)
        return dq_dr - coupling, dict(cg=its, converged=ok)

    def with_rho(self, rho):
        """Same material and grid, new density (recompiles; fine for tests)."""
        return Cell(np.asarray(rho), *self._Enu, material=self._mat, wls=self._wls)


# --- linear homogenized stiffness, homo3d Voigt convention ----------------------------
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]  # xx yy zz yz xz xy (engineering shear)


def unit_strain(I):
    H = np.zeros((3, 3))
    i, j = VOIGT[I]
    if i == j:
        H[i, i] = 1.0
    else:
        H[i, j] = H[j, i] = 0.5
    return H


def C_linear(rho, E=10.0, nu=0.3, rtol=1e-10, cg_tol=1e-10, columns=range(6)):
    """6x6 homogenized stiffness from 6 linear cell problems (the small-strain limit).
    columns: the unit strains to solve (others stay 0), e.g. (0, 3) for a cubic C11, C12, C44."""
    cell = Cell(rho, E, nu, material="linear")
    C = np.zeros((6, 6))
    logs = []
    for I in columns:
        Fbar = np.eye(3) + unit_strain(I)
        w, log = cell.solve(Fbar, rtol=rtol, cg_tol=cg_tol, maxnewton=3)
        P = np.asarray(cell.mean_stress(w, Fbar))
        sig = 0.5 * (P + P.T)  # linear: P is the Cauchy stress
        C[:, I] = [sig[a, b] for a, b in VOIGT]
        logs.append(log)
    return C, logs


class Mixed:
    """Mixed macroscopic control: the listed components of Fbar are unknowns with zero conjugate stress
    (e.g. free=[(1, 1), (2, 2)] with Fbar_11 prescribed gives uniaxial stress along x).
    Solves min over (w, Fbar_free) of the cell energy; the stationarity in Fbar_free is Pbar_free = 0.
    Off-diagonal components should not be freed (rigid rotations would make the problem singular)."""

    def __init__(self, cell, free, tangent="jvp", forcing="fixed"):
        """tangent: "jvp" (Hessian-vector products by forward-over-reverse AD, recomputed in every CG
        iteration) or "stored" (the Gauss-point tangent is formed once per Newton step; same operator, far
        fewer float64 operations per product). forcing: "fixed" (CG tolerance cg_tol) or "ew"
        (Eisenstat-Walker choice 2: loose CG far from the solution, cg_tol near it)."""
        assert tangent in ("jvp", "stored") and forcing in ("fixed", "ew")
        self.tangent, self.forcing = tangent, forcing
        self.c = cell
        self.free = list(free)
        self.n = cell.n
        self.nw = 3 * self.n**3
        self.idx = tuple(np.array(self.free).T)
        n, nw, idx = self.n, self.nw, self.idx

        def unpack(z, Fb):
            return z[:nw].reshape(n, n, n, 3), Fb.at[idx].set(z[nw:])

        def Ez(z, Fb):
            w, F = unpack(z, Fb)
            return cell.energy(w, F)

        self.unpack = jax.jit(unpack)
        self._E = jax.jit(Ez)
        self._g = jax.jit(jax.grad(Ez))
        self._hvp_raw = lambda z, Fb, v: jax.jvp(lambda x: jax.grad(Ez)(x, Fb), (z,), (v,))[1]
        self._cg = jax.jit(self._cg_impl, static_argnums=(5,))
        k = len(self.free)
        Esel = np.zeros((k, 3, 3))  # free component j -> unit 3x3 matrix
        for j, (a, b) in enumerate(self.free):
            Esel[j, a, b] = 1.0
        Esel = jnp.asarray(Esel)

        def Lmap(v):  # (w, Fbar_free) increment -> F increment at the Gauss points (linear)
            dF = jnp.tensordot(v[nw:], Esel, axes=1)
            return dF + cell.grad_w(v[:nw].reshape(n, n, n, 3))

        LT = jax.linear_transpose(Lmap, jnp.zeros(nw + k))

        def hvp_A(A, v):  # A: tuple of the 8 Gauss-point tangents
            dF = Lmap(v)
            dP = jnp.stack([jnp.einsum("ijkabcd,ijkcd->ijkab", A[g], dF[g]) for g in range(8)])
            return LT(dP)[0]

        self._hvp_A = hvp_A
        # one compiled call per Gauss point, kept as 8 arrays: the whole tangent is 4.3 GB at 96^3, and WSL refuses
        # single allocations of 4 GB or more
        tg = jax.jit(lambda z, Fb, g: cell.tangent_g(*unpack(z, Fb), g))
        self._tangent = lambda z, Fb: tuple(tg(z, Fb, g) for g in range(8))
        self._cgA = jax.jit(self._cgA_impl, static_argnums=(5,))

    def _proj(self, v):
        w = v[: self.nw].reshape(self.n, self.n, self.n, 3)
        w = w - jnp.mean(w, axis=(0, 1, 2))
        return jnp.concatenate([w.ravel(), v[self.nw:]])

    def _cg_impl(self, z, Fb, b, Minv, tol, maxit):
        proj = self._proj
        b = proj(b)
        bn = jnp.linalg.norm(b) + 1e-300
        z0 = proj(Minv * b)
        init = (jnp.zeros_like(b), b, z0, z0, jnp.vdot(b, z0), 0, False, False)

        def cond(st):
            return (~st[6]) & (st[5] < maxit)

        def body(st):
            x, r, zz, p, rz, it, done, conv = st
            Ap = proj(self._hvp_raw(z, Fb, p))
            pAp = jnp.vdot(p, Ap)
            neg = pAp <= 0
            a = jnp.where(neg, 0.0, rz / jnp.where(neg, 1.0, pAp))
            x1, r1 = x + a * p, r - a * Ap
            conv1 = jnp.linalg.norm(r1) <= tol * bn
            z1 = proj(Minv * r1)
            rz1 = jnp.vdot(r1, z1)
            p1 = z1 + (rz1 / jnp.where(rz == 0, 1.0, rz)) * p
            x1 = jnp.where(neg & (it == 0), z0, jnp.where(neg, x, x1))
            return (x1, r1, z1, p1, rz1, it + 1, neg | conv1, conv1 & ~neg)

        x, r, zz, p, rz, it, done, conv = jax.lax.while_loop(cond, body, init)
        return x, it, conv

    def _cgA_impl(self, A, Fb, b, Minv, tol, maxit):
        """CG with the stored tangent A (same iteration as _cg_impl)."""
        proj = self._proj
        b = proj(b)
        bn = jnp.linalg.norm(b) + 1e-300
        z0 = proj(Minv * b)
        init = (jnp.zeros_like(b), b, z0, z0, jnp.vdot(b, z0), 0, False, False)

        def cond(st):
            return (~st[6]) & (st[5] < maxit)

        def body(st):
            x, r, zz, p, rz, it, done, conv = st
            Ap = proj(self._hvp_A(A, p))
            pAp = jnp.vdot(p, Ap)
            neg = pAp <= 0
            a = jnp.where(neg, 0.0, rz / jnp.where(neg, 1.0, pAp))
            x1, r1 = x + a * p, r - a * Ap
            conv1 = jnp.linalg.norm(r1) <= tol * bn
            z1 = proj(Minv * r1)
            rz1 = jnp.vdot(r1, z1)
            p1 = z1 + (rz1 / jnp.where(rz == 0, 1.0, rz)) * p
            x1 = jnp.where(neg & (it == 0), z0, jnp.where(neg, x, x1))
            return (x1, r1, z1, p1, rz1, it + 1, neg | conv1, conv1 & ~neg)

        x, r, zz, p, rz, it, done, conv = jax.lax.while_loop(cond, body, init)
        return x, it, conv

    def _minv(self, z, Fb):
        """Jacobi: linear element diagonal for w; Hessian diagonal (2 HVPs) for the macro unknowns."""
        if not hasattr(self, "_minv_jit"):
            self._minv_jit = jax.jit(self._minv_impl)
        return self._minv_jit(z, Fb)

    def _minv_impl(self, z, Fb):
        k = len(self.free)
        dmac = []
        for j in range(k):
            e = jnp.zeros(self.nw + k).at[self.nw + j].set(1.0)
            dmac.append(self._hvp_raw(z, Fb, e)[self.nw + j])
        dmac = jnp.abs(jnp.array(dmac)) + 1e-12
        return jnp.concatenate([1.0 / self.c._diag.ravel(), 1.0 / dmac])

    def solve(self, Fbar, z0=None, rtol=1e-9, maxnewton=25, cg_tol=1e-6, cg_maxit=20000, g_ref=None):
        """Fbar: prescribed components (the free ones are the initial guess if z0 is None).
        g_ref: gradient norm that rtol refers to (default: the first one of this call; pass log["g0"] of an
        earlier call to continue a solve). Returns (w, Fbar_full, log)."""
        Fb = jnp.asarray(Fbar, dtype=jnp.float64)
        if z0 is None:
            z = jnp.concatenate([jnp.zeros(self.nw), Fb[self.idx]])
        else:
            z = z0
        g0, log = g_ref, dict(newton=0, cg=0, converged=False, gnorm=[])
        eta, gprev = None, None
        for k in range(maxnewton):
            g = self._proj(self._g(z, Fb))
            gn = float(jnp.linalg.norm(g))
            g0 = gn if g0 is None else g0
            log["gnorm"].append(gn)
            if gn <= rtol * max(g0, 1e-300) or gn < 1e-14:
                log["converged"] = True
                break
            tol = cg_tol
            if self.forcing == "ew":  # Eisenstat & Walker (1996), choice 2, gamma 0.9, alpha 2, eta_max 0.1
                e_new = 0.1 if gprev is None else 0.9 * (gn / gprev) ** 2
                if eta is not None and 0.9 * eta**2 > 0.1:
                    e_new = max(e_new, 0.9 * eta**2)
                eta = min(e_new, 0.1)
                # never tighter than cg_tol, and no tighter than needed to reach rtol
                tol = max(eta, cg_tol, 0.5 * rtol * g0 / gn) if gprev is not None else max(eta, cg_tol)
            gprev = gn
            Minv = self._minv(z, Fb)
            if self.tangent == "stored":
                dz, its, ok = self._cgA(self._tangent(z, Fb), Fb, -g, Minv, tol, int(cg_maxit))
            else:
                dz, its, ok = self._cg(z, Fb, -g, Minv, tol, int(cg_maxit))
            log["cg"] += int(its)
            if VERBOSE:
                print(f"    newton {k}: |g| {gn:.2e}  cg {int(its)} (tol {tol:.1e}, {'converged' if bool(ok) else 'stopped'})",
                      flush=True)
            E0, slope = float(self._E(z, Fb)), float(jnp.vdot(g, dz))
            if slope >= 0:
                dz = -Minv * g
                slope = float(jnp.vdot(g, dz))
            t = _linesearch(lambda x: self._E(x, Fb), lambda x: float(jnp.linalg.norm(self._proj(self._g(x, Fb)))),
                            z, dz, E0, slope, gn)
            log["newton"] = k + 1
            if t is None:
                log["linesearch_failed"] = True
                if gn <= STALL_RTOL * g0:  # stalled at the round-off floor: accept as converged
                    log["converged"] = log["roundoff_floor"] = True
                break
            z = z + t * dz
        w, F = self.unpack(z, Fb)
        log["z"], log["g0"] = z, g0
        return w, F, log
