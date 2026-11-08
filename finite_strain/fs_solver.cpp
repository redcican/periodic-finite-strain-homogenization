#include "fs_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace fs {

const char* step_fail_name(StepFail f) {
	switch (f) {
	case StepFail::None: return "none";
	case StepFail::NewtonMaxIter: return "newton-maxit";
	case StepFail::LineSearch: return "line-search";
	case StepFail::InvalidJ: return "J_gamma<=0";
	case StepFail::PrecNotPD: return "preconditioner-not-pd";
	}
	return "?";
}

namespace {
const double kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const double kU32 = 5.9604644775390625e-08;  // 2^-24

// In-place Cholesky of an m x m row-major SPD matrix (lower factor); false if not positive definite.
bool chol(double* A, int m) {
	for (int j = 0; j < m; j++) {
		double d = A[j * m + j];
		for (int k = 0; k < j; k++) d -= A[j * m + k] * A[j * m + k];
		if (!(d > 0)) return false;
		d = std::sqrt(d);
		A[j * m + j] = d;
		for (int i = j + 1; i < m; i++) {
			double v = A[i * m + j];
			for (int k = 0; k < j; k++) v -= A[i * m + k] * A[j * m + k];
			A[i * m + j] = v / d;
		}
	}
	return true;
}

void chol_solve(const double* L, int m, const double* b, double* x) {
	double y[9];
	for (int i = 0; i < m; i++) {
		double v = b[i];
		for (int k = 0; k < i; k++) v -= L[i * m + k] * y[k];
		y[i] = v / L[i * m + i];
	}
	for (int i = m - 1; i >= 0; i--) {
		double v = y[i];
		for (int k = i + 1; k < m; k++) v -= L[k * m + i] * x[k];
		x[i] = v / L[i * m + i];
	}
}
}  // namespace

Solver::Solver(int n_, const std::vector<float>& rho, const SolverOptions& o) : n(n_), opt(o) {
	cell_alloc(cell, n, make_material(o.E, o.nu, o.beta, o.rho0, o.p, o.rhomin));
	mg.init(n);
	size_t m3 = 3 * cell.ne;
	sp = Space(m3, 0);
	spK = Space(m3, 0);
	u = sp.alloc();
	g = sp.alloc();
	trial = sp.alloc();
	gt = sp.alloc();
	du = sp.alloc();
	rhs = sp.alloc();
	tmp1 = sp.alloc();
	tmp2 = sp.alloc();
	Z0w = dalloc(m3);
	sqZ0w = dalloc(m3);
	isqZ0w = dalloc(m3);
	for (int k = 0; k < 9; k++) Z0F[k] = 1;
	setDensity(rho);
	resetState();
}

Solver::~Solver() {
	for (BVec* v : {&u, &g, &trial, &gt, &du, &rhs, &tmp1, &tmp2}) sp.release(*v);
	for (auto& x : eig.X) sp.release(x);
	for (float* c : chi) dfree(c);
	dfree(Z0w);
	dfree(sqZ0w);
	dfree(isqZ0w);
	cell_free(cell);
}

void Solver::setFree(const std::vector<int>& comps) {
	if (comps == freeComps && sp.nF == int(comps.size())) return;  // keeps the LOBPCG starting block
	freeComps = comps;
	sp.nF = int(comps.size());
	scalingReady = false;
	for (auto& x : eig.X) sp.release(x);
	eig.X.clear();
	eig.init = false;
	precReady = false;
}

void Solver::setDensity(const std::vector<float>& rho) {
	if (rho.size() != cell.ne) {
		std::fprintf(stderr, "density has %zu values, expected %zu\n", rho.size(), cell.ne);
		std::exit(1);
	}
	rhoHost = rho;
	cell_set_density(cell, rho.data());
	scalingReady = false;
	precReady = false;
}

void Solver::resetState() {
	sp.zero(u);
	std::memcpy(Fbar, kIdentity, sizeof(Fbar));
	for (int k = 0; k < sp.nF; k++) u.f[k] = kIdentity[freeComps[k]];
	valid = false;
}

void Solver::setPrescribed(const double F[9]) { std::memcpy(Fbar, F, sizeof(Fbar)); }

void Solver::fullF(const BVec& state, double F[9]) const {
	std::memcpy(F, Fbar, 9 * sizeof(double));
	for (int k = 0; k < sp.nF; k++) F[freeComps[k]] = state.f[k];
}

bool Solver::evaluate(const BVec& state, BVec& grad, double P[9], EnergyOut& e) {
	double F[9], H[9];
	fullF(state, F);
	for (int c = 0; c < 9; c++) H[c] = F[c] - kIdentity[c];
	cell_cache_update(cell, state.w, H);
	cell_energy(cell, e);
	if (e.badJ > 0 || !std::isfinite(e.energy)) return false;
	cell_residual(cell, grad.w, P);
	for (int k = 0; k < sp.nF; k++) grad.f[k] = P[freeComps[k]];
	for (int c = 0; c < 9; c++)
		if (!std::isfinite(P[c])) return false;
	return true;
}

bool Solver::evaluateCurrent() {
	valid = evaluate(u, g, Pbar, en);
	return valid;
}

void Solver::apply(const BVec& x, BVec& y) {
	double dF[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, t[9];
	for (int k = 0; k < sp.nF; k++) dF[freeComps[k]] = x.f[k];
	cell_product(cell, x.w, dF, y.w, sp.nF ? t : nullptr);
	for (int k = 0; k < sp.nF; k++) y.f[k] = t[freeComps[k]];
}

bool Solver::prec(const BVec& r, BVec& z) {
	if (!precReady) return false;
	mg.vcycle(r.w, z.w);
	if (sp.nF) chol_solve(Ds, sp.nF, r.f, z.f);
	return true;
}

bool Solver::buildPreconditioner(Prec p, float sigma) {
	precReady = false;
	if (sigma > 0.f && !scalingReady) initScaling();
	bool ok = mg.update(cell, p, sigma, sigma > 0.f ? Z0w : nullptr);
	int nF = sp.nF;
	if (nF) {
		double D[81], t[9];
		for (int a = 0; a < nF; a++) {
			double dF[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
			dF[freeComps[a]] = 1;
			cell_product(cell, nullptr, dF, tmp1.w, t);
			for (int b = 0; b < nF; b++) D[b * nF + a] = t[freeComps[b]];
		}
		for (int a = 0; a < nF; a++)
			for (int b = 0; b < a; b++) D[a * nF + b] = D[b * nF + a] = 0.5 * (D[a * nF + b] + D[b * nF + a]);
		for (int a = 0; a < nF; a++) D[a * nF + a] += sigma * Z0F[a];
		std::memcpy(Ds, D, sizeof(double) * nF * nF);
		if (!chol(Ds, nF)) {
			ok = false;
			if (std::getenv("FS_DEBUG")) std::fprintf(stderr, "border block not positive definite (D11 = %g)\n", D[0]);
		}
	}
	precReady = ok;
	sigmaNow = sigma;
	return ok;
}

void Solver::initScaling() {
	double H0[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	cell_cache_update(cell, nullptr, H0);
	float* Dtmp = dalloc(9 * cell.ne);
	cell_diag_blocks(cell, nullptr, 0.f, Dtmp, Z0w);
	dfree(Dtmp);
	size_t m3 = 3 * cell.ne;
	vsqrt(Z0w, sqZ0w, m3);
	vrsqrt(Z0w, isqZ0w, m3);
	for (int k = 0; k < sp.nF; k++) {
		double dF[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, t[9];
		dF[freeComps[k]] = 1;
		cell_product(cell, nullptr, dF, tmp1.w, t);
		Z0F[k] = t[freeComps[k]] > 0 ? t[freeComps[k]] : 1.0;
	}
	scalingReady = true;
	precReady = false;
	evaluateCurrent();  // restore the cache of the current state
}

bool Solver::prepareShifted(Prec p, float& sigma, bool first) {
	if (!(first && precReady && sigma == sigmaNow)) {
		int tries = 0;
		while (!buildPreconditioner(p, sigma)) {
			if (++tries > 12) return false;
			sigma = std::max(2.f * sigma, float(opt.sigmaFloor));
		}
	}
	// Ritz value of the current tangent; a Ritz value bounds lambda_min from above
	double lam = lambdaMin(eig, 1e-2, 30, true);
	float sNew = float(2.0 * std::max(0.0, -lam));
	if (sNew > sigma) {
		sigma = sNew;
		int tries = 0;
		while (!buildPreconditioner(p, sigma)) {
			if (++tries > 12) return false;
			sigma = std::max(2.f * sigma, float(opt.sigmaFloor));
		}
	}
	if (opt.verbose >= 2) std::printf("      MINRES branch: lambda_ritz = %.3e, sigma = %.3e\n", lam, sigma);
	return true;
}

bool Solver::lineSearch(const BVec& dir, double eta) {
	double slope = sp.dot(g, dir);
	double Pi0 = en.energy;
	double gn0 = sp.norm(g);
	double m0 = 0.5 * gn0 * gn0;
	double dPi = 64.0 * kU32 * en.absEnergy;  // rounding level of Pi_h
	bool energyMerit = slope < 0;
	double alpha = 1;
	double Pt[9];
	EnergyOut et;
	for (int h = 0; h <= opt.lsMax; h++, alpha *= 0.5) {
		sp.copy(u, trial);
		sp.axpy(alpha, dir, trial);
		if (!evaluate(trial, gt, Pt, et)) continue;
		double gnt = sp.norm(gt);
		bool acc;
		if (energyMerit) {
			double d = et.energy - Pi0;
			acc = d <= opt.armijo * alpha * slope || (std::fabs(d) <= dPi && gnt < gn0);
		} else {
			acc = 0.5 * gnt * gnt <= (1 - 2 * opt.armijo * alpha * (1 - eta)) * m0;
		}
		if (acc) {
			swap(u, trial);
			swap(g, gt);
			std::memcpy(Pbar, Pt, sizeof(Pbar));
			en = et;
			valid = true;
			return true;
		}
	}
	evaluateCurrent();
	return false;
}

StepStats Solver::newtonSolve(Prec p) {
	StepStats st;
	if (!evaluateCurrent()) {
		st.fail = StepFail::InvalidJ;
		return st;
	}
	double g0 = sp.norm(g);
	st.g0 = g0;
	st.gend = g0;
	if (!(g0 > 0)) return st;
	double gprev = g0, etaPrev = opt.etaMax;
	bool mres = false;
	float sigma = 0.f;

	auto runMinres = [&](double eta) {
		KResult kr;
		for (int tries = 0; tries < 12; tries++) {
			kr = minres(sp, *this, rhs, du, eta, opt.krylovMax);
			st.krylov += kr.iters;
			if (kr.status != KStatus::PrecFail) return kr;
			// the shifted preconditioner is not positive definite: double the shift and restart
			sigma = std::max(2.f * sigma, float(opt.sigmaFloor));
			while (!buildPreconditioner(p, sigma) && tries < 12) {
				sigma *= 2.f;
				tries++;
			}
		}
		return kr;
	};

	for (int j = 0;; j++) {
		double gn = sp.norm(g);
		st.gend = gn;
		if (gn <= opt.newtonTol * g0) break;
		if (j >= opt.newtonMax) {
			st.fail = StepFail::NewtonMaxIter;
			break;
		}
		// Eisenstat-Walker forcing term, second choice
		double eta = opt.etaMax;
		if (j > 0) {
			double etaA = opt.ewGamma * std::pow(gn / gprev, opt.ewAlpha);
			double sg = opt.ewGamma * std::pow(etaPrev, opt.ewAlpha);
			double et = sg <= 0.1 ? etaA : std::max(etaA, sg);
			eta = std::min(opt.etaMax, et);
		}
		eta = std::max(eta, 0.5 * opt.newtonTol * g0 / gn);
		eta = std::min(eta, 0.9);
		etaPrev = eta;

		if (!mres) {
			if (!buildPreconditioner(p, 0.f)) {
				st.fail = StepFail::PrecNotPD;
				break;
			}
		} else if (!prepareShifted(p, sigma, false)) {
			st.fail = StepFail::PrecNotPD;
			break;
		}
		sp.copy(g, rhs);
		sp.scale(-1.0, rhs);
		sp.zero(du);
		KResult kr;
		if (!mres) {
			kr = pcg(sp, *this, rhs, du, eta, opt.krylovMax);
			st.krylov += kr.iters;
			if (kr.status == KStatus::Indefinite) {
				// negative curvature: continue from the current iterate with MINRES for the rest of the step
				mres = true;
				st.minres = true;
				sigma = 0.f;
				if (!prepareShifted(p, sigma, true)) {
					st.fail = StepFail::PrecNotPD;
					break;
				}
				kr = runMinres(eta);
			}
		} else {
			kr = runMinres(eta);
		}
		if (kr.status == KStatus::PrecFail) {
			st.fail = StepFail::PrecNotPD;
			break;
		}
		if (opt.verbose >= 2)
			std::printf("    newton %2d  |g| = %.3e  eta = %.2e  krylov = %4d (%s)  relres = %.2e\n", j, gn / g0, eta,
				kr.iters, mres ? "minres" : "pcg", kr.relres);
		if (!lineSearch(du, eta)) {
			st.fail = StepFail::LineSearch;
			break;
		}
		st.newton++;
		gprev = gn;
	}
	return st;
}

// ---------------------------------------------------------------------------------------------

void LoadPath::F(double t, double Fb[9]) const {
	std::memcpy(Fb, kIdentity, 9 * sizeof(double));
	if (name == "L1") Fb[0] = 1 + t;
	else if (name == "L2") Fb[0] = 1 - t;
	else if (name == "L3") Fb[1] = t;
	else if (name == "L4") Fb[0] = Fb[4] = Fb[8] = 1 + t;
}

double LoadPath::strain(const double Fb[9]) const {
	if (name == "L3") return Fb[1];
	return Fb[0] - 1;
}

LoadPath make_path(const std::string& name, int nsteps, double inc) {
	LoadPath p;
	p.name = name;
	p.nsteps = nsteps;
	if (name == "L1" || name == "L2") {
		p.freeComps = {comp(1, 1), comp(2, 2)};
		p.inc = inc > 0 ? inc : 0.02;
	} else if (name == "L3") {
		p.inc = inc > 0 ? inc : 0.02;
	} else if (name == "L4") {
		p.inc = inc > 0 ? inc : 0.01;
	} else {
		std::fprintf(stderr, "unknown load path %s (L1, L2, L3, L4)\n", name.c_str());
		std::exit(1);
	}
	return p;
}

}  // namespace fs
