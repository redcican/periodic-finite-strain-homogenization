#include "fs_krylov.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace fs {

void swap(BVec& a, BVec& b) {
	std::swap(a.w, b.w);
	for (int k = 0; k < 9; k++) std::swap(a.f[k], b.f[k]);
}

BVec Space::alloc() const {
	BVec v;
	v.w = dalloc(m);
	return v;
}

void Space::release(BVec& v) const {
	dfree(v.w);
	v.w = nullptr;
}

double Space::dot(const BVec& a, const BVec& b) const {
	double d = vdot(a.w, b.w, m);
	for (int k = 0; k < nF; k++) d += a.f[k] * b.f[k];
	return d;
}

double Space::norm(const BVec& a) const { return std::sqrt(std::max(0.0, dot(a, a))); }

void Space::axpy(double a, const BVec& x, BVec& y) const {
	vaxpy(a, x.w, y.w, m);
	for (int k = 0; k < nF; k++) y.f[k] += a * x.f[k];
}

void Space::axpby(double a, const BVec& x, double b, BVec& y) const {
	vaxpby(a, x.w, b, y.w, m);
	for (int k = 0; k < nF; k++) y.f[k] = a * x.f[k] + b * y.f[k];
}

void Space::copy(const BVec& x, BVec& y) const {
	vcopy(x.w, y.w, m);
	for (int k = 0; k < 9; k++) y.f[k] = x.f[k];
}

void Space::zero(BVec& y) const {
	vzero(y.w, m);
	for (int k = 0; k < 9; k++) y.f[k] = 0;
}

void Space::scale(double a, BVec& y) const {
	vscale(a, y.w, m);
	for (int k = 0; k < nF; k++) y.f[k] *= a;
}

namespace {
struct Pool {
	const Space& sp;
	BVec v[8];
	int n;
	Pool(const Space& s, int k) : sp(s), n(k) {
		for (int i = 0; i < n; i++) v[i] = sp.alloc();
	}
	~Pool() {
		for (int i = 0; i < n; i++) sp.release(v[i]);
	}
};
}  // namespace

KResult pcg(const Space& sp, LinOp& A, const BVec& b, BVec& x, double tol, int maxit) {
	KResult res;
	Pool P(sp, 4);
	BVec &r = P.v[0], &z = P.v[1], &p = P.v[2], &q = P.v[3];
	double bn = sp.norm(b);
	if (bn == 0) {
		sp.zero(x);
		res.status = KStatus::Converged;
		res.relres = 0;
		return res;
	}
	A.apply(x, q);
	sp.copy(b, r);
	sp.axpy(-1.0, q, r);
	double rn = sp.norm(r);
	res.relres = rn / bn;
	if (rn <= tol * bn) {
		res.status = KStatus::Converged;
		return res;
	}
	int restarts = 0;
	while (true) {
		if (!A.prec(r, z)) {
			res.status = KStatus::PrecFail;
			return res;
		}
		double rz = sp.dot(r, z);
		if (!(rz > 0)) {
			res.status = KStatus::PrecFail;
			return res;
		}
		sp.copy(z, p);
		bool converged = false;
		while (res.iters < maxit) {
			A.apply(p, q);
			double pq = sp.dot(p, q);
			if (!(pq > 0)) {
				res.status = KStatus::Indefinite;
				return res;
			}
			double alpha = rz / pq;
			sp.axpy(alpha, p, x);
			sp.axpy(-alpha, q, r);
			res.iters++;
			rn = sp.norm(r);
			res.relres = rn / bn;
			if (rn <= tol * bn) {
				converged = true;
				break;
			}
			if (!A.prec(r, z)) {
				res.status = KStatus::PrecFail;
				return res;
			}
			double rz1 = sp.dot(r, z);
			if (!(rz1 > 0)) {
				res.status = KStatus::PrecFail;
				return res;
			}
			sp.axpby(1.0, z, rz1 / rz, p);
			rz = rz1;
		}
		if (!converged) {
			res.status = KStatus::MaxIter;
			return res;
		}
		// the recursive residual drifts in FP32: confirm with the true residual
		A.apply(x, q);
		sp.copy(b, r);
		sp.axpy(-1.0, q, r);
		rn = sp.norm(r);
		res.relres = rn / bn;
		if (rn <= 1.5 * tol * bn || restarts >= 3 || res.iters >= maxit) {
			res.status = rn <= 1.5 * tol * bn ? KStatus::Converged : KStatus::MaxIter;
			return res;
		}
		restarts++;
	}
}

KResult minres(const Space& sp, LinOp& A, const BVec& b, BVec& x, double tol, int maxit) {
	KResult res;
	Pool P(sp, 7);
	BVec &r1 = P.v[0], &r2 = P.v[1], &y = P.v[2], &v = P.v[3], &w = P.v[4], &w1 = P.v[5], &w2 = P.v[6];
	double bn = sp.norm(b);
	if (bn == 0) {
		sp.zero(x);
		res.status = KStatus::Converged;
		res.relres = 0;
		return res;
	}
	double target = tol;  // target of the preconditioned residual estimate, tightened if needed
	for (int restart = 0; restart < 4; restart++) {
		A.apply(x, y);
		sp.copy(b, r1);
		sp.axpy(-1.0, y, r1);
		double rn = sp.norm(r1);
		res.relres = rn / bn;
		if (rn <= tol * bn) {
			res.status = KStatus::Converged;
			return res;
		}
		if (!A.prec(r1, y)) {
			res.status = KStatus::PrecFail;
			return res;
		}
		double b1sq = sp.dot(r1, y);
		if (b1sq < 0) {
			res.status = KStatus::PrecFail;
			return res;
		}
		double beta1 = std::sqrt(b1sq);
		if (beta1 == 0) {
			res.status = KStatus::Converged;
			return res;
		}
		sp.copy(r1, r2);
		sp.zero(w);
		sp.zero(w2);
		double oldb = 0, beta = beta1, dbar = 0, epsln = 0, phibar = beta1, cs = -1, sn = 0;
		bool hit = false;
		for (int itn = 1; res.iters < maxit; itn++) {
			sp.copy(y, v);
			sp.scale(1.0 / beta, v);
			A.apply(v, y);
			if (itn >= 2) sp.axpy(-beta / oldb, r1, y);
			double alfa = sp.dot(v, y);
			sp.axpy(-alfa / beta, r2, y);
			swap(r1, r2);  // r1 <- r2
			sp.copy(y, r2);  // r2 <- y
			if (!A.prec(r2, y)) {
				res.status = KStatus::PrecFail;
				return res;
			}
			oldb = beta;
			double bsq = sp.dot(r2, y);
			if (bsq < 0) {
				res.status = KStatus::PrecFail;
				return res;
			}
			beta = std::sqrt(bsq);
			double oldeps = epsln;
			double delta = cs * dbar + sn * alfa;
			double gbar = sn * dbar - cs * alfa;
			epsln = sn * beta;
			dbar = -cs * beta;
			double gamma = std::max(std::sqrt(gbar * gbar + beta * beta), 1e-300);
			cs = gbar / gamma;
			sn = beta / gamma;
			double phi = cs * phibar;
			phibar = sn * phibar;
			// w <- (v - oldeps w1 - delta w2) / gamma with w1 <- w2, w2 <- w
			swap(w1, w2);  // w1 <- old w2
			swap(w2, w);   // w2 <- old w, w <- old w1 (overwritten next)
			sp.copy(v, w);
			sp.axpy(-oldeps, w1, w);
			sp.axpy(-delta, w2, w);
			sp.scale(1.0 / gamma, w);
			sp.axpy(phi, w, x);
			res.iters++;
			if (phibar <= target * beta1 || beta == 0) {
				hit = true;
				break;
			}
		}
		// check the true residual in the Euclidean norm
		A.apply(x, y);
		sp.copy(b, r1);
		sp.axpy(-1.0, y, r1);
		rn = sp.norm(r1);
		res.relres = rn / bn;
		if (rn <= tol * bn) {
			res.status = KStatus::Converged;
			return res;
		}
		if (!hit || res.iters >= maxit) break;
		target *= std::max(0.05, std::min(0.5, tol * bn / rn));
	}
	res.status = KStatus::MaxIter;
	return res;
}

}  // namespace fs
