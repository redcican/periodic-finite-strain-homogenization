#include "fs_stability.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace fs {

namespace {

// y = sum_i c_i x_i
void lincomb(const Space& sp, BVec& y, const std::vector<const BVec*>& xs, const std::vector<double>& c) {
	sp.zero(y);
	for (size_t i = 0; i < xs.size(); i++)
		if (c[i] != 0) sp.axpy(c[i], *xs[i], y);
}

// Smallest k eigenpairs of the pencil (GA, GB), robust to a rank-deficient GB.
bool rayleigh_ritz(const Eigen::MatrixXd& GA, const Eigen::MatrixXd& GB, int k, Eigen::VectorXd& lam, Eigen::MatrixXd& C) {
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eb(0.5 * (GB + GB.transpose()));
	if (eb.info() != Eigen::Success) return false;
	double bmax = eb.eigenvalues().maxCoeff();
	if (!(bmax > 0)) return false;
	std::vector<int> keep;
	for (int i = 0; i < GB.rows(); i++)
		if (eb.eigenvalues()(i) > 1e-10 * bmax) keep.push_back(i);
	if (int(keep.size()) < k) return false;
	Eigen::MatrixXd V(GB.rows(), keep.size());
	for (size_t j = 0; j < keep.size(); j++) V.col(j) = eb.eigenvectors().col(keep[j]) / std::sqrt(eb.eigenvalues()(keep[j]));
	Eigen::MatrixXd Ar = V.transpose() * (0.5 * (GA + GA.transpose())) * V;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ea(Ar);
	if (ea.info() != Eigen::Success) return false;
	lam = ea.eigenvalues().head(k);
	C = V * ea.eigenvectors().leftCols(k);
	return true;
}

double qmin(const double A[81], const double nv[3]) {
	Eigen::Matrix3d Q;
	for (int i = 0; i < 3; i++)
		for (int k = 0; k < 3; k++) {
			double v = 0;
			for (int j = 0; j < 3; j++)
				for (int l = 0; l < 3; l++) v += A[(3 * i + j) * 9 + 3 * k + l] * nv[j] * nv[l];
			Q(i, k) = v;
		}
	Q = 0.5 * (Q + Q.transpose()).eval();
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Q, Eigen::EigenvaluesOnly);
	return es.eigenvalues()(0);
}

double qmin_angles(const double A[81], double th, double ph) {
	double nv[3] = {std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
	return qmin(A, nv);
}

}  // namespace

double ellipticity_min(const double Abar[81], double nmin[3]) {
	const int ND = 1000;
	const double golden = M_PI * (3 - std::sqrt(5.0));
	std::vector<std::pair<double, int>> vals;
	std::vector<double> th(ND), ph(ND);
	for (int i = 0; i < ND; i++) {
		double z = (i + 0.5) / ND;  // hemisphere z > 0 (Q(n) = Q(-n))
		th[i] = std::acos(z);
		ph[i] = std::fmod(i * golden, 2 * M_PI);
		vals.push_back({qmin_angles(Abar, th[i], ph[i]), i});
	}
	std::sort(vals.begin(), vals.end());
	double best = vals[0].first, bt = th[vals[0].second], bp = ph[vals[0].second];
	for (int s = 0; s < 3; s++) {
		// Nelder-Mead in (theta, phi)
		int i0 = vals[s].second;
		double P[3][2] = {{th[i0], ph[i0]}, {th[i0] + 0.05, ph[i0]}, {th[i0], ph[i0] + 0.05}};
		double F[3];
		for (int v = 0; v < 3; v++) F[v] = qmin_angles(Abar, P[v][0], P[v][1]);
		for (int it = 0; it < 200; it++) {
			int o[3] = {0, 1, 2};
			std::sort(o, o + 3, [&](int a, int b) { return F[a] < F[b]; });
			int b = o[0], m = o[1], w = o[2];
			if (std::fabs(F[w] - F[b]) < 1e-14 * (1 + std::fabs(F[b]))) break;
			double c[2] = {0.5 * (P[b][0] + P[m][0]), 0.5 * (P[b][1] + P[m][1])};
			double r[2] = {2 * c[0] - P[w][0], 2 * c[1] - P[w][1]};
			double fr = qmin_angles(Abar, r[0], r[1]);
			if (fr < F[b]) {
				double e[2] = {3 * c[0] - 2 * P[w][0], 3 * c[1] - 2 * P[w][1]};
				double fe = qmin_angles(Abar, e[0], e[1]);
				if (fe < fr) {
					P[w][0] = e[0]; P[w][1] = e[1]; F[w] = fe;
				} else {
					P[w][0] = r[0]; P[w][1] = r[1]; F[w] = fr;
				}
			} else if (fr < F[m]) {
				P[w][0] = r[0]; P[w][1] = r[1]; F[w] = fr;
			} else {
				double k[2] = {0.5 * (c[0] + P[w][0]), 0.5 * (c[1] + P[w][1])};
				double fk = qmin_angles(Abar, k[0], k[1]);
				if (fk < F[w]) {
					P[w][0] = k[0]; P[w][1] = k[1]; F[w] = fk;
				} else {
					for (int v : {m, w}) {
						P[v][0] = 0.5 * (P[v][0] + P[b][0]);
						P[v][1] = 0.5 * (P[v][1] + P[b][1]);
						F[v] = qmin_angles(Abar, P[v][0], P[v][1]);
					}
				}
			}
		}
		for (int v = 0; v < 3; v++)
			if (F[v] < best) {
				best = F[v];
				bt = P[v][0];
				bp = P[v][1];
			}
	}
	if (nmin) {
		nmin[0] = std::sin(bt) * std::cos(bp);
		nmin[1] = std::sin(bt) * std::sin(bp);
		nmin[2] = std::cos(bt);
	}
	return best;
}

double Solver::lambdaMin(EigState& st, double tol, int maxit, bool usePrec, int* iters, bool* conv) {
	if (!scalingReady) initScaling();
	const int m = st.m;
	const size_t m3 = 3 * cell.ne;
	const int nF = sp.nF;
	double sF[9], isF[9];
	for (int k = 0; k < nF; k++) {
		sF[k] = std::sqrt(Z0F[k]);
		isF[k] = 1.0 / sF[k];
	}
	if (int(st.X.size()) != m) {
		for (auto& x : st.X) sp.release(x);
		st.X.assign(m, BVec());
		for (auto& x : st.X) x = sp.alloc();
		st.init = false;
	}
	std::vector<BVec> AX(m), W(m), AW(m), Pv(m), AP(m), T1(m), T2(m);
	for (int i = 0; i < m; i++) {
		AX[i] = sp.alloc(); W[i] = sp.alloc(); AW[i] = sp.alloc();
		Pv[i] = sp.alloc(); AP[i] = sp.alloc(); T1[i] = sp.alloc(); T2[i] = sp.alloc();
	}
	BVec sc = sp.alloc();
	auto applyHat = [&](const BVec& x, BVec& y) {
		vmul(isqZ0w, x.w, sc.w, m3);
		for (int k = 0; k < nF; k++) sc.f[k] = x.f[k] * isF[k];
		apply(sc, y);
		vmul(isqZ0w, y.w, y.w, m3);
		for (int k = 0; k < nF; k++) y.f[k] *= isF[k];
	};
	auto precHat = [&](const BVec& r, BVec& z) {
		vmul(sqZ0w, r.w, sc.w, m3);
		for (int k = 0; k < nF; k++) sc.f[k] = r.f[k] * sF[k];
		if (!prec(sc, z)) return false;
		vmul(sqZ0w, z.w, z.w, m3);
		for (int k = 0; k < nF; k++) z.f[k] *= sF[k];
		return true;
	};
	auto project = [&](BVec& v) { vproject_weighted(v.w, sqZ0w, cell.ne); };
	auto normalize = [&](BVec& v) {
		double nr = sp.norm(v);
		if (nr > 0) sp.scale(1.0 / nr, v);
	};

	if (!st.init) {
		for (int i = 0; i < m; i++) {
			vrandom(st.X[i].w, m3, st.seed * 977u + unsigned(i));
			for (int k = 0; k < 9; k++) st.X[i].f[k] = 0;
			for (int k = 0; k < nF; k++) st.X[i].f[k] = 1e-3 * std::sin(1.0 + i + 3.0 * k);
		}
		st.init = true;
	}
	for (int i = 0; i < m; i++) {
		project(st.X[i]);
		normalize(st.X[i]);
		applyHat(st.X[i], AX[i]);
	}
	std::vector<BVec>& X = st.X;
	Eigen::VectorXd lam(m);
	{
		Eigen::MatrixXd GA(m, m), GB(m, m), C;
		for (int i = 0; i < m; i++)
			for (int j = i; j < m; j++) {
				GA(i, j) = GA(j, i) = sp.dot(X[i], AX[j]);
				GB(i, j) = GB(j, i) = sp.dot(X[i], X[j]);
			}
		if (rayleigh_ritz(GA, GB, m, lam, C)) {
			for (int j = 0; j < m; j++) {
				std::vector<const BVec*> xs, axs;
				std::vector<double> c;
				for (int i = 0; i < m; i++) {
					xs.push_back(&X[i]);
					axs.push_back(&AX[i]);
					c.push_back(C(i, j));
				}
				lincomb(sp, T1[j], xs, c);
				lincomb(sp, T2[j], axs, c);
			}
			for (int j = 0; j < m; j++) {
				swap(X[j], T1[j]);
				swap(AX[j], T2[j]);
			}
		} else {
			for (int i = 0; i < m; i++) lam(i) = sp.dot(X[i], AX[i]);
		}
	}

	bool haveP = false, converged = false;
	int it = 0;
	for (; it < maxit; it++) {
		std::vector<double> rn(m);
		for (int i = 0; i < m; i++) {
			sp.copy(AX[i], W[i]);
			sp.axpy(-lam(i), X[i], W[i]);
			rn[i] = sp.norm(W[i]);
		}
		if (rn[0] <= tol * std::max(std::fabs(lam(0)), 1e-3)) {
			converged = true;
			break;
		}
		for (int i = 0; i < m; i++) {
			if (usePrec && precHat(W[i], T1[i]) && sp.dot(W[i], T1[i]) > 0) swap(W[i], T1[i]);
			project(W[i]);
			normalize(W[i]);
			applyHat(W[i], AW[i]);
		}
		int k = haveP ? 3 * m : 2 * m;
		std::vector<BVec*> S, AS;
		for (int i = 0; i < m; i++) {
			S.push_back(&X[i]);
			AS.push_back(&AX[i]);
		}
		for (int i = 0; i < m; i++) {
			S.push_back(&W[i]);
			AS.push_back(&AW[i]);
		}
		if (haveP)
			for (int i = 0; i < m; i++) {
				S.push_back(&Pv[i]);
				AS.push_back(&AP[i]);
			}
		Eigen::MatrixXd GA(k, k), GB(k, k), C;
		for (int i = 0; i < k; i++)
			for (int j = i; j < k; j++) {
				GA(i, j) = GA(j, i) = sp.dot(*S[i], *AS[j]);
				GB(i, j) = GB(j, i) = sp.dot(*S[i], *S[j]);
			}
		Eigen::VectorXd lnew;
		if (!rayleigh_ritz(GA, GB, m, lnew, C)) break;
		// P_new = W C_W + P C_P,  X_new = X C_X + P_new
		for (int j = 0; j < m; j++) {
			std::vector<const BVec*> ps, aps;
			std::vector<double> c;
			for (int i = m; i < k; i++) {
				ps.push_back(S[i]);
				aps.push_back(AS[i]);
				c.push_back(C(i, j));
			}
			lincomb(sp, T1[j], ps, c);
			lincomb(sp, T2[j], aps, c);
		}
		for (int j = 0; j < m; j++) {
			// W is free now: W_j = X C_X + P_new_j
			std::vector<const BVec*> xs, axs;
			std::vector<double> c;
			for (int i = 0; i < m; i++) {
				xs.push_back(&X[i]);
				axs.push_back(&AX[i]);
				c.push_back(C(i, j));
			}
			xs.push_back(&T1[j]);
			axs.push_back(&T2[j]);
			c.push_back(1.0);
			lincomb(sp, W[j], xs, c);
			lincomb(sp, AW[j], axs, c);
		}
		for (int j = 0; j < m; j++) {
			swap(X[j], W[j]);
			swap(AX[j], AW[j]);
			swap(Pv[j], T1[j]);
			swap(AP[j], T2[j]);
		}
		lam = lnew;
		haveP = true;
	}
	for (int i = 0; i < m; i++) {
		sp.release(AX[i]); sp.release(W[i]); sp.release(AW[i]);
		sp.release(Pv[i]); sp.release(AP[i]); sp.release(T1[i]); sp.release(T2[i]);
	}
	sp.release(sc);
	if (iters) *iters = it;
	if (conv) *conv = converged;
	return lam(0);
}

bool Solver::effectiveTangent(double Abar[81], double tol, int* iters) {
	KOp op(*this);
	if (chi.size() < 15) {
		while (chi.size() < 15) chi.push_back(dalloc(3 * cell.ne));
	}
	BVec b = spK.alloc(), x;
	int total = 0;
	bool ok = true;
	for (int kl = 0; kl < 9; kl++) {
		double dF[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, t[9];
		dF[kl] = 1;
		cell_product(cell, nullptr, dF, b.w, t);  // B column and D9 column
		x.w = chi[6 + kl];
		KResult kr = pcg(spK, op, b, x, tol, opt.krylovMax);
		total += kr.iters;
		if (kr.status != KStatus::Converged && kr.status != KStatus::MaxIter) ok = false;
		double bx[9];
		cell_product(cell, x.w, nullptr, tmp1.w, bx);  // (B^T x)_ij = <s A[Grad x]>_ij
		for (int ij = 0; ij < 9; ij++) Abar[ij * 9 + kl] = t[ij] - bx[ij];
	}
	spK.release(b);
	for (int i = 0; i < 9; i++)
		for (int j = 0; j < i; j++) Abar[i * 9 + j] = Abar[j * 9 + i] = 0.5 * (Abar[i * 9 + j] + Abar[j * 9 + i]);
	if (iters) *iters = total;
	return ok;
}

double Stability::supercell(double tol, int maxit, Prec p) {
	int n = S.n, n2 = 2 * n;
	if (!sc) {
		std::vector<float> rho2(size_t(n2) * n2 * n2);
		for (int k = 0; k < n2; k++)
			for (int j = 0; j < n2; j++)
				for (int i = 0; i < n2; i++)
					rho2[i + size_t(n2) * (j + size_t(n2) * k)] = S.rhoHost[(i % n) + size_t(n) * ((j % n) + size_t(n) * (k % n))];
		SolverOptions o = S.opt;
		o.verbose = 0;
		sc.reset(new Solver(n2, rho2, o));
	}
	if (sc->freeComps != S.freeComps || !sc->scalingReady) {
		sc->setFree(S.freeComps);
		sc->resetState();
		sc->initScaling();
	}
	tile_vector(n, S.u.w, sc->u.w);
	for (int k = 0; k < 9; k++) sc->u.f[k] = S.u.f[k];
	sc->setPrescribed(S.Fbar);
	if (!sc->evaluateCurrent()) return std::numeric_limits<double>::quiet_NaN();
	bool havePrec = sc->buildPreconditioner(p, 0.f) || sc->buildPreconditioner(Prec::FP32, 0.f);
	return sc->lambdaMin(sc->eig, tol, maxit, havePrec);
}

double Stability::rankOne(double* AbarOut, double tol) {
	double Abar[81];
	if (!S.effectiveTangent(Abar, tol)) return std::numeric_limits<double>::quiet_NaN();
	if (AbarOut)
		for (int c = 0; c < 81; c++) AbarOut[c] = Abar[c];
	return ellipticity_min(Abar);
}

}  // namespace fs
