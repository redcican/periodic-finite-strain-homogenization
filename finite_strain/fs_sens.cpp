// Adjoint sensitivities, small-strain moduli and third-order elastic constants.
#include "fs_solver.h"

#include <cmath>
#include <cstring>

namespace fs {

namespace {

// Voigt vector (engineering shears) of the symmetric part of X.
void voigt_of(const double X[9], double v[6]) {
	v[0] = X[0];
	v[1] = X[4];
	v[2] = X[8];
	v[3] = X[5] + X[7];
	v[4] = X[2] + X[6];
	v[5] = X[1] + X[3];
}

void matmul3(const double* A, const double* B, double* C) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			double v = 0;
			for (int k = 0; k < 3; k++) v += A[i * 3 + k] * B[k * 3 + j];
			C[i * 3 + j] = v;
		}
}

// C^H[e_a, sym(e_b e_c)]
double c_term(const double C[36], int a, int b, int c) {
	double Eb[9], Ec[9], X[9], v[6];
	voigt_basis(b, Eb);
	voigt_basis(c, Ec);
	matmul3(Eb, Ec, X);
	voigt_of(X, v);
	double s = 0;
	for (int L = 0; L < 6; L++) s += C[a * 6 + L] * v[L];
	return s;
}

}  // namespace

bool Solver::linearModuli(double C[36], double T56[56], bool toec, double tol, Prec p) {
	resetState();
	if (!evaluateCurrent()) return false;
	if (!buildPreconditioner(p, 0.f) && !buildPreconditioner(Prec::FP32, 0.f)) return false;
	while (chi.size() < 6) chi.push_back(dalloc(3 * cell.ne));
	KOp op(*this);
	BVec b = spK.alloc(), x;
	bool ok = true;
	for (int I = 0; I < 6; I++) {
		double E[9];
		voigt_basis(I, E);
		cell_product(cell, nullptr, E, b.w, nullptr);  // B e_I
		vscale(-1.0, b.w, spK.m);                      // K chi_I = -B e_I
		x.w = chi[I];
		KResult kr = pcg(spK, op, b, x, tol, opt.krylovMax);
		if (kr.status != KStatus::Converged) ok = false;
		if (opt.verbose >= 2) std::printf("    corrector %d: %d iterations, relres %.2e\n", I, kr.iters, kr.relres);
	}
	spK.release(b);
	double G[56];
	cell_linear_moduli(cell, chi.data(), C, G, toec);
	if (toec && T56) {
		// T_IJK = <s gamma^3 G_NH[h_I, h_J, h_K]> - 3 Sym_IJK C^H[e_I, e_J e_K]
		int t = 0;
		for (int I = 0; I < 6; I++)
			for (int J = I; J < 6; J++)
				for (int K = J; K < 6; K++) {
					double corr = (c_term(C, I, J, K) + c_term(C, I, K, J) + c_term(C, J, I, K) + c_term(C, J, K, I) +
									  c_term(C, K, I, J) + c_term(C, K, J, I)) / 6.0;
					T56[t] = G[t] - 3.0 * corr;
					t++;
				}
	}
	return ok;
}

void Solver::moduliSensitivity(const double W[36], double* dJ) { cell_moduli_sensitivity(cell, chi.data(), W, dJ); }

bool Solver::adjoint(const double a[9], const double* b, double* dJ, double tol, Prec p, int* iters) {
	if (!preconditionerReady()) {
		if (!buildPreconditioner(p, 0.f) && !buildPreconditioner(Prec::FP32, 0.f)) return false;
	}
	double t[9];
	cell_product(cell, nullptr, a, rhs.w, t);  // dJ/dw = sum_ij a_ij B_ij
	for (int k = 0; k < sp.nF; k++) rhs.f[k] = t[freeComps[k]] + (b ? b[k] : 0.0);
	sp.zero(du);
	KResult kr = pcg(sp, *this, rhs, du, tol, opt.krylovMax);
	int it = kr.iters;
	if (kr.status == KStatus::Indefinite) {
		kr = minres(sp, *this, rhs, du, tol, opt.krylovMax);
		it += kr.iters;
	}
	if (iters) *iters = it;
	if (kr.status == KStatus::PrecFail) return false;
	double LamBar[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	for (int k = 0; k < sp.nF; k++) LamBar[freeComps[k]] = du.f[k];
	cell_sensitivity(cell, du.w, LamBar, a, dJ);
	return kr.status == KStatus::Converged;
}

}  // namespace fs
