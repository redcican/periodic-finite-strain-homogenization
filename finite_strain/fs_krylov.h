// Krylov solvers on bordered vectors u = (w, Fbar_F): fluctuation block on the device (FP32) and the
// free macroscopic components on the host (FP64). All inner products are accumulated in FP64.
#pragma once

#include "fs_device.h"

namespace fs {

struct BVec {
	float* w = nullptr;
	double f[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

void swap(BVec& a, BVec& b);

class Space {
public:
	Space() = default;
	Space(size_t m, int nF) : m(m), nF(nF) {}
	size_t m = 0;  // length of the fluctuation block (3 n^3)
	int nF = 0;    // number of free macroscopic components
	BVec alloc() const;
	void release(BVec& v) const;
	double dot(const BVec& a, const BVec& b) const;
	double norm(const BVec& a) const;
	void axpy(double a, const BVec& x, BVec& y) const;              // y += a x
	void axpby(double a, const BVec& x, double b, BVec& y) const;   // y = a x + b y
	void copy(const BVec& x, BVec& y) const;
	void zero(BVec& y) const;
	void scale(double a, BVec& y) const;
};

// Linear operator with preconditioner; prec returns false if the preconditioner is not available.
class LinOp {
public:
	virtual ~LinOp() = default;
	virtual void apply(const BVec& x, BVec& y) = 0;
	virtual bool prec(const BVec& r, BVec& z) = 0;
};

enum class KStatus { Converged, MaxIter, Indefinite, PrecFail };

struct KResult {
	KStatus status = KStatus::MaxIter;
	int iters = 0;
	double relres = 1;
};

// Preconditioned conjugate gradients for A x = b from the initial x, to ||b - A x|| <= tol ||b||.
// Returns Indefinite at a direction with p^T A p <= 0 (x is then the current iterate) and PrecFail
// at r^T M^{-1} r <= 0.
KResult pcg(const Space& sp, LinOp& A, const BVec& b, BVec& x, double tol, int maxit);

// Preconditioned MINRES (Paige and Saunders) for symmetric A and symmetric positive definite M,
// from the initial x, to ||b - A x|| <= tol ||b||. Returns PrecFail at r^T M^{-1} r < 0.
KResult minres(const Space& sp, LinOp& A, const BVec& b, BVec& x, double tol, int maxit);

}  // namespace fs
