// Geometric multigrid preconditioner for the tangent of the finite-strain cell problem.
//
// Level 0 is the fine grid, on which the tangent is applied matrix-free from the deformation-gradient
// cache. Level 1 is the Galerkin restriction of the fine tangent, evaluated element by element; every
// coarser level is the Galerkin product of the stored stencils of the level above. Levels 1 .. L-1
// store 27-point stencils with 3x3 blocks in the symmetric scaling Z^{-1/2} K Z^{-1/2} (Z = diag K,
// kept in FP32) in FP16 or FP32. The coarsest level L has 8^3 nodes and is solved by a dense FP64
// Cholesky factorization with the rigid translations removed.
#pragma once

#include "fs_device.h"

#include <vector>

namespace fs {

class Multigrid {
public:
	Multigrid() = default;
	~Multigrid();
	Multigrid(const Multigrid&) = delete;
	Multigrid& operator=(const Multigrid&) = delete;

	// Allocates the hierarchy for a fine grid with n = 8 * 2^L nodes per direction, L >= 1.
	void init(int n);
	void release();

	// Restricts the tangent of `cell` (plus sigma diag(Zs) if sigma > 0) to all coarse levels and
	// factorizes the coarsest operator. Returns false if an operator is not positive definite.
	bool update(const Cell& cell, Prec prec, float sigma, const float* Zs);

	// z = V r, one symmetric V-cycle on the fine level; r and z are fine nodal vectors (3 n^3).
	void vcycle(const float* r, float* z);

	int levels() const { return L + 1; }
	Prec precision() const { return prec_; }
	// Device memory of the coarse operators in bytes (stored stencils, factor, workspace).
	size_t coarse_bytes() const;

private:
	struct Level {
		int n = 0;
		size_t N = 0;
		float* sqZ = nullptr;   // sqrt of the diagonal, 3 N
		__half* Sh = nullptr;   // FP16 scaled stencils, 243 N
		float* Sf = nullptr;    // FP32 scaled stencils, 243 N (allocated on first FP32 use)
		float* Dinv = nullptr;  // inverses of the nodal diagonal blocks, 9 N
		float* x = nullptr;
		float* f = nullptr;
		float* r = nullptr;
	};

	bool store_level(int l, const float* S32);
	bool factor_dense(const float* S32);
	void dense_solve(const float* f, float* x);
	void smooth(int l, const float* f, float* x, bool forward);
	void residual(int l, const float* f, const float* x, float* r);
	void cycle(int l, const float* f, float* x);

	int nf = 0, L = 0;
	const Cell* cell_ = nullptr;
	float sigma_ = 0.f;
	const float* Zs_ = nullptr;
	Prec prec_ = Prec::FP16;
	float* Dinv0 = nullptr;  // fine diagonal block inverses
	float* f0 = nullptr;     // fine right-hand side (projected copy of r)
	float* r0 = nullptr;     // fine residual
	float* ws = nullptr;     // FP32 accumulation workspace, 243 N_1
	std::vector<Level> lv;
	// coarsest level
	int Md = 0;
	double* Kd = nullptr;
	double* dwork = nullptr;
	double* drhs = nullptr;
	int* dinfo = nullptr;
	int lwork = 0;
	void* handle = nullptr;
	bool built = false;
};

}  // namespace fs
