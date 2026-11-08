// Host-callable wrappers of the finite-strain device kernels.
//
// Grid conventions: the cell [0,1]^3 has n^3 voxels and n^3 nodes (periodic). Node and voxel
// (i, j, k) have the lexicographic index i + n (j + n k). Voxel (i, j, k) has the nodes
// (i + a, j + b, k + c), a, b, c in {0, 1}, with local index l = a + 2 b + 4 c. Nodal vectors
// store three components per node, index 3 node + component. Tensors are row-major, comp(i, j) = 3 i + j.
#pragma once

#include "fs_common.h"
#include <cuda_fp16.h>

namespace fs {

// ---------------------------------------------------------------------------------------------
// One cell problem on the device
// ---------------------------------------------------------------------------------------------
struct Cell {
	int n = 0;
	size_t ne = 0;            // n^3 voxels (= nodes)
	Material mat{};
	float* rho = nullptr;     // physical density rho in [0, 1]
	float* s = nullptr;       // stiffness scale rhomin + rho^p
	float* ds = nullptr;      // d s / d rho
	float* gam = nullptr;     // gamma(rho^p)
	float* dgam = nullptr;    // d gamma / d rho
	unsigned char* isvoid = nullptr;  // rho^p < rho0
	float* Hc = nullptr;      // deformation-gradient cache, F - I at the 8 Gauss points, [72][ne]
	long long nvoid = 0;      // number of void voxels
};

void cell_alloc(Cell& c, int n, const Material& m);
void cell_free(Cell& c);
// Sets the density (host array of n^3 values) and the derived per-voxel fields.
void cell_set_density(Cell& c, const float* rho_host);

// Fills the cache with F - I = (Fbar - I) + Grad w at all Gauss points; w == nullptr means w = 0.
void cell_cache_update(const Cell& c, const float* w, const double FbarMinusI[9]);

struct EnergyOut {
	double energy = 0;          // Pi_h
	double absEnergy = 0;       // sum of omega s |psi|, the scale of the rounding level of Pi_h
	double fnorm = 0;           // root of the summed squared element forces, the scale of the rounding level of r
	long long badJ = 0;         // Gauss points with J_gamma <= 0
	long long voidDistorted = 0;  // void voxels with J < 0.1 at a Gauss point
};
void cell_energy(const Cell& c, EnergyOut& out);

// r = dPi_h/dw (3 n^3), Pbar = <s P>, fnorm = the root of the summed squared element forces (the scale
// of the FP32 rounding of r, which is assembled from them).
void cell_residual(const Cell& c, float* r, double Pbar[9], double* fnorm = nullptr);

// y = K_T x + B dF and t = <s A[Grad x + dF]> (the second gives B^T x + D dF on the free components).
// x == nullptr means x = 0; t == nullptr skips the reduction.
void cell_product(const Cell& c, const float* x, const double dF[9], float* y, double t[9]);

// Inverses of the nodal 3x3 diagonal blocks of K_T + sigma diag(Zs) (Dinv, 9 per node), optionally
// the diagonal of K_T (diag, 3 per node). Returns the number of blocks that are not positive definite.
int cell_diag_blocks(const Cell& c, const float* Zs, float sigma, float* Dinv, float* diag);

// One colour of the matrix-free Gauss-Seidel sweep on the fine level for K_T + sigma diag(Zs).
void cell_gs(const Cell& c, int color, const float* f, float* x, const float* Dinv, const float* Zs, float sigma);

// Galerkin restriction of the fine tangent to the first coarse level, evaluated element by element.
// S1 (FP32, 243 entries per coarse node, layout [243][nc^3]) must be zero on entry.
void cell_restrict_tangent(const Cell& c, float* S1);
// Adds P^T sigma diag(Zs) P to S1 (the shift of the MINRES preconditioner).
void restrict_shift(int nf, const float* Zs, float sigma, float* S1);

// Adjoint sensitivity: dJ[e] += sum_g omega d(sP)/drho (F_g; rho_e) : (a - Lambda_g),
// Lambda_g = Grad lamw (g) + LamBar.
void cell_sensitivity(const Cell& c, const float* lamw, const double LamBar[9], const double a[9], double* dJ);

// Small-strain moduli and third-order constants from the six correctors chi[I] at F = I
// (Voigt order 11, 22, 33, 23, 13, 12 with engineering shears):
//   C[I][J] = <s h_I : A_L : h_J>, h_I = e_I + Grad chi_I
//   G56[IJK] = <s gamma^3 G_NH[h_I, h_J, h_K]> for I <= J <= K (if toec)
void cell_linear_moduli(const Cell& c, float* const chi[6], double C[36], double G56[56], bool toec);
// dJ[e] += sum_g omega ds/drho sum_IJ W[I][J] h_I : A_L : h_J
void cell_moduli_sensitivity(const Cell& c, float* const chi[6], const double W[36], double* dJ);

// Voigt basis tensor e_I (engineering shears: e_23 = (E23 + E32) / 2).
void voigt_basis(int I, double E[9]);

// Periodic tiling of a nodal 3-vector field from n^3 to (2n)^3.
void tile_vector(int n, const float* w, float* w2);

// ---------------------------------------------------------------------------------------------
// Vector operations (FP32 storage, FP64 reductions)
// ---------------------------------------------------------------------------------------------
float* dalloc(size_t m);
double* dalloc_d(size_t m);
void dfree(void* p);
double vdot(const float* a, const float* b, size_t m);
void vaxpy(double a, const float* x, float* y, size_t m);                 // y += a x
void vaxpby(double a, const float* x, double b, float* y, size_t m);      // y = a x + b y
void vcopy(const float* x, float* y, size_t m);
void vzero(float* y, size_t m);
void vscale(double a, float* y, size_t m);
void vmul(const float* a, const float* x, float* y, size_t m);            // y = a .* x
void vaxmy(double a, const float* d, const float* x, float* y, size_t m); // y += a d .* x
void vsqrt(const float* a, float* y, size_t m);
void vrsqrt(const float* a, float* y, size_t m);
void vrandom(float* v, size_t m, unsigned seed);
// Removes the mean of each of the three components (projection onto the complement of the translations).
void vproject_mean(float* v, size_t nodes);
// Removes the components of v along q_c = sq .* e_c, c = 1..3 (the scaled translations).
void vproject_weighted(float* v, const float* sq, size_t nodes);
void dzero(double* y, size_t m);
void dscale(double a, double* y, size_t m);
void dfill(double v, double* y, size_t m);
void dmul(const double* a, const double* x, double* y, size_t m);  // y = a .* x
void host2d(const double* h, double* d, size_t m);
void d2host(const double* d, double* h, size_t m);
void f2host(const float* d, float* h, size_t m);
void host2f(const float* h, float* d, size_t m);

// ---------------------------------------------------------------------------------------------
// Density filter and projection of the design layer
// ---------------------------------------------------------------------------------------------
// Periodic hat filter of radius r (in voxels) of an n^3 field; symmetric, so it is its own adjoint.
void filter_periodic(int n, float r, const double* in, double* out);
// rho = H_beta(rt) with threshold eta, drho = dH/drt.
void project_heaviside(size_t m, float beta, float eta, const double* rt, double* rho, double* drho);

}  // namespace fs
