// Element kernels of the finite-strain cell problem: deformation-gradient cache, energy, residual,
// tangent product, matrix-free Gauss-Seidel, restriction to the first coarse level, sensitivities
// and small-strain moduli.
#include "fs_device.h"
#include "fs_material.cuh"
#include "fs_reduce.cuh"

#include <vector>

namespace fs {

namespace {

__constant__ float c_dN[8][8][3];  // reference shape gradients [Gauss point][local node][direction]
__constant__ float c_q[2];         // Gauss coordinates on [0, 1]

struct F9 {
	float v[9];
};
struct F36 {
	float v[36];
};
struct Ptr6 {
	const float* p[6];
};

F9 make_f9(const double* a) {
	F9 f;
	for (int c = 0; c < 9; c++) f.v[c] = a ? float(a[c]) : 0.f;
	return f;
}

void init_constants() {
	static bool done = false;
	if (done) return;
	done = true;
	double q[2] = {0.5 * (1 - 1 / std::sqrt(3.0)), 0.5 * (1 + 1 / std::sqrt(3.0))};
	float dN[8][8][3];
	for (int g = 0; g < 8; g++) {
		double xi[3] = {q[g & 1], q[(g >> 1) & 1], q[g >> 2]};
		for (int l = 0; l < 8; l++) {
			int a[3] = {l & 1, (l >> 1) & 1, l >> 2};
			double N[3], dn[3];
			for (int d = 0; d < 3; d++) {
				N[d] = a[d] ? xi[d] : 1 - xi[d];
				dn[d] = a[d] ? 1 : -1;
			}
			dN[g][l][0] = float(dn[0] * N[1] * N[2]);
			dN[g][l][1] = float(N[0] * dn[1] * N[2]);
			dN[g][l][2] = float(N[0] * N[1] * dn[2]);
		}
	}
	float qf[2] = {float(q[0]), float(q[1])};
	FS_CUDA(cudaMemcpyToSymbol(c_dN, dN, sizeof(dN)));
	FS_CUDA(cudaMemcpyToSymbol(c_q, qf, sizeof(qf)));
}

double* scratch() {
	static double* p = nullptr;
	if (!p) FS_CUDA(cudaMalloc(&p, 128 * sizeof(double)));
	return p;
}

__device__ inline void elem_nodes(int n, int i, int j, int k, size_t* nd) {
#pragma unroll
	for (int l = 0; l < 8; l++) nd[l] = nid(wrapi(i + (l & 1), n), wrapi(j + ((l >> 1) & 1), n), wrapi(k + (l >> 2), n), n);
}

__device__ inline void load_D(const float* Hc, size_t ne, size_t e, int g, float* D) {
#pragma unroll
	for (int c = 0; c < 9; c++) D[c] = Hc[(size_t(g) * 9 + c) * ne + e];
}

// Gradient at Gauss point g of the element field xe (8 nodes x 3), scaled by n.
__device__ inline void grad_at(const float* xe, int g, float n, float* G) {
#pragma unroll
	for (int a = 0; a < 3; a++)
#pragma unroll
		for (int b = 0; b < 3; b++) {
			float v = 0.f;
#pragma unroll
			for (int l = 0; l < 8; l++) v += xe[l * 3 + a] * c_dN[g][l][b];
			G[a * 3 + b] = n * v;
		}
}

// Voxel of colour `color` (parity of i, j, k) handled by thread t; false if t is out of range.
__device__ inline bool colour_voxel(int n, int color, size_t t, int& i, int& j, int& k) {
	int h = n / 2;
	if (t >= size_t(h) * h * h) return false;
	int i2 = int(t % h), j2 = int((t / h) % h), k2 = int(t / (size_t(h) * h));
	i = 2 * i2 + (color & 1);
	j = 2 * j2 + ((color >> 1) & 1);
	k = 2 * k2 + (color >> 2);
	return true;
}

__device__ inline void voigt_E(int I, float* E) {
#pragma unroll
	for (int c = 0; c < 9; c++) E[c] = 0.f;
	switch (I) {
	case 0: E[0] = 1.f; break;
	case 1: E[4] = 1.f; break;
	case 2: E[8] = 1.f; break;
	case 3: E[5] = E[7] = 0.5f; break;
	case 4: E[2] = E[6] = 0.5f; break;
	default: E[1] = E[3] = 0.5f; break;
	}
}

// ---------------------------------------------------------------------------------------------

__global__ void k_set_density(size_t ne, Material m, const float* rho, float* s, float* ds, float* gam, float* dgam,
	unsigned char* isvoid) {
	size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (e >= ne) return;
	float r = fminf(fmaxf(rho[e], 0.f), 1.f);
	float rp = powf(r, m.p);
	float drp = r > 0.f ? m.p * powf(r, m.p - 1.f) : 0.f;
	s[e] = m.rhomin + rp;
	ds[e] = drp;
	gam[e] = fs_gamma(m, rp);
	dgam[e] = drp * fs_dgamma(m, rp);
	isvoid[e] = rp < m.rho0 ? 1 : 0;
}

__global__ void k_cache(int n, size_t ne, const float* __restrict__ w, F9 Hb, float* __restrict__ Hc) {
	size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (e >= ne) return;
	int i = int(e % n), j = int((e / n) % n), k = int(e / (size_t(n) * n));
	size_t nd[8];
	elem_nodes(n, i, j, k, nd);
	float xe[24];
#pragma unroll
	for (int l = 0; l < 8; l++)
#pragma unroll
		for (int a = 0; a < 3; a++) xe[l * 3 + a] = w ? w[nd[l] * 3 + a] : 0.f;
	for (int g = 0; g < 8; g++) {
		float G[9];
		grad_at(xe, g, float(n), G);
#pragma unroll
		for (int c = 0; c < 9; c++) Hc[(size_t(g) * 9 + c) * ne + e] = Hb.v[c] + G[c];
	}
}

__global__ void k_energy(int n, size_t ne, Material m, const float* s, const float* gam, const unsigned char* isvoid,
	const float* Hc, float om, double* out) {
	size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	double v[4] = {0, 0, 0, 0};
	if (e < ne) {
		float acc = 0.f, accabs = 0.f;
		int bad = 0;
		bool dist = false;
		float ge = gam[e];
		for (int g = 0; g < 8; g++) {
			float D[9];
			load_D(Hc, ne, e, g, D);
			GPState st;
			if (!gp_prepare(D, ge, st)) {
				bad++;
				continue;
			}
			float psi = energy(m, st);
			acc += psi;
			accabs += fabsf(psi);
			if (isvoid[e]) {
				float Fi[9], j1, i23;
				detinv(D, Fi, j1, i23);
				if (1.f + j1 < 0.1f) dist = true;
			}
		}
		double w = double(om) * double(s[e]);
		v[0] = w * acc;
		v[1] = w * accabs;
		v[2] = bad;
		v[3] = dist ? 1.0 : 0.0;
	}
	block_sum_atomic<4>(v, out);
}

__global__ void k_residual(int n, int color, size_t ne, Material m, const float* s, const float* gam, const float* Hc,
	float om, float* r, double* Pbar) {
	size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	double acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	int i, j, k;
	if (colour_voxel(n, color, t, i, j, k)) {
		size_t e = nid(i, j, k, n);
		size_t nd[8];
		elem_nodes(n, i, j, k, nd);
		float fe[24];
#pragma unroll
		for (int c = 0; c < 24; c++) fe[c] = 0.f;
		float w = om * s[e], wn = w * float(n), ge = gam[e];
		for (int g = 0; g < 8; g++) {
			float D[9], P[9];
			load_D(Hc, ne, e, g, D);
			GPState st;
			if (!gp_prepare(D, ge, st)) continue;
			stress(m, st, P);
#pragma unroll
			for (int c = 0; c < 9; c++) acc[c] += double(w * P[c]);
#pragma unroll
			for (int l = 0; l < 8; l++)
#pragma unroll
				for (int a = 0; a < 3; a++)
					fe[l * 3 + a] += wn * (P[a * 3 + 0] * c_dN[g][l][0] + P[a * 3 + 1] * c_dN[g][l][1] + P[a * 3 + 2] * c_dN[g][l][2]);
		}
#pragma unroll
		for (int l = 0; l < 8; l++)
#pragma unroll
			for (int a = 0; a < 3; a++) r[nd[l] * 3 + a] += fe[l * 3 + a];
	}
	block_sum_atomic<9>(acc, Pbar);
}

__global__ void k_product(int n, int color, size_t ne, Material m, const float* s, const float* gam, const float* Hc,
	float om, const float* x, F9 dF, float* y, double* tout) {
	size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	double acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	int i, j, k;
	if (colour_voxel(n, color, t, i, j, k)) {
		size_t e = nid(i, j, k, n);
		size_t nd[8];
		elem_nodes(n, i, j, k, nd);
		float xe[24], fe[24];
#pragma unroll
		for (int l = 0; l < 8; l++)
#pragma unroll
			for (int a = 0; a < 3; a++) {
				xe[l * 3 + a] = x ? x[nd[l] * 3 + a] : 0.f;
				fe[l * 3 + a] = 0.f;
			}
		float w = om * s[e], wn = w * float(n), ge = gam[e];
		for (int g = 0; g < 8; g++) {
			float D[9], H[9], Y[9];
			load_D(Hc, ne, e, g, D);
			GPState st;
			if (!gp_prepare(D, ge, st)) continue;
			grad_at(xe, g, float(n), H);
#pragma unroll
			for (int c = 0; c < 9; c++) H[c] += dF.v[c];
			tangent(m, st, H, Y);
#pragma unroll
			for (int c = 0; c < 9; c++) acc[c] += double(w * Y[c]);
#pragma unroll
			for (int l = 0; l < 8; l++)
#pragma unroll
				for (int a = 0; a < 3; a++)
					fe[l * 3 + a] += wn * (Y[a * 3 + 0] * c_dN[g][l][0] + Y[a * 3 + 1] * c_dN[g][l][1] + Y[a * 3 + 2] * c_dN[g][l][2]);
		}
#pragma unroll
		for (int l = 0; l < 8; l++)
#pragma unroll
			for (int a = 0; a < 3; a++) y[nd[l] * 3 + a] += fe[l * 3 + a];
	}
	block_sum_atomic<9>(acc, tout);
}

__device__ inline bool inv3_spd(const float* K, float* Ki) {
	float d0 = K[0];
	float d1 = K[0] * K[4] - K[1] * K[3];
	float c00 = K[4] * K[8] - K[5] * K[7];
	float c01 = K[2] * K[7] - K[1] * K[8];
	float c02 = K[1] * K[5] - K[2] * K[4];
	float c10 = K[5] * K[6] - K[3] * K[8];
	float c11 = K[0] * K[8] - K[2] * K[6];
	float c12 = K[2] * K[3] - K[0] * K[5];
	float c20 = K[3] * K[7] - K[4] * K[6];
	float c21 = K[1] * K[6] - K[0] * K[7];
	float c22 = K[0] * K[4] - K[1] * K[3];
	float det = K[0] * c00 + K[1] * c10 + K[2] * c20;
	bool pd = d0 > 0.f && d1 > 0.f && det > 0.f;
	float id = det != 0.f ? 1.f / det : 0.f;
	Ki[0] = c00 * id; Ki[1] = c01 * id; Ki[2] = c02 * id;
	Ki[3] = c10 * id; Ki[4] = c11 * id; Ki[5] = c12 * id;
	Ki[6] = c20 * id; Ki[7] = c21 * id; Ki[8] = c22 * id;
	return pd;
}

__global__ void k_diag(int n, size_t ne, Material m, const float* s, const float* gam, const float* Hc, float om,
	const float* Zs, float sigma, float* Dinv, float* diag, int* notpd) {
	size_t a = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (a >= ne) return;
	int i = int(a % n), j = int((a / n) % n), k = int(a / (size_t(n) * n));
	float K[9];
#pragma unroll
	for (int c = 0; c < 9; c++) K[c] = 0.f;
	for (int le = 0; le < 8; le++) {
		int la = le & 1, lb = (le >> 1) & 1, lc = le >> 2;
		size_t e = nid(wrapi(i - la, n), wrapi(j - lb, n), wrapi(k - lc, n), n);
		float w = om * s[e], ge = gam[e];
		for (int g = 0; g < 8; g++) {
			float D[9];
			load_D(Hc, ne, e, g, D);
			GPState st;
			if (!gp_prepare(D, ge, st)) continue;
			float dn[3] = {float(n) * c_dN[g][le][0], float(n) * c_dN[g][le][1], float(n) * c_dN[g][le][2]};
			for (int kk = 0; kk < 3; kk++) {
				float H[9], Y[9];
#pragma unroll
				for (int c = 0; c < 9; c++) H[c] = 0.f;
				H[kk * 3 + 0] = dn[0];
				H[kk * 3 + 1] = dn[1];
				H[kk * 3 + 2] = dn[2];
				tangent(m, st, H, Y);
#pragma unroll
				for (int ia = 0; ia < 3; ia++) K[ia * 3 + kk] += w * (Y[ia * 3 + 0] * dn[0] + Y[ia * 3 + 1] * dn[1] + Y[ia * 3 + 2] * dn[2]);
			}
		}
	}
	if (diag) {
		diag[a * 3 + 0] = K[0];
		diag[a * 3 + 1] = K[4];
		diag[a * 3 + 2] = K[8];
	}
	if (Zs && sigma != 0.f) {
		K[0] += sigma * Zs[a * 3 + 0];
		K[4] += sigma * Zs[a * 3 + 1];
		K[8] += sigma * Zs[a * 3 + 2];
	}
	float Ki[9];
	if (!inv3_spd(K, Ki)) atomicAdd(notpd, 1);
#pragma unroll
	for (int c = 0; c < 9; c++) Dinv[a * 9 + c] = Ki[c];
}

__global__ void k_gs_fine(int n, int color, size_t ne, Material m, const float* s, const float* gam, const float* Hc,
	float om, const float* f, float* x, const float* Dinv, const float* Zs, float sigma) {
	size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	int i, j, k;
	if (!colour_voxel(n, color, t, i, j, k)) return;
	size_t a = nid(i, j, k, n);
	float acc[3] = {0.f, 0.f, 0.f};
	for (int le = 0; le < 8; le++) {
		int la = le & 1, lb = (le >> 1) & 1, lc = le >> 2;
		int ei = wrapi(i - la, n), ej = wrapi(j - lb, n), ek = wrapi(k - lc, n);
		size_t e = nid(ei, ej, ek, n);
		size_t nd[8];
		elem_nodes(n, ei, ej, ek, nd);
		float xe[24];
#pragma unroll
		for (int l = 0; l < 8; l++)
#pragma unroll
			for (int c = 0; c < 3; c++) xe[l * 3 + c] = x[nd[l] * 3 + c];
		float w = om * s[e], ge = gam[e];
		for (int g = 0; g < 8; g++) {
			float D[9], H[9], Y[9];
			load_D(Hc, ne, e, g, D);
			GPState st;
			if (!gp_prepare(D, ge, st)) continue;
			grad_at(xe, g, float(n), H);
			tangent(m, st, H, Y);
			float d0 = float(n) * c_dN[g][le][0], d1 = float(n) * c_dN[g][le][1], d2 = float(n) * c_dN[g][le][2];
#pragma unroll
			for (int c = 0; c < 3; c++) acc[c] += w * (Y[c * 3 + 0] * d0 + Y[c * 3 + 1] * d1 + Y[c * 3 + 2] * d2);
		}
	}
	float r[3];
#pragma unroll
	for (int c = 0; c < 3; c++) {
		r[c] = f[a * 3 + c] - acc[c];
		if (Zs && sigma != 0.f) r[c] -= sigma * Zs[a * 3 + c] * x[a * 3 + c];
	}
#pragma unroll
	for (int c = 0; c < 3; c++)
		x[a * 3 + c] += Dinv[a * 9 + c * 3 + 0] * r[0] + Dinv[a * 9 + c * 3 + 1] * r[1] + Dinv[a * 9 + c * 3 + 2] * r[2];
}

// One block of 64 threads per coarse voxel of colour colorC; thread = (fine voxel, Gauss point).
// The trilinear prolongation reproduces the coarse shape functions exactly, so the Galerkin coarse
// element matrix is the coarse-shape-function integral of the fine tangent with the fine quadrature.
__global__ void k_restrict_tangent(int n, int colorC, size_t ne, Material m, const float* s, const float* gam,
	const float* Hc, float om, float* S1) {
	__shared__ float sA[64][81];
	__shared__ float sN[64][8][3];
	int nc = n / 2, hc = nc / 2;
	size_t Nc = size_t(nc) * nc * nc;
	int b = blockIdx.x;
	int I = 2 * (b % hc) + (colorC & 1);
	int J = 2 * ((b / hc) % hc) + ((colorC >> 1) & 1);
	int K = 2 * (b / (hc * hc)) + (colorC >> 2);
	int t = threadIdx.x;
	int fe = t >> 3, g = t & 7;
	int fa = fe & 1, fb = (fe >> 1) & 1, fc = fe >> 2;
	size_t e = nid(2 * I + fa, 2 * J + fb, 2 * K + fc, n);
	float xi[3] = {0.5f * (fa + c_q[g & 1]), 0.5f * (fb + c_q[(g >> 1) & 1]), 0.5f * (fc + c_q[g >> 2])};
	for (int L = 0; L < 8; L++) {
		int aL[3] = {L & 1, (L >> 1) & 1, L >> 2};
		float N[3], dn[3];
		for (int d = 0; d < 3; d++) {
			N[d] = aL[d] ? xi[d] : 1.f - xi[d];
			dn[d] = aL[d] ? 1.f : -1.f;
		}
		sN[t][L][0] = float(nc) * dn[0] * N[1] * N[2];
		sN[t][L][1] = float(nc) * N[0] * dn[1] * N[2];
		sN[t][L][2] = float(nc) * N[0] * N[1] * dn[2];
	}
	float D[9];
	load_D(Hc, ne, e, g, D);
	GPState st;
	bool ok = gp_prepare(D, gam[e], st);
	float w = om * s[e];
	for (int kl = 0; kl < 9; kl++) {
		float H[9], Y[9];
#pragma unroll
		for (int c = 0; c < 9; c++) H[c] = (c == kl) ? 1.f : 0.f;
		if (ok) tangent(m, st, H, Y);
		else lin_tangent(m, H, Y);
#pragma unroll
		for (int ij = 0; ij < 9; ij++) sA[t][ij * 9 + kl] = w * Y[ij];
	}
	__syncthreads();
	for (int ent = t; ent < 576; ent += 64) {
		int row = ent / 24, col = ent % 24;
		int L = row / 3, i = row % 3, M = col / 3, k = col % 3;
		float v = 0.f;
		for (int gg = 0; gg < 64; gg++) {
#pragma unroll
			for (int jj = 0; jj < 3; jj++) {
				float nl = sN[gg][L][jj];
#pragma unroll
				for (int ll = 0; ll < 3; ll++) v += nl * sA[gg][(i * 3 + jj) * 9 + k * 3 + ll] * sN[gg][M][ll];
			}
		}
		int aL = L & 1, bL = (L >> 1) & 1, cL = L >> 2;
		int aM = M & 1, bM = (M >> 1) & 1, cM = M >> 2;
		size_t nodeL = nid(wrapi(I + aL, nc), wrapi(J + bL, nc), wrapi(K + cL, nc), nc);
		int oi = (aM - aL + 1) + 3 * (bM - bL + 1) + 9 * (cM - cL + 1);
		S1[size_t(oi * 9 + i * 3 + k) * Nc + nodeL] += v;
	}
}

__device__ inline float pw(int d) { return d == 0 ? 1.f : 0.5f; }

__global__ void k_restrict_shift(int nf, const float* Zs, float sigma, float* S1) {
	int nc = nf / 2;
	size_t Nc = size_t(nc) * nc * nc;
	size_t A = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (A >= Nc) return;
	int I = int(A % nc), J = int((A / nc) % nc), K = int(A / (size_t(nc) * nc));
	for (int da = 0; da < 27; da++) {
		int dx = da % 3 - 1, dy = (da / 3) % 3 - 1, dz = da / 9 - 1;
		size_t a = nid(wrapi(2 * I + dx, nf), wrapi(2 * J + dy, nf), wrapi(2 * K + dz, nf), nf);
		float wa = pw(dx) * pw(dy) * pw(dz);
		for (int oi = 0; oi < 27; oi++) {
			int ox = oi % 3 - 1, oy = (oi / 3) % 3 - 1, oz = oi / 9 - 1;
			int bx = dx - 2 * ox, by = dy - 2 * oy, bz = dz - 2 * oz;
			if (abs(bx) > 1 || abs(by) > 1 || abs(bz) > 1) continue;
			float wab = sigma * wa * pw(bx) * pw(by) * pw(bz);
			for (int c = 0; c < 3; c++) S1[size_t(oi * 9 + c * 4) * Nc + A] += wab * Zs[a * 3 + c];
		}
	}
}

__global__ void k_sensitivity(int n, size_t ne, Material m, const float* s, const float* ds, const float* gam,
	const float* dgam, const float* Hc, float om, const float* lamw, F9 LamBar, F9 aJ, double* dJ) {
	size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (e >= ne) return;
	int i = int(e % n), j = int((e / n) % n), k = int(e / (size_t(n) * n));
	size_t nd[8];
	elem_nodes(n, i, j, k, nd);
	float le[24];
#pragma unroll
	for (int l = 0; l < 8; l++)
#pragma unroll
		for (int a = 0; a < 3; a++) le[l * 3 + a] = lamw ? lamw[nd[l] * 3 + a] : 0.f;
	float se = s[e], dse = ds[e], ge = gam[e], dge = dgam[e];
	double acc = 0.0;
	for (int g = 0; g < 8; g++) {
		float D[9], P[9], dPg[9], Lam[9];
		load_D(Hc, ne, e, g, D);
		GPState st;
		if (!gp_prepare(D, ge, st)) continue;
		stress(m, st, P);
		stress_dgamma(m, st, dPg);
		grad_at(le, g, float(n), Lam);
		float v = 0.f;
#pragma unroll
		for (int c = 0; c < 9; c++) {
			float dsP = dse * P[c] + se * dge * dPg[c];
			v += dsP * (aJ.v[c] - (Lam[c] + LamBar.v[c]));
		}
		acc += double(om) * double(v);
	}
	dJ[e] += acc;
}

__global__ void k_moduli(int n, size_t ne, Material m, const float* s, const float* gam, float om, Ptr6 chi, int toec,
	double* out) {
	size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	double v[77];
#pragma unroll
	for (int c = 0; c < 77; c++) v[c] = 0.0;
	if (e < ne) {
		int i = int(e % n), j = int((e / n) % n), k = int(e / (size_t(n) * n));
		size_t nd[8];
		elem_nodes(n, i, j, k, nd);
		float w = om * s[e], ge = gam[e];
		float g3 = ge * ge * ge;
		for (int g = 0; g < 8; g++) {
			float h[6][9];
			for (int I = 0; I < 6; I++) {
				float xe[24];
				for (int l = 0; l < 8; l++)
					for (int a = 0; a < 3; a++) xe[l * 3 + a] = chi.p[I][nd[l] * 3 + a];
				float E[9];
				voigt_E(I, E);
				grad_at(xe, g, float(n), h[I]);
				for (int c = 0; c < 9; c++) h[I][c] += E[c];
			}
			int q = 0;
			for (int I = 0; I < 6; I++) {
				float Y[9];
				lin_tangent(m, h[I], Y);
				for (int J = I; J < 6; J++) {
					float d = 0.f;
					for (int c = 0; c < 9; c++) d += Y[c] * h[J][c];
					v[q++] += double(w * d);
				}
			}
			if (toec) {
				for (int I = 0; I < 6; I++)
					for (int J = I; J < 6; J++)
						for (int K = J; K < 6; K++) v[q++] += double(w * g3 * nh_third(m, h[I], h[J], h[K]));
			}
		}
	}
	block_sum_atomic<77>(v, out);
}

__global__ void k_moduli_sens(int n, size_t ne, Material m, const float* ds, float om, Ptr6 chi, F36 W, double* dJ) {
	size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (e >= ne) return;
	int i = int(e % n), j = int((e / n) % n), k = int(e / (size_t(n) * n));
	size_t nd[8];
	elem_nodes(n, i, j, k, nd);
	double acc = 0.0;
	for (int g = 0; g < 8; g++) {
		float h[6][9];
		for (int I = 0; I < 6; I++) {
			float xe[24];
			for (int l = 0; l < 8; l++)
				for (int a = 0; a < 3; a++) xe[l * 3 + a] = chi.p[I][nd[l] * 3 + a];
			float E[9];
			voigt_E(I, E);
			grad_at(xe, g, float(n), h[I]);
			for (int c = 0; c < 9; c++) h[I][c] += E[c];
		}
		float vg = 0.f;
		for (int I = 0; I < 6; I++) {
			float Y[9];
			lin_tangent(m, h[I], Y);
			for (int J = 0; J < 6; J++) {
				float d = 0.f;
				for (int c = 0; c < 9; c++) d += Y[c] * h[J][c];
				vg += W.v[I * 6 + J] * d;
			}
		}
		acc += double(om) * double(ds[e]) * double(vg);
	}
	dJ[e] += acc;
}

__global__ void k_tile(int n, const float* w, float* w2) {
	int n2 = 2 * n;
	size_t N2 = size_t(n2) * n2 * n2;
	size_t a = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (a >= N2) return;
	int i = int(a % n2), j = int((a / n2) % n2), k = int(a / (size_t(n2) * n2));
	size_t b = nid(i % n, j % n, k % n, n);
	for (int c = 0; c < 3; c++) w2[a * 3 + c] = w[b * 3 + c];
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------------------------

void voigt_basis(int I, double E[9]) {
	for (int c = 0; c < 9; c++) E[c] = 0;
	switch (I) {
	case 0: E[0] = 1; break;
	case 1: E[4] = 1; break;
	case 2: E[8] = 1; break;
	case 3: E[5] = E[7] = 0.5; break;
	case 4: E[2] = E[6] = 0.5; break;
	default: E[1] = E[3] = 0.5; break;
	}
}

void cell_alloc(Cell& c, int n, const Material& m) {
	init_constants();
	c.n = n;
	c.ne = size_t(n) * n * n;
	c.mat = m;
	FS_CUDA(cudaMalloc(&c.rho, c.ne * sizeof(float)));
	FS_CUDA(cudaMalloc(&c.s, c.ne * sizeof(float)));
	FS_CUDA(cudaMalloc(&c.ds, c.ne * sizeof(float)));
	FS_CUDA(cudaMalloc(&c.gam, c.ne * sizeof(float)));
	FS_CUDA(cudaMalloc(&c.dgam, c.ne * sizeof(float)));
	FS_CUDA(cudaMalloc(&c.isvoid, c.ne));
	FS_CUDA(cudaMalloc(&c.Hc, 72 * c.ne * sizeof(float)));
	FS_CUDA(cudaMemset(c.Hc, 0, 72 * c.ne * sizeof(float)));
}

void cell_free(Cell& c) {
	cudaFree(c.rho);
	cudaFree(c.s);
	cudaFree(c.ds);
	cudaFree(c.gam);
	cudaFree(c.dgam);
	cudaFree(c.isvoid);
	cudaFree(c.Hc);
	c = Cell();
}

void cell_set_density(Cell& c, const float* rho_host) {
	FS_CUDA(cudaMemcpy(c.rho, rho_host, c.ne * sizeof(float), cudaMemcpyHostToDevice));
	k_set_density<<<nblocks(c.ne, 256), 256>>>(c.ne, c.mat, c.rho, c.s, c.ds, c.gam, c.dgam, c.isvoid);
	FS_KERNEL_CHECK();
	long long nv = 0;
	for (size_t e = 0; e < c.ne; e++) {
		float r = std::fmin(std::fmax(rho_host[e], 0.f), 1.f);
		if (std::pow(r, c.mat.p) < c.mat.rho0) nv++;
	}
	c.nvoid = nv;
}

static float omega(const Cell& c) { return float(1.0 / (8.0 * double(c.ne))); }

void cell_cache_update(const Cell& c, const float* w, const double FbarMinusI[9]) {
	k_cache<<<nblocks(c.ne, 128), 128>>>(c.n, c.ne, w, make_f9(FbarMinusI), c.Hc);
	FS_KERNEL_CHECK();
}

void cell_energy(const Cell& c, EnergyOut& out) {
	double* d = scratch();
	FS_CUDA(cudaMemset(d, 0, 4 * sizeof(double)));
	k_energy<<<nblocks(c.ne, 256), 256>>>(c.n, c.ne, c.mat, c.s, c.gam, c.isvoid, c.Hc, omega(c), d);
	FS_KERNEL_CHECK();
	double h[4];
	FS_CUDA(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
	out.energy = h[0];
	out.absEnergy = h[1];
	out.badJ = (long long)(h[2] + 0.5);
	out.voidDistorted = (long long)(h[3] + 0.5);
}

void cell_residual(const Cell& c, float* r, double Pbar[9]) {
	FS_CUDA(cudaMemset(r, 0, 3 * c.ne * sizeof(float)));
	double* d = scratch();
	FS_CUDA(cudaMemset(d, 0, 9 * sizeof(double)));
	size_t n8 = c.ne / 8;
	for (int color = 0; color < 8; color++) {
		k_residual<<<nblocks(n8, 128), 128>>>(c.n, color, c.ne, c.mat, c.s, c.gam, c.Hc, omega(c), r, d);
		FS_KERNEL_CHECK();
	}
	if (Pbar) FS_CUDA(cudaMemcpy(Pbar, d, 9 * sizeof(double), cudaMemcpyDeviceToHost));
}

void cell_product(const Cell& c, const float* x, const double dF[9], float* y, double t[9]) {
	FS_CUDA(cudaMemset(y, 0, 3 * c.ne * sizeof(float)));
	double* d = t ? scratch() : nullptr;
	if (d) FS_CUDA(cudaMemset(d, 0, 9 * sizeof(double)));
	size_t n8 = c.ne / 8;
	F9 f = make_f9(dF);
	for (int color = 0; color < 8; color++) {
		k_product<<<nblocks(n8, 128), 128>>>(c.n, color, c.ne, c.mat, c.s, c.gam, c.Hc, omega(c), x, f, y, d);
		FS_KERNEL_CHECK();
	}
	if (t) FS_CUDA(cudaMemcpy(t, d, 9 * sizeof(double), cudaMemcpyDeviceToHost));
}

int cell_diag_blocks(const Cell& c, const float* Zs, float sigma, float* Dinv, float* diag) {
	static int* dcount = nullptr;
	if (!dcount) FS_CUDA(cudaMalloc(&dcount, sizeof(int)));
	FS_CUDA(cudaMemset(dcount, 0, sizeof(int)));
	k_diag<<<nblocks(c.ne, 128), 128>>>(c.n, c.ne, c.mat, c.s, c.gam, c.Hc, omega(c), Zs, sigma, Dinv, diag, dcount);
	FS_KERNEL_CHECK();
	int h = 0;
	FS_CUDA(cudaMemcpy(&h, dcount, sizeof(int), cudaMemcpyDeviceToHost));
	return h;
}

void cell_gs(const Cell& c, int color, const float* f, float* x, const float* Dinv, const float* Zs, float sigma) {
	size_t n8 = c.ne / 8;
	k_gs_fine<<<nblocks(n8, 128), 128>>>(c.n, color, c.ne, c.mat, c.s, c.gam, c.Hc, omega(c), f, x, Dinv, Zs, sigma);
	FS_KERNEL_CHECK();
}

void cell_restrict_tangent(const Cell& c, float* S1) {
	int hc = c.n / 4;
	int nb = hc * hc * hc;
	for (int color = 0; color < 8; color++) {
		k_restrict_tangent<<<nb, 64>>>(c.n, color, c.ne, c.mat, c.s, c.gam, c.Hc, omega(c), S1);
		FS_KERNEL_CHECK();
	}
}

void restrict_shift(int nf, const float* Zs, float sigma, float* S1) {
	size_t Nc = size_t(nf / 2) * (nf / 2) * (nf / 2);
	k_restrict_shift<<<nblocks(Nc, 128), 128>>>(nf, Zs, sigma, S1);
	FS_KERNEL_CHECK();
}

void cell_sensitivity(const Cell& c, const float* lamw, const double LamBar[9], const double a[9], double* dJ) {
	k_sensitivity<<<nblocks(c.ne, 128), 128>>>(c.n, c.ne, c.mat, c.s, c.ds, c.gam, c.dgam, c.Hc, omega(c), lamw,
		make_f9(LamBar), make_f9(a), dJ);
	FS_KERNEL_CHECK();
}

void cell_linear_moduli(const Cell& c, float* const chi[6], double C[36], double G56[56], bool toec) {
	double* d = scratch();
	FS_CUDA(cudaMemset(d, 0, 77 * sizeof(double)));
	Ptr6 p;
	for (int I = 0; I < 6; I++) p.p[I] = chi[I];
	k_moduli<<<nblocks(c.ne, 128), 128>>>(c.n, c.ne, c.mat, c.s, c.gam, omega(c), p, toec ? 1 : 0, d);
	FS_KERNEL_CHECK();
	double h[77];
	FS_CUDA(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
	int q = 0;
	for (int I = 0; I < 6; I++)
		for (int J = I; J < 6; J++) {
			C[I * 6 + J] = h[q];
			C[J * 6 + I] = h[q];
			q++;
		}
	if (G56)
		for (int t = 0; t < 56; t++) G56[t] = toec ? h[21 + t] : 0.0;
}

void cell_moduli_sensitivity(const Cell& c, float* const chi[6], const double W[36], double* dJ) {
	Ptr6 p;
	for (int I = 0; I < 6; I++) p.p[I] = chi[I];
	F36 w;
	for (int t = 0; t < 36; t++) w.v[t] = float(W[t]);
	k_moduli_sens<<<nblocks(c.ne, 128), 128>>>(c.n, c.ne, c.mat, c.ds, omega(c), p, w, dJ);
	FS_KERNEL_CHECK();
}

void tile_vector(int n, const float* w, float* w2) {
	size_t N2 = size_t(8) * n * n * n;
	k_tile<<<nblocks(N2, 256), 256>>>(n, w, w2);
	FS_KERNEL_CHECK();
}

}  // namespace fs
