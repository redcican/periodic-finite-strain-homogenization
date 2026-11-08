// Multigrid hierarchy with scaled FP16/FP32 coarse stencils and an FP64 dense coarsest solve.
#include "fs_mg.h"
#include "fs_reduce.cuh"

#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace fs {

namespace {

template <typename T>
__device__ inline float ldv(const T* p, size_t i);
template <>
__device__ inline float ldv<float>(const float* p, size_t i) {
	return p[i];
}
template <>
__device__ inline float ldv<__half>(const __half* p, size_t i) {
	return __half2float(p[i]);
}

template <typename T>
__device__ inline T stv(float v);
template <>
__device__ inline float stv<float>(float v) {
	return v;
}
template <>
__device__ inline __half stv<__half>(float v) {
	return __float2half_rn(v);
}

__device__ inline size_t nbr(int n, int i, int j, int k, int oi) {
	return nid(wrapi(i + oi % 3 - 1, n), wrapi(j + (oi / 3) % 3 - 1, n), wrapi(k + oi / 9 - 1, n), n);
}

__device__ inline void ijk(size_t a, int n, int& i, int& j, int& k) {
	i = int(a % n);
	j = int((a / n) % n);
	k = int(a / (size_t(n) * n));
}

__device__ inline float pwt(int d) { return d == 0 ? 1.f : 0.5f; }

__device__ inline bool inv3(const float* K, float* Ki) {
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
	bool pd = K[0] > 0.f && c22 > 0.f && det > 0.f;
	float id = det != 0.f ? 1.f / det : 0.f;
	Ki[0] = c00 * id; Ki[1] = c01 * id; Ki[2] = c02 * id;
	Ki[3] = c10 * id; Ki[4] = c11 * id; Ki[5] = c12 * id;
	Ki[6] = c20 * id; Ki[7] = c21 * id; Ki[8] = c22 * id;
	return pd;
}

__global__ void k_extract_diag(size_t N, const float* S32, float* sqZ, int* notpd) {
	size_t A = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (A >= N) return;
	for (int c = 0; c < 3; c++) {
		float z = S32[size_t(13 * 9 + c * 4) * N + A];
		if (!(z > 0.f)) {
			atomicAdd(notpd, 1);
			z = 1e-30f;
		}
		sqZ[A * 3 + c] = sqrtf(z);
	}
}

template <typename T>
__global__ void k_scale(int n, const float* S32, const float* sqZ, T* ST) {
	size_t N = size_t(n) * n * n;
	size_t A = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (A >= N) return;
	int i, j, k;
	ijk(A, n, i, j, k);
	for (int oi = 0; oi < 27; oi++) {
		size_t B = nbr(n, i, j, k, oi);
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++) {
				size_t idx = size_t(oi * 9 + a * 3 + b) * N + A;
				ST[idx] = stv<T>(S32[idx] / (sqZ[A * 3 + a] * sqZ[B * 3 + b]));
			}
	}
}

template <typename T>
__global__ void k_dinv(size_t N, const T* ST, const float* sqZ, float* Dinv, int* notpd) {
	size_t A = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (A >= N) return;
	float K[9], Ki[9];
	for (int a = 0; a < 3; a++)
		for (int b = 0; b < 3; b++)
			K[a * 3 + b] = ldv(ST, size_t(13 * 9 + a * 3 + b) * N + A) * sqZ[A * 3 + a] * sqZ[A * 3 + b];
	if (!inv3(K, Ki)) atomicAdd(notpd, 1);
	for (int c = 0; c < 9; c++) Dinv[A * 9 + c] = Ki[c];
}

template <typename T>
__device__ inline void stencil_apply(int n, size_t N, const T* ST, const float* sqZ, const float* x, size_t A, int i, int j,
	int k, float* acc) {
	acc[0] = acc[1] = acc[2] = 0.f;
	for (int oi = 0; oi < 27; oi++) {
		size_t B = nbr(n, i, j, k, oi);
		float y[3] = {sqZ[B * 3 + 0] * x[B * 3 + 0], sqZ[B * 3 + 1] * x[B * 3 + 1], sqZ[B * 3 + 2] * x[B * 3 + 2]};
#pragma unroll
		for (int a = 0; a < 3; a++) {
			size_t base = size_t(oi * 9 + a * 3) * N + A;
			acc[a] += ldv(ST, base) * y[0] + ldv(ST, base + N) * y[1] + ldv(ST, base + 2 * N) * y[2];
		}
	}
#pragma unroll
	for (int a = 0; a < 3; a++) acc[a] *= sqZ[A * 3 + a];
}

template <typename T>
__global__ void k_gs(int n, int color, const T* ST, const float* sqZ, const float* Dinv, const float* f, float* x) {
	int h = n / 2;
	size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (t >= size_t(h) * h * h) return;
	int i = 2 * int(t % h) + (color & 1);
	int j = 2 * int((t / h) % h) + ((color >> 1) & 1);
	int k = 2 * int(t / (size_t(h) * h)) + (color >> 2);
	size_t N = size_t(n) * n * n;
	size_t A = nid(i, j, k, n);
	float acc[3];
	stencil_apply(n, N, ST, sqZ, x, A, i, j, k, acc);
	float r[3] = {f[A * 3 + 0] - acc[0], f[A * 3 + 1] - acc[1], f[A * 3 + 2] - acc[2]};
#pragma unroll
	for (int a = 0; a < 3; a++)
		x[A * 3 + a] += Dinv[A * 9 + a * 3 + 0] * r[0] + Dinv[A * 9 + a * 3 + 1] * r[1] + Dinv[A * 9 + a * 3 + 2] * r[2];
}

template <typename T>
__global__ void k_resid(int n, const T* ST, const float* sqZ, const float* f, const float* x, float* r) {
	size_t N = size_t(n) * n * n;
	size_t A = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (A >= N) return;
	int i, j, k;
	ijk(A, n, i, j, k);
	float acc[3];
	stencil_apply(n, N, ST, sqZ, x, A, i, j, k, acc);
	for (int a = 0; a < 3; a++) r[A * 3 + a] = f[A * 3 + a] - acc[a];
}

__global__ void k_restrict_vec(int nf, const float* rf, float* rc) {
	int nc = nf / 2;
	size_t Nc = size_t(nc) * nc * nc;
	size_t A = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (A >= Nc) return;
	int I, J, K;
	ijk(A, nc, I, J, K);
	float acc[3] = {0.f, 0.f, 0.f};
	for (int da = 0; da < 27; da++) {
		int dx = da % 3 - 1, dy = (da / 3) % 3 - 1, dz = da / 9 - 1;
		size_t a = nid(wrapi(2 * I + dx, nf), wrapi(2 * J + dy, nf), wrapi(2 * K + dz, nf), nf);
		float w = pwt(dx) * pwt(dy) * pwt(dz);
		for (int c = 0; c < 3; c++) acc[c] += w * rf[a * 3 + c];
	}
	for (int c = 0; c < 3; c++) rc[A * 3 + c] = acc[c];
}

__global__ void k_prolong_add(int nc, const float* xc, float* xf) {
	int nf = 2 * nc;
	size_t Nf = size_t(nf) * nf * nf;
	size_t a = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (a >= Nf) return;
	int i, j, k;
	ijk(a, nf, i, j, k);
	int ci[3][2], cn[3];
	float cw[3];
	int p[3] = {i, j, k};
	for (int d = 0; d < 3; d++) {
		if (p[d] % 2 == 0) {
			ci[d][0] = p[d] / 2;
			cn[d] = 1;
			cw[d] = 1.f;
		} else {
			ci[d][0] = (p[d] - 1) / 2;
			ci[d][1] = wrapi((p[d] + 1) / 2, nc);
			cn[d] = 2;
			cw[d] = 0.5f;
		}
	}
	float acc[3] = {0.f, 0.f, 0.f};
	for (int x = 0; x < cn[0]; x++)
		for (int y = 0; y < cn[1]; y++)
			for (int z = 0; z < cn[2]; z++) {
				size_t A = nid(ci[0][x], ci[1][y], ci[2][z], nc);
				float w = cw[0] * cw[1] * cw[2];
				for (int c = 0; c < 3; c++) acc[c] += w * xc[A * 3 + c];
			}
	for (int c = 0; c < 3; c++) xf[a * 3 + c] += acc[c];
}

// Galerkin product P^T K P of a stored stencil level (thread per coarse node, FP32 accumulation).
template <typename T>
__global__ void k_restrict_stencil(int nf, const T* ST, const float* sqZ, float* S32c) {
	int nc = nf / 2;
	size_t Nf = size_t(nf) * nf * nf, Nc = size_t(nc) * nc * nc;
	size_t A = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (A >= Nc) return;
	int I, J, K;
	ijk(A, nc, I, J, K);
	float acc[243];
	for (int c = 0; c < 243; c++) acc[c] = 0.f;
	for (int da = 0; da < 27; da++) {
		int d[3] = {da % 3 - 1, (da / 3) % 3 - 1, da / 9 - 1};
		int ai = wrapi(2 * I + d[0], nf), aj = wrapi(2 * J + d[1], nf), ak = wrapi(2 * K + d[2], nf);
		size_t a = nid(ai, aj, ak, nf);
		float wa = pwt(d[0]) * pwt(d[1]) * pwt(d[2]);
		for (int ob = 0; ob < 27; ob++) {
			int o[3] = {ob % 3 - 1, (ob / 3) % 3 - 1, ob / 9 - 1};
			size_t b = nbr(nf, ai, aj, ak, ob);
			float Kab[9];
			for (int p = 0; p < 3; p++)
				for (int q = 0; q < 3; q++)
					Kab[p * 3 + q] = ldv(ST, size_t(ob * 9 + p * 3 + q) * Nf + a) * sqZ[a * 3 + p] * sqZ[b * 3 + q];
			// coarse offsets oc with |d + o - 2 oc| <= 1 in every direction
			int cand[3][2], ncand[3];
			float cwt[3][2];
			for (int dd = 0; dd < 3; dd++) {
				int s = d[dd] + o[dd];
				ncand[dd] = 0;
				for (int oc = -1; oc <= 1; oc++) {
					int db = s - 2 * oc;
					if (db >= -1 && db <= 1) {
						cand[dd][ncand[dd]] = oc;
						cwt[dd][ncand[dd]] = pwt(db);
						ncand[dd]++;
					}
				}
			}
			for (int x = 0; x < ncand[0]; x++)
				for (int y = 0; y < ncand[1]; y++)
					for (int z = 0; z < ncand[2]; z++) {
						int oi = (cand[0][x] + 1) + 3 * (cand[1][y] + 1) + 9 * (cand[2][z] + 1);
						float w = wa * cwt[0][x] * cwt[1][y] * cwt[2][z];
						for (int c = 0; c < 9; c++) acc[oi * 9 + c] += w * Kab[c];
					}
		}
	}
	for (int c = 0; c < 243; c++) S32c[size_t(c) * Nc + A] = acc[c];
}

__global__ void k_dense(int nc, const float* S32, double* Kd, int M) {
	size_t Nc = size_t(nc) * nc * nc;
	size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (t >= Nc * 27) return;
	size_t A = t % Nc;
	int oi = int(t / Nc);
	int i, j, k;
	ijk(A, nc, i, j, k);
	size_t B = nbr(nc, i, j, k, oi);
	for (int a = 0; a < 3; a++)
		for (int b = 0; b < 3; b++)
			Kd[(A * 3 + a) + (B * 3 + b) * size_t(M)] = double(S32[size_t(oi * 9 + a * 3 + b) * Nc + A]);
}

// dsc = 1 / sqrt(diag K); counts non-positive diagonal entries.
__global__ void k_dense_diag(const double* Kd, int M, double* dsc, int* notpd) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M) return;
	double d = Kd[i + size_t(i) * M];
	if (!(d > 0)) {
		atomicAdd(notpd, 1);
		d = 1.0;
	}
	dsc[i] = 1.0 / sqrt(d);
}

__global__ void k_dense_scale(double* Kd, int M, const double* dsc) {
	size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (t >= size_t(M) * M) return;
	int r = int(t % M), col = int(t / M);
	Kd[t] *= dsc[r] * dsc[col];
}

// |q_c|^2 = sum_A d_(A, c) for the three scaled translations q_c = D^{1/2} 1_c (one block).
__global__ void k_qnorm(const double* dsc, int M, double* out) {
	double v[3] = {0.0, 0.0, 0.0};
	for (int i = threadIdx.x; i < M; i += blockDim.x) v[i % 3] += 1.0 / (dsc[i] * dsc[i]);
	if (threadIdx.x < 3) out[threadIdx.x] = 0.0;
	__syncthreads();
	block_sum_atomic<3>(v, out);
}

// Kd += sum_c q_c q_c^T / |q_c|^2 with (q_c)_(A, c) = sqrt(d_(A, c)) = 1 / dsc.
__global__ void k_dense_addQ(double* Kd, int M, const double* dsc, double q0, double q1, double q2) {
	size_t t = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (t >= size_t(M) * M) return;
	int r = int(t % M), col = int(t / M);
	int c = r % 3;
	if (c != col % 3) return;
	double qn = c == 0 ? q0 : (c == 1 ? q1 : q2);
	Kd[t] += 1.0 / (dsc[r] * dsc[col] * qn);
}

__global__ void k_f2d(const float* f, double* d, const double* dsc, int M) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < M) d[i] = double(f[i]) * dsc[i];
}

__global__ void k_d2f(const double* d, float* f, const double* dsc, int M) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < M) f[i] = float(d[i] * dsc[i]);
}

int* counter() {
	static int* p = nullptr;
	if (!p) FS_CUDA(cudaMalloc(&p, sizeof(int)));
	return p;
}

const bool debug = std::getenv("FS_DEBUG") != nullptr;

int read_counter() {
	int h = 0;
	FS_CUDA(cudaMemcpy(&h, counter(), sizeof(int), cudaMemcpyDeviceToHost));
	return h;
}

}  // namespace

Multigrid::~Multigrid() { release(); }

void Multigrid::release() {
	for (auto& l : lv) {
		dfree(l.sqZ);
		dfree(l.Sh);
		dfree(l.Sf);
		dfree(l.Dinv);
		dfree(l.x);
		dfree(l.f);
		dfree(l.r);
	}
	lv.clear();
	dfree(Dinv0);
	dfree(f0);
	dfree(r0);
	dfree(ws);
	dfree(Kd);
	dfree(dwork);
	dfree(drhs);
	dfree(dsc);
	dfree(dinfo);
	Dinv0 = f0 = r0 = ws = nullptr;
	Kd = dwork = drhs = dsc = nullptr;
	dinfo = nullptr;
	if (handle) cusolverDnDestroy(static_cast<cusolverDnHandle_t>(handle));
	handle = nullptr;
	built = false;
	nf = L = 0;
}

void Multigrid::init(int n) {
	release();
	int l = 0, m = n;
	while (m > 8 && m % 2 == 0) {
		m /= 2;
		l++;
	}
	if (m != 8 || l < 1) {
		std::fprintf(stderr, "multigrid: the resolution must be 8 * 2^L with L >= 1 (16, 32, 64, ...), got %d\n", n);
		std::exit(1);
	}
	nf = n;
	L = l;
	size_t N0 = size_t(n) * n * n;
	Dinv0 = dalloc(9 * N0);
	f0 = dalloc(3 * N0);
	r0 = dalloc(3 * N0);
	lv.resize(L + 1);
	for (int k = 1; k <= L; k++) {
		Level& v = lv[k];
		v.n = n >> k;
		v.N = size_t(v.n) * v.n * v.n;
		v.x = dalloc(3 * v.N);
		v.f = dalloc(3 * v.N);
		v.r = dalloc(3 * v.N);
		if (k < L) {
			v.sqZ = dalloc(3 * v.N);
			v.Dinv = dalloc(9 * v.N);
			FS_CUDA(cudaMalloc(&v.Sh, 243 * v.N * sizeof(__half)));
		}
	}
	ws = dalloc(243 * lv[1].N);
	Md = int(3 * lv[L].N);
	FS_CUDA(cudaMalloc(&Kd, size_t(Md) * Md * sizeof(double)));
	FS_CUDA(cudaMalloc(&drhs, size_t(Md) * sizeof(double)));
	FS_CUDA(cudaMalloc(&dsc, size_t(Md) * sizeof(double)));
	FS_CUDA(cudaMalloc(&dinfo, sizeof(int)));
	cusolverDnHandle_t h;
	if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) {
		std::fprintf(stderr, "multigrid: cusolverDnCreate failed\n");
		std::exit(1);
	}
	handle = h;
	if (cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, Md, Kd, Md, &lwork) != CUSOLVER_STATUS_SUCCESS) {
		std::fprintf(stderr, "multigrid: potrf buffer query failed\n");
		std::exit(1);
	}
	FS_CUDA(cudaMalloc(&dwork, size_t(lwork > 0 ? lwork : 1) * sizeof(double)));
}

size_t Multigrid::coarse_bytes() const {
	size_t b = 0;
	for (int k = 1; k < L; k++) {
		const Level& v = lv[k];
		b += 243 * v.N * sizeof(__half) + 12 * v.N * sizeof(float);
		if (v.Sf) b += 243 * v.N * sizeof(float);
	}
	if (L >= 1) b += 243 * lv[1].N * sizeof(float);
	b += size_t(Md) * Md * sizeof(double) + size_t(lwork) * sizeof(double);
	return b;
}

bool Multigrid::store_level(int l, const float* S32) {
	Level& v = lv[l];
	FS_CUDA(cudaMemset(counter(), 0, sizeof(int)));
	k_extract_diag<<<nblocks(v.N, 256), 256>>>(v.N, S32, v.sqZ, counter());
	FS_KERNEL_CHECK();
	if (prec_ == Prec::FP32) {
		if (!v.Sf) v.Sf = dalloc(243 * v.N);
		k_scale<float><<<nblocks(v.N, 128), 128>>>(v.n, S32, v.sqZ, v.Sf);
		FS_KERNEL_CHECK();
		k_dinv<float><<<nblocks(v.N, 256), 256>>>(v.N, v.Sf, v.sqZ, v.Dinv, counter());
	} else {
		k_scale<__half><<<nblocks(v.N, 128), 128>>>(v.n, S32, v.sqZ, v.Sh);
		FS_KERNEL_CHECK();
		k_dinv<__half><<<nblocks(v.N, 256), 256>>>(v.N, v.Sh, v.sqZ, v.Dinv, counter());
	}
	FS_KERNEL_CHECK();
	return read_counter() == 0;
}

bool Multigrid::factor_dense(const float* S32) {
	Level& v = lv[L];
	FS_CUDA(cudaMemset(Kd, 0, size_t(Md) * Md * sizeof(double)));
	k_dense<<<nblocks(v.N * 27, 256), 256>>>(v.n, S32, Kd, Md);
	FS_KERNEL_CHECK();
	// Symmetric Jacobi scaling D^{-1/2} K D^{-1/2}: the void stiffness (rho_min) is far below the FP32
	// rounding of the solid entries, so the unscaled matrix is indefinite at round-off level; in the
	// scaled matrix every mode has O(1) weight and only the three translations are (near-)null.
	FS_CUDA(cudaMemset(counter(), 0, sizeof(int)));
	k_dense_diag<<<nblocks(Md, 256), 256>>>(Kd, Md, dsc, counter());
	FS_KERNEL_CHECK();
	if (read_counter() != 0) return false;
	k_dense_scale<<<nblocks(size_t(Md) * Md, 256), 256>>>(Kd, Md, dsc);
	FS_KERNEL_CHECK();
	// The scaled translations q_c = D^{1/2} 1_c span the null space: add q_c q_c^T / |q_c|^2 (unit diagonal
	// scale) so that the factorization exists; right-hand sides orthogonal to 1_c are solved exactly.
	double qn[3];
	k_qnorm<<<1, 256>>>(dsc, Md, drhs);
	FS_KERNEL_CHECK();
	FS_CUDA(cudaMemcpy(qn, drhs, 3 * sizeof(double), cudaMemcpyDeviceToHost));
	for (int c = 0; c < 3; c++)
		if (!(qn[c] > 0)) return false;
	k_dense_addQ<<<nblocks(size_t(Md) * Md, 256), 256>>>(Kd, Md, dsc, qn[0], qn[1], qn[2]);
	FS_KERNEL_CHECK();
	auto h = static_cast<cusolverDnHandle_t>(handle);
	cusolverStatus_t cs = cusolverDnDpotrf(h, CUBLAS_FILL_MODE_LOWER, Md, Kd, Md, dwork, lwork, dinfo);
	int info = 0;
	FS_CUDA(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (debug && (cs != CUSOLVER_STATUS_SUCCESS || info != 0))
		std::fprintf(stderr, "multigrid: potrf status %d, info %d, M = %d\n", int(cs), info, Md);
	return cs == CUSOLVER_STATUS_SUCCESS && info == 0;
}

void Multigrid::dense_solve(const float* f, float* x) {
	Level& v = lv[L];
	vcopy(f, v.r, 3 * v.N);
	vproject_mean(v.r, v.N);
	k_f2d<<<nblocks(Md, 256), 256>>>(v.r, drhs, dsc, Md);  // D^{-1/2} f
	FS_KERNEL_CHECK();
	auto h = static_cast<cusolverDnHandle_t>(handle);
	cusolverDnDpotrs(h, CUBLAS_FILL_MODE_LOWER, Md, 1, Kd, Md, drhs, Md, dinfo);
	k_d2f<<<nblocks(Md, 256), 256>>>(drhs, x, dsc, Md);  // x = D^{-1/2} y
	FS_KERNEL_CHECK();
	vproject_mean(x, v.N);
}

bool Multigrid::update(const Cell& cell, Prec prec, float sigma, const float* Zs) {
	cell_ = &cell;
	sigma_ = sigma;
	Zs_ = Zs;
	prec_ = prec;
	int bad0 = cell_diag_blocks(cell, Zs, sigma, Dinv0, nullptr);
	bool ok = bad0 == 0;
	if (bad0 && debug) std::fprintf(stderr, "multigrid: %d fine diagonal blocks not positive definite\n", bad0);
	FS_CUDA(cudaMemset(ws, 0, 243 * lv[1].N * sizeof(float)));
	cell_restrict_tangent(cell, ws);
	if (sigma > 0.f && Zs) restrict_shift(nf, Zs, sigma, ws);
	for (int l = 1; l <= L; l++) {
		if (l > 1) {
			Level& c = lv[l];
			FS_CUDA(cudaMemset(ws, 0, 243 * c.N * sizeof(float)));
			const Level& f = lv[l - 1];
			if (prec_ == Prec::FP32) k_restrict_stencil<float><<<nblocks(c.N, 64), 64>>>(f.n, f.Sf, f.sqZ, ws);
			else k_restrict_stencil<__half><<<nblocks(c.N, 64), 64>>>(f.n, f.Sh, f.sqZ, ws);
			FS_KERNEL_CHECK();
		}
		bool lok = l == L ? factor_dense(ws) : store_level(l, ws);
		if (!lok && debug) std::fprintf(stderr, "multigrid: level %d (%d^3) %s not positive definite\n", l, lv[l].n, l == L ? "dense factorization" : "diagonal");
		ok = lok && ok;
	}
	built = true;
	return ok;
}

double Multigrid::selfcheck(const Cell& cell) {
	cell_ = &cell;
	sigma_ = 0.f;
	Zs_ = nullptr;
	Level& c = lv[1];
	size_t N0 = size_t(nf) * nf * nf;
	FS_CUDA(cudaMemset(ws, 0, 243 * c.N * sizeof(float)));
	cell_restrict_tangent(cell, ws);
	std::vector<float> S(243 * c.N), col(3 * c.N);
	FS_CUDA(cudaMemcpy(S.data(), ws, S.size() * sizeof(float), cudaMemcpyDeviceToHost));
	int nc = c.n;
	auto nidh = [&](int i, int j, int k) { return size_t(((i % nc) + nc) % nc) + size_t(nc) * (((j % nc) + nc) % nc + size_t(nc) * (((k % nc) + nc) % nc)); };
	int test[5][3] = {{0, 0, 0}, {3, 0, 0}, {nc / 2, nc / 2, nc / 2}, {nc / 2, nc / 2, 0}, {1, 2, 3}};
	double worst = 0;
	for (int t = 0; t < 5; t++) {
		int Bi = test[t][0] % nc, Bj = test[t][1] % nc, Bk = test[t][2] % nc;
		size_t B = nidh(Bi, Bj, Bk);
		for (int b = 0; b < 3; b++) {
			vzero(c.x, 3 * c.N);
			float one = 1.f;
			FS_CUDA(cudaMemcpy(c.x + B * 3 + b, &one, sizeof(float), cudaMemcpyHostToDevice));
			vzero(f0, 3 * N0);
			k_prolong_add<<<nblocks(N0, 128), 128>>>(nc, c.x, f0);
			FS_KERNEL_CHECK();
			cell_product(cell, f0, nullptr, r0, nullptr);
			k_restrict_vec<<<nblocks(c.N, 128), 128>>>(nf, r0, c.f);
			FS_KERNEL_CHECK();
			FS_CUDA(cudaMemcpy(col.data(), c.f, col.size() * sizeof(float), cudaMemcpyDeviceToHost));
			double dmax = 0, vmax = 0, far = 0;
			std::vector<char> near(c.N, 0);
			for (int oi = 0; oi < 27; oi++) {
				// A = B - offset(oi): B is the neighbour of A at offset oi
				int dx = oi % 3 - 1, dy = (oi / 3) % 3 - 1, dz = oi / 9 - 1;
				size_t A = nidh(Bi - dx, Bj - dy, Bk - dz);
				near[A] = 1;
				for (int a = 0; a < 3; a++) {
					double sv = S[size_t(oi * 9 + a * 3 + b) * c.N + A], cv = col[A * 3 + a];
					dmax = std::max(dmax, std::fabs(sv - cv));
					vmax = std::max(vmax, std::fabs(cv));
				}
			}
			for (size_t A = 0; A < c.N; A++)
				if (!near[A])
					for (int a = 0; a < 3; a++) far = std::max(far, double(std::fabs(col[A * 3 + a])));
			double rel = dmax / std::max(vmax, 1e-300);
			worst = std::max(worst, rel);
			std::printf("  coarse node (%d %d %d) comp %d: |column| %.3e, stencil error %.3e (rel %.2e), outside 27-point %.2e\n", Bi, Bj, Bk, b,
				vmax, dmax, rel, far);
		}
	}
	return worst;
}

void Multigrid::smooth(int l, const float* f, float* x, bool forward) {
	for (int s = 0; s < 8; s++) {
		int color = forward ? s : 7 - s;
		if (l == 0) {
			cell_gs(*cell_, color, f, x, Dinv0, Zs_, sigma_);
		} else {
			Level& v = lv[l];
			size_t h3 = v.N / 8;
			if (prec_ == Prec::FP32) k_gs<float><<<nblocks(h3, 128), 128>>>(v.n, color, v.Sf, v.sqZ, v.Dinv, f, x);
			else k_gs<__half><<<nblocks(h3, 128), 128>>>(v.n, color, v.Sh, v.sqZ, v.Dinv, f, x);
			FS_KERNEL_CHECK();
		}
	}
}

void Multigrid::residual(int l, const float* f, const float* x, float* r) {
	if (l == 0) {
		size_t m = 3 * cell_->ne;
		cell_product(*cell_, x, nullptr, r, nullptr);
		vaxpby(1.0, f, -1.0, r, m);  // r = f - K x
		if (Zs_ && sigma_ > 0.f) vaxmy(-sigma_, Zs_, x, r, m);
	} else {
		Level& v = lv[l];
		if (prec_ == Prec::FP32) k_resid<float><<<nblocks(v.N, 128), 128>>>(v.n, v.Sf, v.sqZ, f, x, r);
		else k_resid<__half><<<nblocks(v.N, 128), 128>>>(v.n, v.Sh, v.sqZ, f, x, r);
		FS_KERNEL_CHECK();
	}
}

void Multigrid::cycle(int l, const float* f, float* x) {
	if (l == L) {
		dense_solve(f, x);
		return;
	}
	int n = l == 0 ? nf : lv[l].n;
	size_t N = size_t(n) * n * n;
	float* r = l == 0 ? r0 : lv[l].r;
	vzero(x, 3 * N);
	smooth(l, f, x, true);
	residual(l, f, x, r);
	Level& c = lv[l + 1];
	k_restrict_vec<<<nblocks(c.N, 128), 128>>>(n, r, c.f);
	FS_KERNEL_CHECK();
	cycle(l + 1, c.f, c.x);
	k_prolong_add<<<nblocks(N, 128), 128>>>(c.n, c.x, x);
	FS_KERNEL_CHECK();
	smooth(l, f, x, false);
}

void Multigrid::vcycle(const float* r, float* z) {
	size_t N0 = size_t(nf) * nf * nf;
	vcopy(r, f0, 3 * N0);
	vproject_mean(f0, N0);
	cycle(0, f0, z);
	vproject_mean(z, N0);
}

}  // namespace fs
