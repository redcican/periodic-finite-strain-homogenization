// Vector operations with FP64 reductions, and the density filter and projection of the design layer.
#include "fs_device.h"
#include "fs_reduce.cuh"

namespace fs {

namespace {

double* rscratch() {
	static double* p = nullptr;
	if (!p) FS_CUDA(cudaMalloc(&p, 16 * sizeof(double)));
	return p;
}

__global__ void k_dot(const float* a, const float* b, size_t m, double* out) {
	double v[1] = {0.0};
	for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < m; i += size_t(gridDim.x) * blockDim.x)
		v[0] += double(a[i]) * double(b[i]);
	block_sum_atomic<1>(v, out);
}

__global__ void k_axpby(float a, const float* x, float b, float* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] = a * x[i] + b * y[i];
}

__global__ void k_axpy(float a, const float* x, float* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] += a * x[i];
}

__global__ void k_scale(float a, float* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] *= a;
}

__global__ void k_mul(const float* a, const float* x, float* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] = a[i] * x[i];
}

__global__ void k_axmy(float a, const float* d, const float* x, float* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] += a * d[i] * x[i];
}

__global__ void k_sqrt(const float* a, float* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] = sqrtf(fmaxf(a[i], 0.f));
}

__global__ void k_rsqrt(const float* a, float* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] = a[i] > 0.f ? rsqrtf(a[i]) : 0.f;
}

__global__ void k_random(float* v, size_t m, unsigned seed) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i >= m) return;
	// integer hash, uniform in [-1, 1]
	unsigned x = unsigned(i) * 2654435761u ^ (seed * 2246822519u + 374761393u);
	x ^= x >> 15;
	x *= 2246822519u;
	x ^= x >> 13;
	x *= 3266489917u;
	x ^= x >> 16;
	v[i] = float(x) * (2.f / 4294967295.f) - 1.f;
}

__global__ void k_comp_sums(const float* v, const float* sq, size_t nodes, double* out) {
	double s[6] = {0, 0, 0, 0, 0, 0};
	for (size_t a = blockIdx.x * size_t(blockDim.x) + threadIdx.x; a < nodes; a += size_t(gridDim.x) * blockDim.x) {
		for (int c = 0; c < 3; c++) {
			double q = sq ? double(sq[a * 3 + c]) : 1.0;
			s[c] += q * double(v[a * 3 + c]);
			s[3 + c] += q * q;
		}
	}
	block_sum_atomic<6>(s, out);
}

__global__ void k_sub_comp(float* v, const float* sq, size_t nodes, float c0, float c1, float c2) {
	size_t a = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (a >= nodes) return;
	float c[3] = {c0, c1, c2};
	for (int k = 0; k < 3; k++) v[a * 3 + k] -= c[k] * (sq ? sq[a * 3 + k] : 1.f);
}

__global__ void k_dscale(double a, double* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] *= a;
}

__global__ void k_dfill(double v, double* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] = v;
}

__global__ void k_dmul(const double* a, const double* x, double* y, size_t m) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i < m) y[i] = a[i] * x[i];
}

__global__ void k_filter(int n, int R, float r, double wsum, const double* in, double* out) {
	size_t N = size_t(n) * n * n;
	size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (e >= N) return;
	int i = int(e % n), j = int((e / n) % n), k = int(e / (size_t(n) * n));
	double acc = 0.0;
	for (int dz = -R; dz <= R; dz++)
		for (int dy = -R; dy <= R; dy++)
			for (int dx = -R; dx <= R; dx++) {
				float d = sqrtf(float(dx * dx + dy * dy + dz * dz));
				float w = r - d;
				if (w <= 0.f) continue;
				int ii = ((i + dx) % n + n) % n, jj = ((j + dy) % n + n) % n, kk = ((k + dz) % n + n) % n;
				acc += double(w) * in[nid(ii, jj, kk, n)];
			}
	out[e] = acc / wsum;
}

__global__ void k_project(size_t m, float beta, float eta, const double* rt, double* rho, double* drho) {
	size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
	if (i >= m) return;
	double tb = tanh(double(beta) * eta), den = tb + tanh(double(beta) * (1.0 - eta));
	double th = tanh(double(beta) * (rt[i] - eta));
	rho[i] = (tb + th) / den;
	drho[i] = double(beta) * (1.0 - th * th) / den;
}

}  // namespace

float* dalloc(size_t m) {
	float* p = nullptr;
	FS_CUDA(cudaMalloc(&p, (m ? m : 1) * sizeof(float)));
	FS_CUDA(cudaMemset(p, 0, (m ? m : 1) * sizeof(float)));
	return p;
}

double* dalloc_d(size_t m) {
	double* p = nullptr;
	FS_CUDA(cudaMalloc(&p, (m ? m : 1) * sizeof(double)));
	FS_CUDA(cudaMemset(p, 0, (m ? m : 1) * sizeof(double)));
	return p;
}

void dfree(void* p) {
	if (p) cudaFree(p);
}

double vdot(const float* a, const float* b, size_t m) {
	double* d = rscratch();
	FS_CUDA(cudaMemset(d, 0, sizeof(double)));
	int nb = nblocks(m, 256);
	if (nb > 1024) nb = 1024;
	if (nb < 1) nb = 1;
	k_dot<<<nb, 256>>>(a, b, m, d);
	FS_KERNEL_CHECK();
	double h;
	FS_CUDA(cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost));
	return h;
}

void vaxpy(double a, const float* x, float* y, size_t m) {
	k_axpy<<<nblocks(m, 256), 256>>>(float(a), x, y, m);
	FS_KERNEL_CHECK();
}

void vaxpby(double a, const float* x, double b, float* y, size_t m) {
	k_axpby<<<nblocks(m, 256), 256>>>(float(a), x, float(b), y, m);
	FS_KERNEL_CHECK();
}

void vcopy(const float* x, float* y, size_t m) {
	if (x != y) FS_CUDA(cudaMemcpy(y, x, m * sizeof(float), cudaMemcpyDeviceToDevice));
}

void vzero(float* y, size_t m) { FS_CUDA(cudaMemset(y, 0, m * sizeof(float))); }

void vscale(double a, float* y, size_t m) {
	k_scale<<<nblocks(m, 256), 256>>>(float(a), y, m);
	FS_KERNEL_CHECK();
}

void vmul(const float* a, const float* x, float* y, size_t m) {
	k_mul<<<nblocks(m, 256), 256>>>(a, x, y, m);
	FS_KERNEL_CHECK();
}

void vaxmy(double a, const float* d, const float* x, float* y, size_t m) {
	k_axmy<<<nblocks(m, 256), 256>>>(float(a), d, x, y, m);
	FS_KERNEL_CHECK();
}

void vsqrt(const float* a, float* y, size_t m) {
	k_sqrt<<<nblocks(m, 256), 256>>>(a, y, m);
	FS_KERNEL_CHECK();
}

void vrsqrt(const float* a, float* y, size_t m) {
	k_rsqrt<<<nblocks(m, 256), 256>>>(a, y, m);
	FS_KERNEL_CHECK();
}

void vrandom(float* v, size_t m, unsigned seed) {
	k_random<<<nblocks(m, 256), 256>>>(v, m, seed);
	FS_KERNEL_CHECK();
}

static void project_comp(float* v, const float* sq, size_t nodes) {
	double* d = rscratch();
	FS_CUDA(cudaMemset(d, 0, 6 * sizeof(double)));
	int nb = nblocks(nodes, 256);
	if (nb > 1024) nb = 1024;
	if (nb < 1) nb = 1;
	k_comp_sums<<<nb, 256>>>(v, sq, nodes, d);
	FS_KERNEL_CHECK();
	double h[6];
	FS_CUDA(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
	float c[3];
	for (int k = 0; k < 3; k++) c[k] = h[3 + k] > 0 ? float(h[k] / h[3 + k]) : 0.f;
	k_sub_comp<<<nblocks(nodes, 256), 256>>>(v, sq, nodes, c[0], c[1], c[2]);
	FS_KERNEL_CHECK();
}

void vproject_mean(float* v, size_t nodes) { project_comp(v, nullptr, nodes); }

void vproject_weighted(float* v, const float* sq, size_t nodes) { project_comp(v, sq, nodes); }

void dzero(double* y, size_t m) { FS_CUDA(cudaMemset(y, 0, m * sizeof(double))); }

void dscale(double a, double* y, size_t m) {
	k_dscale<<<nblocks(m, 256), 256>>>(a, y, m);
	FS_KERNEL_CHECK();
}

void dfill(double v, double* y, size_t m) {
	k_dfill<<<nblocks(m, 256), 256>>>(v, y, m);
	FS_KERNEL_CHECK();
}

void dmul(const double* a, const double* x, double* y, size_t m) {
	k_dmul<<<nblocks(m, 256), 256>>>(a, x, y, m);
	FS_KERNEL_CHECK();
}

void host2d(const double* h, double* d, size_t m) { FS_CUDA(cudaMemcpy(d, h, m * sizeof(double), cudaMemcpyHostToDevice)); }

void d2host(const double* d, double* h, size_t m) { FS_CUDA(cudaMemcpy(h, d, m * sizeof(double), cudaMemcpyDeviceToHost)); }
void f2host(const float* d, float* h, size_t m) { FS_CUDA(cudaMemcpy(h, d, m * sizeof(float), cudaMemcpyDeviceToHost)); }
void host2f(const float* h, float* d, size_t m) { FS_CUDA(cudaMemcpy(d, h, m * sizeof(float), cudaMemcpyHostToDevice)); }

void filter_periodic(int n, float r, const double* in, double* out) {
	int R = int(std::ceil(r)) - 1;
	if (R < 0) R = 0;
	double wsum = 0;
	for (int dz = -R; dz <= R; dz++)
		for (int dy = -R; dy <= R; dy++)
			for (int dx = -R; dx <= R; dx++) {
				float d = std::sqrt(float(dx * dx + dy * dy + dz * dz));
				float w = r - d;
				if (w > 0.f) wsum += w;
			}
	size_t N = size_t(n) * n * n;
	k_filter<<<nblocks(N, 128), 128>>>(n, R, r, wsum, in, out);
	FS_KERNEL_CHECK();
}

void project_heaviside(size_t m, float beta, float eta, const double* rt, double* rho, double* drho) {
	k_project<<<nblocks(m, 256), 256>>>(m, beta, eta, rt, rho, drho);
	FS_KERNEL_CHECK();
}

}  // namespace fs
