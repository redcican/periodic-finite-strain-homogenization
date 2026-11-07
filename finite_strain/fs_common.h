// Finite-strain periodic homogenization: common definitions shared by host and device code.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define FS_CUDA(call)                                                                              \
	do {                                                                                           \
		cudaError_t fs_e_ = (call);                                                                \
		if (fs_e_ != cudaSuccess) {                                                                \
			std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n", cudaGetErrorString(fs_e_), __FILE__, \
				__LINE__);                                                                         \
			std::exit(1);                                                                          \
		}                                                                                          \
	} while (0)

#define FS_KERNEL_CHECK() FS_CUDA(cudaGetLastError())

namespace fs {

// Material and interpolation constants, all in FP32 for the device kernels.
//   psi(F; rho) = psi_NH(F_g) - psi_L(F_g) + psi_L(F),  F_g = I + gamma(rho^p) (F - I)
//   s(rho) = rhomin + rho^p
//   gamma(v) = (tanh(beta rho0) + tanh(beta (v - rho0))) / (tanh(beta rho0) + tanh(beta (1 - rho0)))
struct Material {
	float lam, mu;     // Lame constants of the solid
	float beta, rho0;  // smoothed step of the energy interpolation
	float tb0, gden;   // tanh(beta rho0), tanh(beta rho0) + tanh(beta (1 - rho0))
	float p, rhomin;   // stiffness scaling s = rhomin + rho^p
};

inline Material make_material(double E, double nu, double beta, double rho0, double p, double rhomin) {
	Material m;
	m.lam = float(E * nu / ((1 + nu) * (1 - 2 * nu)));
	m.mu = float(E / (2 * (1 + nu)));
	m.beta = float(beta);
	m.rho0 = float(rho0);
	m.tb0 = float(std::tanh(beta * rho0));
	m.gden = float(std::tanh(beta * rho0) + std::tanh(beta * (1 - rho0)));
	m.p = float(p);
	m.rhomin = float(rhomin);
	return m;
}

// Storage precision of the coarse stencils (levels between the fine grid and the 8^3 grid).
enum class Prec { FP16 = 0, FP32 = 1 };

inline const char* prec_name(Prec p) { return p == Prec::FP16 ? "fp16" : "fp32"; }

inline int nblocks(size_t n, int bs) { return int((n + bs - 1) / bs); }

// Tensor component index (row-major 3x3): comp(i, j) = 3 i + j.
inline constexpr int comp(int i, int j) { return 3 * i + j; }

}  // namespace fs
