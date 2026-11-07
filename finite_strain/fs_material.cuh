// Neo-Hookean solid with the energy interpolation of Wang et al. (2014), device functions.
//
// All functions take the displacement gradient D = F - I instead of F, so that stresses and
// energies are evaluated without cancellation in FP32 at small strain:
//   F - F^{-T} = D + F^{-T} D^T,   J - 1 = tr D + I2(D) + I3(D),
//   psi_NH = mu/2 D:D - mu (I2 + I3) + mu (j1 - log(1 + j1)) + lam/2 log^2(1 + j1).
// With D_g = gamma D (displacement gradient of F_g) and psi_L(F_g) = gamma^2 psi_L(F):
//   psi = psi_NH(D_g) + (1 - gamma^2) psi_L(D)
//   P   = gamma P_NH(D_g) + (1 - gamma^2) P_L(D)
//   A   = gamma^2 A_NH(D_g) + (1 - gamma^2) A_L
#pragma once

#include "fs_common.h"

namespace fs {

__host__ __device__ inline float fs_gamma(const Material& m, float v) {
	return (m.tb0 + tanhf(m.beta * (v - m.rho0))) / m.gden;
}

__host__ __device__ inline float fs_dgamma(const Material& m, float v) {
	float t = tanhf(m.beta * (v - m.rho0));
	return m.beta * (1.f - t * t) / m.gden;
}

// x - log(1 + x), with a series for small |x|.
__device__ inline float xm_log1p(float x) {
	if (fabsf(x) < 0.1f) {
		return x * x *
			(0.5f + x * (-1.f / 3.f + x * (0.25f + x * (-0.2f + x * (1.f / 6.f + x * (-1.f / 7.f + x * (0.125f + x * (-1.f / 9.f))))))));
	}
	return x - log1pf(x);
}

// Inverse of F = I + D, j1 = det F - 1 and i23 = I2(D) + I3(D).
__device__ inline void detinv(const float* D, float* Fi, float& j1, float& i23) {
	float f00 = 1.f + D[0], f01 = D[1], f02 = D[2];
	float f10 = D[3], f11 = 1.f + D[4], f12 = D[5];
	float f20 = D[6], f21 = D[7], f22 = 1.f + D[8];
	float i2 = D[0] * D[4] + D[4] * D[8] + D[0] * D[8] - D[1] * D[3] - D[5] * D[7] - D[2] * D[6];
	float i3 = D[0] * (D[4] * D[8] - D[5] * D[7]) - D[1] * (D[3] * D[8] - D[5] * D[6]) + D[2] * (D[3] * D[7] - D[4] * D[6]);
	i23 = i2 + i3;
	j1 = (D[0] + D[4] + D[8]) + i23;
	float iJ = 1.f / (1.f + j1);
	Fi[0] = (f11 * f22 - f12 * f21) * iJ;
	Fi[1] = (f02 * f21 - f01 * f22) * iJ;
	Fi[2] = (f01 * f12 - f02 * f11) * iJ;
	Fi[3] = (f12 * f20 - f10 * f22) * iJ;
	Fi[4] = (f00 * f22 - f02 * f20) * iJ;
	Fi[5] = (f02 * f10 - f00 * f12) * iJ;
	Fi[6] = (f10 * f21 - f11 * f20) * iJ;
	Fi[7] = (f01 * f20 - f00 * f21) * iJ;
	Fi[8] = (f00 * f11 - f01 * f10) * iJ;
}

// Data of one Gauss point, prepared once and used by several tangent products.
struct GPState {
	float D[9];   // displacement gradient F - I
	float Fi[9];  // F_gamma^{-1}
	float L;      // log J_gamma
	float j1;     // J_gamma - 1
	float i23;    // I2 + I3 of D_gamma
	float g, g2;  // gamma, gamma^2
};

// Returns false if J_gamma <= 0, where psi_NH is not defined.
__device__ inline bool gp_prepare(const float* D, float gam, GPState& s) {
	float Dg[9];
#pragma unroll
	for (int c = 0; c < 9; c++) {
		s.D[c] = D[c];
		Dg[c] = gam * D[c];
	}
	s.g = gam;
	s.g2 = gam * gam;
	detinv(Dg, s.Fi, s.j1, s.i23);
	if (!(1.f + s.j1 > 0.f)) return false;
	s.L = log1pf(s.j1);
	return true;
}

// Y = A_NH(F_g)[H] = mu H + (mu - lam L) F^{-T} H^T F^{-T} + lam tr(F^{-1} H) F^{-T}
__device__ inline void nh_tangent(const Material& m, const GPState& s, const float* H, float* Y) {
	float T[9], U[9];
	// T = Fi H
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int j = 0; j < 3; j++)
			T[i * 3 + j] = s.Fi[i * 3 + 0] * H[0 * 3 + j] + s.Fi[i * 3 + 1] * H[1 * 3 + j] + s.Fi[i * 3 + 2] * H[2 * 3 + j];
	// U = T Fi = Fi H Fi; F^{-T} H^T F^{-T} = U^T
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int j = 0; j < 3; j++)
			U[i * 3 + j] = T[i * 3 + 0] * s.Fi[0 * 3 + j] + T[i * 3 + 1] * s.Fi[1 * 3 + j] + T[i * 3 + 2] * s.Fi[2 * 3 + j];
	float trT = T[0] + T[4] + T[8];
	float c1 = m.mu - m.lam * s.L;
	float c2 = m.lam * trT;
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int j = 0; j < 3; j++) Y[i * 3 + j] = m.mu * H[i * 3 + j] + c1 * U[j * 3 + i] + c2 * s.Fi[j * 3 + i];
}

// Y = A_L[H] = lam tr(H) I + mu (H + H^T)
__device__ inline void lin_tangent(const Material& m, const float* H, float* Y) {
	float tr = H[0] + H[4] + H[8];
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int j = 0; j < 3; j++) Y[i * 3 + j] = m.mu * (H[i * 3 + j] + H[j * 3 + i]) + (i == j ? m.lam * tr : 0.f);
}

// Y = A(F; rho)[H] = gamma^2 A_NH(F_g)[H] + (1 - gamma^2) A_L[H]
__device__ inline void tangent(const Material& m, const GPState& s, const float* H, float* Y) {
	float Yn[9], Yl[9];
	nh_tangent(m, s, H, Yn);
	lin_tangent(m, H, Yl);
#pragma unroll
	for (int c = 0; c < 9; c++) Y[c] = s.g2 * Yn[c] + (1.f - s.g2) * Yl[c];
}

// P_NH(F_g) = mu (D_g + F_g^{-T} D_g^T) + lam L F_g^{-T}
__device__ inline void nh_stress(const Material& m, const GPState& s, float* P) {
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int j = 0; j < 3; j++) {
			// (F^{-T} D_g^T)_ij = sum_k Fi_ki Dg_jk
			float t = s.Fi[0 * 3 + i] * s.D[j * 3 + 0] + s.Fi[1 * 3 + i] * s.D[j * 3 + 1] + s.Fi[2 * 3 + i] * s.D[j * 3 + 2];
			P[i * 3 + j] = m.mu * (s.g * s.D[i * 3 + j] + s.g * t) + m.lam * s.L * s.Fi[j * 3 + i];
		}
}

// P_L(D) = lam tr(D) I + 2 mu sym(D)
__device__ inline void lin_stress(const Material& m, const float* D, float* P) {
	float tr = D[0] + D[4] + D[8];
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int j = 0; j < 3; j++) P[i * 3 + j] = m.mu * (D[i * 3 + j] + D[j * 3 + i]) + (i == j ? m.lam * tr : 0.f);
}

// P(F; rho) = gamma P_NH(F_g) + (1 - gamma^2) P_L(D)
__device__ inline void stress(const Material& m, const GPState& s, float* P) {
	float Pn[9], Pl[9];
	nh_stress(m, s, Pn);
	lin_stress(m, s.D, Pl);
#pragma unroll
	for (int c = 0; c < 9; c++) P[c] = s.g * Pn[c] + (1.f - s.g2) * Pl[c];
}

// psi(F; rho) = psi_NH(F_g) + (1 - gamma^2) psi_L(D)
__device__ inline float energy(const Material& m, const GPState& s) {
	float dd = 0.f;
#pragma unroll
	for (int c = 0; c < 9; c++) dd += s.D[c] * s.D[c];
	float psin = 0.5f * m.mu * s.g2 * dd - m.mu * s.i23 + m.mu * xm_log1p(s.j1) + 0.5f * m.lam * s.L * s.L;
	float tr = s.D[0] + s.D[4] + s.D[8];
	float ee = 0.f;
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int j = 0; j < 3; j++) {
			float e = 0.5f * (s.D[i * 3 + j] + s.D[j * 3 + i]);
			ee += e * e;
		}
	float psil = 0.5f * m.lam * tr * tr + m.mu * ee;
	return psin + (1.f - s.g2) * psil;
}

// dP/dgamma = P_NH(F_g) + gamma A_NH(F_g)[D] - 2 gamma P_L(D)
__device__ inline void stress_dgamma(const Material& m, const GPState& s, float* dP) {
	float Pn[9], An[9], Pl[9];
	nh_stress(m, s, Pn);
	nh_tangent(m, s, s.D, An);
	lin_stress(m, s.D, Pl);
#pragma unroll
	for (int c = 0; c < 9; c++) dP[c] = Pn[c] + s.g * An[c] - 2.f * s.g * Pl[c];
}

// Third derivative of psi_NH at F = I:
//   G[H1,H2,H3] = -mu (tr(H1 H2 H3) + tr(H2 H1 H3)) - lam (tr(H1 H2) tr H3 + tr H1 tr(H2 H3) + tr H2 tr(H1 H3))
__device__ inline float tr2(const float* A, const float* B) {
	float t = 0.f;
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int k = 0; k < 3; k++) t += A[i * 3 + k] * B[k * 3 + i];
	return t;
}
__device__ inline float tr3m(const float* A, const float* B, const float* C) {
	float t = 0.f;
#pragma unroll
	for (int i = 0; i < 3; i++)
#pragma unroll
		for (int k = 0; k < 3; k++) {
			float ab = A[i * 3 + 0] * B[0 * 3 + k] + A[i * 3 + 1] * B[1 * 3 + k] + A[i * 3 + 2] * B[2 * 3 + k];
			t += ab * C[k * 3 + i];
		}
	return t;
}
__device__ inline float nh_third(const Material& m, const float* H1, const float* H2, const float* H3) {
	float t1 = H1[0] + H1[4] + H1[8], t2 = H2[0] + H2[4] + H2[8], t3 = H3[0] + H3[4] + H3[8];
	return -m.mu * (tr3m(H1, H2, H3) + tr3m(H2, H1, H3)) -
		m.lam * (tr2(H1, H2) * t3 + t1 * tr2(H2, H3) + t2 * tr2(H1, H3));
}

}  // namespace fs
