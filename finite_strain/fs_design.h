// Finite-strain design of cells with cubic symmetry by MMA:
//   Poisson's ratio program:  min z  s.t. (nu12(eps_k) - nu*(eps_k))^2 <= z, k = 1..N
//   stress-strain program:    min J_S = sum_k (P11(eps_k) - P*(eps_k))^2 / sum_k P*(eps_k)^2
// both subject to V <= V*, E^H >= E* and 0 <= x <= 1, along path L1 with the N load steps as load points.
#pragma once

#include "fs_solver.h"

#include <string>
#include <vector>

namespace fs {

struct DesignOptions {
	std::string program = "nu";  // nu or stiff
	double nuStar = 0.0;         // nu*(eps) = nuStar + nuSlope eps
	double nuSlope = 0.0;
	double S = 2.0;              // stiffening ratio E_t(eps_max) / E_t(0)
	double Vstar = 0, Estar = 0;
	int iters = 240;
	double move = 0.1;
	double filterR = 0;          // filter radius in voxels (default n / 32)
	int betaEvery = 40;
	double betaMax = 32;
	int steps = 10;
	double inc = 0.02;
	std::string prec = "adaptive";
	int crit1Every = 2;
	double linTol = 1e-6;
	int saveEvery = 20;
	std::string prefix = "./";
	bool finalCheck = true;      // thresholded design along L1 with increments 0.005 and all criteria
};

// x0: initial physical density on the full cell (symmetrized by the design map).
int run_design(Solver& S, const std::vector<float>& x0, const DesignOptions& o);

// Volume fraction and E^H = 1 / [C^-1]_11 of a density field (state of S is reset).
bool reference_values(Solver& S, const std::vector<float>& rho, Prec p, double tol, double& V, double& E);

}  // namespace fs
