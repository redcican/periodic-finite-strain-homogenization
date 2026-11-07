// Stability criteria along a load path:
//   1. lambda_min of the scaled bordered tangent of the cell (Solver::lambdaMin, LOBPCG)
//   2. the same on a 2x2x2 supercell carrying the repeated state of the cell
//   3. strong ellipticity of the effective tangent, min over directions n of lambda_min(Q(n))
#pragma once

#include "fs_solver.h"

#include <memory>

namespace fs {

class Stability {
public:
	explicit Stability(Solver& cell) : S(cell) {}
	// Criterion 2 at the current converged state of the cell.
	double supercell(double tol, int maxit, Prec p);
	// Criterion 3 at the current converged state; requires the preconditioner of the cell at that state.
	// Abar (optional) receives the effective tangent, dPbar_ij/dFbar_kl at [3i+j][3k+l].
	double rankOne(double* Abar = nullptr, double tol = 1e-4);

private:
	Solver& S;
	std::unique_ptr<Solver> sc;
};

// min over unit n of the smallest eigenvalue of Q_ik(n) = Abar_ijkl n_j n_l: 1000 directions on a
// hemisphere, then a local minimization from the three smallest.
double ellipticity_min(const double Abar[81], double nmin[3] = nullptr);

}  // namespace fs
