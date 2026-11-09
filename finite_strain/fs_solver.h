// Matrix-free mixed-precision Newton-Krylov multigrid solver of the finite-strain cell problem.
//
// Unknowns u = (w, Fbar_F): the periodic fluctuation and the free components of the macroscopic
// deformation gradient (zero conjugate stress). Gradient and Hessian of Pi_h with respect to u:
//   g = [R; Pbar_F],  H = [K_T B; B^T D].
#pragma once

#include "fs_krylov.h"
#include "fs_mg.h"

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace fs {

constexpr double kU32 = 5.9604644775390625e-08;  // 2^-24, unit roundoff of FP32

struct SolverOptions {
	double E = 10, nu = 0.3;              // base material
	double beta = 500, rho0 = 0.01;       // energy interpolation
	double p = 3, rhomin = 1e-9;          // stiffness scaling
	double newtonTol = 1e-6;              // ||g|| <= max(newtonTol ||g_0||, newtonFloor u32 ||f_e||), ||f_e|| the norm of the
	double newtonFloor = 64;              // element forces the FP32 residual is assembled from (its rounding level)
	int newtonMax = 12;
	int lsMax = 8;                        // line-search halvings
	double armijo = 1e-4;
	double etaMax = 0.1, ewGamma = 0.9, ewAlpha = 2;  // Eisenstat-Walker, second choice
	int krylovMax = 4000;
	int maxHalvings = 3;                  // substep divisions of a failed load step
	double adaptThresh = 0.1;             // FP32 switch when extrapolated lambda_min < adaptThresh lambda_min(0)
	double sigmaFloor = 1e-3;             // smallest MINRES preconditioner shift (scaled units)
	int verbose = 1;
};

enum class StepFail { None = 0, NewtonMaxIter, LineSearch, InvalidJ, PrecNotPD };
const char* step_fail_name(StepFail f);

struct StepStats {
	int newton = 0, krylov = 0;
	bool minres = false;
	StepFail fail = StepFail::None;
	double g0 = 0, gend = 0;
};

// State of the LOBPCG eigenvalue iteration, kept between calls as starting block.
struct EigState {
	std::vector<BVec> X;
	int m = 6;
	bool init = false;
	unsigned seed = 1;
};

class Solver : public LinOp {
public:
	Solver(int n, const std::vector<float>& rho, const SolverOptions& opt);
	~Solver() override;
	Solver(const Solver&) = delete;
	Solver& operator=(const Solver&) = delete;

	// ---- problem definition
	void setFree(const std::vector<int>& comps);  // free components of Fbar (comp(i, i) for diagonal ones)
	void setDensity(const std::vector<float>& rho);
	void resetState();                             // w = 0, Fbar = I
	void setPrescribed(const double F[9]);         // prescribed components of Fbar (free ones are ignored)
	void fullF(const BVec& state, double F[9]) const;

	// ---- evaluation (fills the deformation-gradient cache of `state`)
	bool evaluate(const BVec& state, BVec& grad, double P[9], EnergyOut& e);
	bool evaluateCurrent();

	// ---- operator and preconditioner of the bordered system
	void apply(const BVec& x, BVec& y) override;  // y = H x
	bool prec(const BVec& r, BVec& z) override;   // z = M^{-1} r
	// Restricts the tangent of the current cache (shifted by sigma Z0 if sigma > 0) and builds the
	// border block. Returns false if the preconditioner is not positive definite.
	bool buildPreconditioner(Prec p, float sigma);
	void initScaling();  // Z0 = diag H at Fbar = I (requires setFree)

	// ---- Newton iteration of one load step from the current state with the current Fbar
	StepStats newtonSolve(Prec p);

	// ---- stability criteria (fs_stability.cpp)
	// Smallest eigenvalue of Z0^{-1/2} H Z0^{-1/2} orthogonal to the scaled translations (LOBPCG).
	double lambdaMin(EigState& st, double tol, int maxit, bool usePrec, int* iters = nullptr, bool* conv = nullptr);
	// Effective tangent Abar = D9 - B9^T K^{-1} B9 (nine K_T solves, warm-started); requires buildPreconditioner.
	bool effectiveTangent(double Abar[81], double tol, int* iters = nullptr);
	// Branch switching at a loss of stability: perturbs the state along the critical eigenvector v of the last
	// lambdaMin call (max|v_w| = 1), u <- u + alpha v with alpha = frac max(max|w|, 1/n) doubled while the energy
	// along +-v decreases (the lowest energy fixes the sign and alpha), and re-converges the Newton iteration
	// at the same load (three times the Newton iterations of a load step). moved: the re-converged state has a
	// lower energy than the start beyond the rounding level (the saddle was left); otherwise the start is restored.
	StepStats switchBranch(Prec p, double frac, double& alpha, bool& moved);

	// ---- small-strain moduli at Fbar = I (fs_sens.cpp); the state is reset
	bool linearModuli(double C[36], double T56[56], bool toec, double tol, Prec p);
	void moduliSensitivity(const double W[36], double* dJ);  // after linearModuli
	// Adjoint solve at the converged state for J(Pbar, Fbar_F) with dJ/dPbar = a and dJ/dFbar_F = b,
	// accumulates the sensitivity dJ/drho into dJ (device, n^3).
	bool adjoint(const double a[9], const double* b, double* dJ, double tol, Prec p, int* iters = nullptr);

	// ---- data
	int n;
	SolverOptions opt;
	Cell cell;
	Multigrid mg;
	Space sp;                      // bordered space (3 n^3, nF)
	Space spK;                     // fluctuation-only space (3 n^3, 0)
	std::vector<int> freeComps;
	double Fbar[9];                // prescribed components (free ones live in u.f)
	BVec u, g;                     // state and its gradient
	double Pbar[9];
	EnergyOut en;
	bool valid = false;
	float* Z0w = nullptr;          // diag K_T at Fbar = I
	float* sqZ0w = nullptr;
	float* isqZ0w = nullptr;
	double Z0F[9];                 // diag D at Fbar = I on the free components
	bool scalingReady = false;
	EigState eig;                  // criterion 1, cell
	std::vector<float*> chi;       // warm starts: 6 correctors, then 9 columns of the effective tangent
	std::vector<float> rhoHost;

private:
	bool lineSearch(const BVec& du, double eta);
	bool prepareShifted(Prec p, float& sigma, bool first);
	// Regularized Newton step of a branch switch: PCG on H + sigma Z0 with sigma = 2 max(0, -lambda_ritz), a descent
	// direction that doubles the amplitude along an unstable mode per iteration and is the Newton step once the
	// tangent is positive definite.
	bool regularizedSolve(Prec p, float& sigma, double eta, KResult& kr, int& krylov);
	bool descent = false;          // newtonSolve takes regularized steps instead of MINRES (switchBranch)

	BVec trial, gt, du, rhs, tmp1, tmp2;
	double Ds[81];                 // Cholesky factor of the border block D + sigma Z0F
	bool precReady = false;
	float sigmaNow = 0.f;

public:
	bool preconditionerReady() const { return precReady && sigmaNow == 0.f; }
};

// Fluctuation block K_T alone (all components of Fbar prescribed), with the V-cycle as preconditioner.
struct KOp : LinOp {
	Solver& S;
	explicit KOp(Solver& s) : S(s) {}
	void apply(const BVec& x, BVec& y) override { cell_product(S.cell, x.w, nullptr, y.w, nullptr); }
	bool prec(const BVec& r, BVec& z) override {
		S.mg.vcycle(r.w, z.w);
		return true;
	}
};

// ---------------------------------------------------------------------------------------------
// Load paths and Algorithm 1
// ---------------------------------------------------------------------------------------------
struct LoadPath {
	std::string name;           // L1 .. L4
	std::vector<int> freeComps;
	int nsteps = 10;
	double inc = 0.02;
	void F(double t, double Fb[9]) const;  // prescribed Fbar at load parameter t
	double strain(const double Fb[9]) const;  // epsilon (L1, L2: Fbar_11 - 1), gamma (L3), epsilon (L4)
};
LoadPath make_path(const std::string& name, int nsteps, double inc);

struct PointRecord {
	int k = 0;
	double t = 0, eps = 0;
	double F[9], P[9];
	double nu12 = std::numeric_limits<double>::quiet_NaN(), nu13 = std::numeric_limits<double>::quiet_NaN();
	int newton = 0, krylov = 0, substeps = 0, failures = 0;
	bool minres = false;
	Prec prec = Prec::FP16;
	double lam1 = std::numeric_limits<double>::quiet_NaN();   // criterion 1, cell
	double lam2 = std::numeric_limits<double>::quiet_NaN();   // criterion 2, 2x2x2 supercell
	double ell = std::numeric_limits<double>::quiet_NaN();    // criterion 3, min_n lambda_min(Q(n))
	double voidDistortedFrac = 0;
	double seconds = 0;  // wall time of the load step (Newton iterations, criteria and hook)
	int branch = 0;      // branch switches so far (-branch)
	double energy = std::numeric_limits<double>::quiet_NaN();  // Pi_h at the load point
};

struct PathOptions {
	std::string precMode = "adaptive";  // fp16, fp32 or adaptive
	int crit1Every = 0;                 // criterion 1 every k-th load point (0: never)
	bool crit2 = false, crit3 = false;  // criteria 2 and 3 at every load point
	double eigTol = 1e-3;
	int eigMaxit = 200;
	bool stopOnDistortion = true;
	bool branch = false;                // switch to the stable branch at lambda_cell <= 0 (requires crit1Every > 0)
	double branchFrac = 1e-2;           // first kick, fraction of max(max|w|, voxel size); x10 per retry
	int branchTries = 3;                // failed kicks in a row per load point (at most 2 branchTries switches)
	std::function<void(int k, Solver& s, PointRecord& rec)> onLoadPoint;  // adjoints etc.
};

struct PathResult {
	bool completed = false, stopped = false, failed = false;
	double tFail = 0;
	double lam0 = std::numeric_limits<double>::quiet_NaN();
	std::vector<PointRecord> pts;
	int newton = 0, krylov = 0;
	bool branch = false;  // the CSV carries the branch and energy columns
};

class Stability;  // supercell and rank-one criteria (fs_stability.h)

PathResult run_path(Solver& s, const LoadPath& path, PathOptions& po, Stability* stab);
void write_path_csv(const std::string& file, const PathResult& r);

}  // namespace fs
