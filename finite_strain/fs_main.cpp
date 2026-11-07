// fs3d: finite-strain periodic homogenization of voxel cells on the GPU.
#include "fs_io.h"
#include "fs_stability.h"

#include <gflags/gflags.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <fstream>
#include <stdexcept>
#include <string>

DEFINE_string(mode, "path", "path: one load path with stability criteria; toec: small-strain moduli and third-order constants; fdcheck: adjoint gradient against central differences");
DEFINE_string(in, "", "density field (OpenVDB file of the GPU solver, or .npy of shape [n, n, n])");
DEFINE_int32(reso, 0, "resolution n (required for OpenVDB input; n = 16, 32, 64, 128, 256)");
DEFINE_string(prefix, "./", "output directory, suffixed with /");
DEFINE_double(E, 10.0, "Young's modulus of the solid");
DEFINE_double(nu, 0.3, "Poisson's ratio of the solid");
DEFINE_string(path, "L1", "load path: L1 uniaxial tension, L2 uniaxial compression, L3 simple shear, L4 volumetric expansion");
DEFINE_int32(steps, 10, "number of load steps");
DEFINE_double(inc, 0.0, "increment per load step (default 0.02 for L1-L3, 0.01 for L4)");
DEFINE_string(prec, "adaptive", "coarse stencil precision: fp16, fp32 or adaptive");
DEFINE_string(crit, "1", "stability criteria to evaluate: any of 1 (cell), 2 (2x2x2 supercell), 3 (rank-one); empty for none");
DEFINE_int32(crit_every, 1, "evaluate criterion 1 every k-th load point");
DEFINE_double(newton_tol, 1e-6, "relative Newton tolerance");
DEFINE_int32(krylov_max, 4000, "maximal Krylov iterations per linear solve");
DEFINE_bool(stop_distortion, true, "stop the path once more than 1% of the void voxels reach J < 0.1");
DEFINE_int32(verbose, 1, "0 quiet, 1 load steps, 2 Newton iterations");
DEFINE_double(lin_tol, 1e-6, "relative tolerance of the corrector and adjoint solves");
DEFINE_string(obj, "nu12", "fdcheck objective at load point -fd_k: nu12 (secant Poisson's ratio, L1/L2) or P11");
DEFINE_int32(fd_k, 0, "fdcheck load point (default: the last one)");
DEFINE_int32(fd_n, 6, "fdcheck: number of voxels to perturb");
DEFINE_double(fd_h, 1e-3, "fdcheck: density perturbation");

namespace {

using fs::Prec;

fs::SolverOptions solver_options() {
	fs::SolverOptions o;
	o.E = FLAGS_E;
	o.nu = FLAGS_nu;
	o.newtonTol = FLAGS_newton_tol;
	o.krylovMax = FLAGS_krylov_max;
	o.verbose = FLAGS_verbose;
	return o;
}

std::vector<float> load_density(int& n) {
	if (FLAGS_in.empty()) throw std::runtime_error("no density given (-in)");
	n = FLAGS_reso;
	return fs::read_density(FLAGS_in, n);
}

void write_path_csv(const std::string& file, const fs::PathResult& r) {
	std::ofstream o(file);
	o << "k,t,eps,F11,F22,F33,F12,P11,P12,P13,P21,P22,P23,P31,P32,P33,nu12,nu13,newton,krylov,substeps,failures,minres,"
		 "prec,lambda_cell,lambda_supercell,ellipticity,void_distorted\n";
	o.precision(10);
	for (const auto& p : r.pts) {
		o << p.k << "," << p.t << "," << p.eps << "," << p.F[0] << "," << p.F[4] << "," << p.F[8] << "," << p.F[1];
		for (int c = 0; c < 9; c++) o << "," << p.P[c];
		o << "," << p.nu12 << "," << p.nu13 << "," << p.newton << "," << p.krylov << "," << p.substeps << "," << p.failures << ","
		  << (p.minres ? 1 : 0) << "," << fs::prec_name(p.prec) << "," << p.lam1 << "," << p.lam2 << "," << p.ell << ","
		  << p.voidDistortedFrac << "\n";
	}
}

int run_path_mode() {
	int n = 0;
	std::vector<float> rho = load_density(n);
	fs::Solver S(n, rho, solver_options());
	fs::LoadPath path = fs::make_path(FLAGS_path, FLAGS_steps, FLAGS_inc);
	fs::PathOptions po;
	po.precMode = FLAGS_prec;
	po.crit1Every = FLAGS_crit.find('1') != std::string::npos ? FLAGS_crit_every : 0;
	po.crit2 = FLAGS_crit.find('2') != std::string::npos;
	po.crit3 = FLAGS_crit.find('3') != std::string::npos;
	po.stopOnDistortion = FLAGS_stop_distortion;
	fs::Stability stab(S);
	std::printf("path %s on %d^3 voxels, %d steps, %s stencils\n", path.name.c_str(), n, path.nsteps, FLAGS_prec.c_str());
	fs::PathResult r = fs::run_path(S, path, po, &stab);
	std::string file = FLAGS_prefix + "path_" + path.name + ".csv";
	write_path_csv(file, r);
	std::printf("%s: %zu load points, %d Newton and %d Krylov iterations -> %s\n",
		r.completed ? "completed" : (r.stopped ? "stopped" : "failed"), r.pts.size(), r.newton, r.krylov, file.c_str());
	return r.failed ? 2 : 0;
}

const char* voigt_label(int I) {
	static const char* l[6] = {"1", "2", "3", "4", "5", "6"};
	return l[I];
}

int run_toec_mode() {
	int n = 0;
	std::vector<float> rho = load_density(n);
	fs::Solver S(n, rho, solver_options());
	double C[36], T[56];
	Prec p = FLAGS_prec == "fp32" ? Prec::FP32 : Prec::FP16;
	bool ok = S.linearModuli(C, T, true, FLAGS_lin_tol, p);
	std::string file = FLAGS_prefix + "moduli.txt";
	std::ofstream o(file);
	o.precision(10);
	o << "# small-strain moduli C^H (Voigt 11 22 33 23 13 12, engineering shears)\n";
	for (int I = 0; I < 6; I++) {
		for (int J = 0; J < 6; J++) o << (J ? " " : "") << C[I * 6 + J];
		o << "\n";
	}
	o << "# third-order elastic constants T_IJK, I <= J <= K\n";
	int t = 0;
	for (int I = 0; I < 6; I++)
		for (int J = I; J < 6; J++)
			for (int K = J; K < 6; K++) o << "C" << voigt_label(I) << voigt_label(J) << voigt_label(K) << " " << T[t++] << "\n";
	std::printf("C11 = %.6e  C12 = %.6e  C44 = %.6e  C111 = %.6e  (%s) -> %s\n", C[0], C[1], C[21], T[0],
		ok ? "converged" : "corrector solves not converged", file.c_str());
	return ok ? 0 : 2;
}

// Objective at load point K of the path and its adjoint sensitivity (accumulated into dJ if non-null).
double fd_objective(fs::Solver& S, const fs::LoadPath& path, int K, double* dJ, bool* ok) {
	fs::PathOptions po;
	po.precMode = FLAGS_prec;
	po.stopOnDistortion = false;
	double J = std::numeric_limits<double>::quiet_NaN();
	Prec p = FLAGS_prec == "fp32" ? Prec::FP32 : Prec::FP16;
	po.onLoadPoint = [&](int k, fs::Solver& s, fs::PointRecord& rec) {
		if (k != K) return;
		double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, b[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		if (FLAGS_obj == "P11") {
			J = rec.P[0];
			a[0] = 1;
		} else {
			J = rec.nu12;
			b[0] = -1.0 / rec.eps;  // free component 0 is Fbar_22
		}
		if (dJ && !s.adjoint(a, b, dJ, FLAGS_lin_tol, p)) *ok = false;
	};
	fs::PathResult r = fs::run_path(S, path, po, nullptr);
	if (!r.completed) *ok = false;
	return J;
}

int run_fdcheck_mode() {
	int n = 0;
	std::vector<float> rho = load_density(n);
	fs::SolverOptions o = solver_options();
	o.verbose = 0;
	fs::Solver S(n, rho, o);
	fs::LoadPath path = fs::make_path(FLAGS_path, FLAGS_steps, FLAGS_inc);
	if (FLAGS_obj == "nu12" && path.freeComps.empty()) throw std::runtime_error("nu12 requires path L1 or L2");
	int K = FLAGS_fd_k > 0 ? FLAGS_fd_k : path.nsteps;
	double* dJ = fs::dalloc_d(S.cell.ne);
	bool ok = true;
	double J0 = fd_objective(S, path, K, dJ, &ok);
	std::vector<double> g(S.cell.ne);
	fs::d2host(dJ, g.data(), g.size());
	fs::dfree(dJ);
	std::printf("J = %s(load point %d) = %.10e%s\n", FLAGS_obj.c_str(), K, J0, ok ? "" : "  (path or adjoint not converged)");
	// voxels with intermediate density, spread over the cell
	std::vector<size_t> pick;
	for (size_t e = 0; e < rho.size() && int(pick.size()) < FLAGS_fd_n; e += 1 + rho.size() / (64 * size_t(FLAGS_fd_n)))
		if (rho[e] > 0.2f && rho[e] < 0.8f) pick.push_back(e);
	for (size_t e = 0; e < rho.size() && int(pick.size()) < FLAGS_fd_n; e += 1 + rho.size() / size_t(FLAGS_fd_n))
		if (rho[e] >= 0.8f) pick.push_back(e);
	double worst = 0;
	for (size_t e : pick) {
		std::vector<float> rp = rho, rm = rho;
		double h = FLAGS_fd_h;
		rp[e] = float(std::min(1.0, rho[e] + h));
		rm[e] = float(std::max(0.0, rho[e] - h));
		double hh = double(rp[e]) - double(rm[e]);
		bool okp = true, okm = true;
		S.setDensity(rp);
		double Jp = fd_objective(S, path, K, nullptr, &okp);
		S.setDensity(rm);
		double Jm = fd_objective(S, path, K, nullptr, &okm);
		double fd = (Jp - Jm) / hh;
		double rel = std::fabs(fd - g[e]) / std::max(std::fabs(fd), 1e-30);
		worst = std::max(worst, rel);
		std::printf("voxel %8zu  rho = %.3f  adjoint = %+.6e  central difference = %+.6e  rel. error = %.2e%s\n", e, rho[e], g[e], fd, rel,
			okp && okm ? "" : "  (not converged)");
	}
	S.setDensity(rho);
	std::printf("largest relative error %.2e over %zu voxels\n", worst, pick.size());
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	gflags::SetUsageMessage("finite-strain periodic homogenization of voxel cells");
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	try {
		if (FLAGS_mode == "path") return run_path_mode();
		if (FLAGS_mode == "toec") return run_toec_mode();
		if (FLAGS_mode == "fdcheck") return run_fdcheck_mode();
		std::fprintf(stderr, "unknown mode %s\n", FLAGS_mode.c_str());
		return 1;
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
}
