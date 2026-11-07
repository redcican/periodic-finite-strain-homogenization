// fs3d: finite-strain periodic homogenization of voxel cells on the GPU.
#include "fs_io.h"
#include "fs_stability.h"

#include <gflags/gflags.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

DEFINE_string(mode, "path", "path: one load path with stability criteria");
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

namespace {

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

}  // namespace

int main(int argc, char** argv) {
	gflags::SetUsageMessage("finite-strain periodic homogenization of voxel cells");
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	try {
		if (FLAGS_mode == "path") return run_path_mode();
		std::fprintf(stderr, "unknown mode %s\n", FLAGS_mode.c_str());
		return 1;
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
}
