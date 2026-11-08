// One load path with Newton-Krylov iterations, step control, stability criteria and the hook for
// adjoint accumulation at the load points (Algorithm 1 of the paper).
#include "fs_stability.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

namespace fs {

PathResult run_path(Solver& s, const LoadPath& path, PathOptions& po, Stability* stab) {
	PathResult res;
	s.setFree(path.freeComps);
	s.resetState();
	s.initScaling();
	const bool canSwitch = po.precMode != "fp16";
	const bool adaptive = po.precMode == "adaptive";
	Prec prec = po.precMode == "fp32" ? Prec::FP32 : Prec::FP16;

	auto buildAtState = [&]() {
		if (!s.buildPreconditioner(prec, 0.f)) s.buildPreconditioner(Prec::FP32, 0.f);
	};

	std::vector<double> lamHist;
	if (po.crit1Every > 0) {
		s.evaluateCurrent();
		buildAtState();
		res.lam0 = s.lambdaMin(s.eig, po.eigTol, po.eigMaxit, true);
		lamHist.push_back(res.lam0);
		if (s.opt.verbose) std::printf("  lambda_min(0) = %.6e\n", res.lam0);
	}

	BVec backup = s.sp.alloc();
	double t = 0;
	for (int k = 1; k <= path.nsteps; k++) {
		const double tk = k * path.inc;
		double dt = tk - t;
		int d = 0;
		PointRecord rec;
		rec.k = k;
		auto t0 = std::chrono::steady_clock::now();
		while (t < tk - 1e-12 * path.inc) {
			s.sp.copy(s.u, backup);
			double Fp[9];
			path.F(t + dt, Fp);
			s.setPrescribed(Fp);
			StepStats st = s.newtonSolve(prec);
			rec.newton += st.newton;
			rec.krylov += st.krylov;
			rec.minres = rec.minres || st.minres;
			res.newton += st.newton;
			res.krylov += st.krylov;
			if (st.fail != StepFail::None) {
				rec.failures++;
				s.sp.copy(backup, s.u);
				double Fo[9];
				path.F(t, Fo);
				s.setPrescribed(Fo);
				if (s.opt.verbose)
					std::printf("  step to t = %.5f failed (%s, %s stencils, |g|/|g0| = %.2e)\n", t + dt, step_fail_name(st.fail),
						prec_name(prec), st.g0 > 0 ? st.gend / st.g0 : 0.0);
				if (prec == Prec::FP16 && canSwitch) {
					prec = Prec::FP32;  // repeat the step with FP32 stencils, for the rest of the path
				} else if (d < s.opt.maxHalvings) {
					dt *= 0.5;
					d++;
				} else {
					res.failed = true;
					res.tFail = t;
					s.evaluateCurrent();
					s.sp.release(backup);
					return res;
				}
			} else {
				t += dt;
				rec.substeps++;
				dt = std::min(dt, tk - t);
			}
		}
		t = tk;
		rec.t = tk;
		s.fullF(s.u, rec.F);
		for (int c = 0; c < 9; c++) rec.P[c] = s.Pbar[c];
		rec.eps = path.strain(rec.F);
		if ((path.name == "L1" || path.name == "L2") && rec.eps != 0) {
			rec.nu12 = -(rec.F[4] - 1) / rec.eps;
			rec.nu13 = -(rec.F[8] - 1) / rec.eps;
		}
		rec.prec = prec;
		rec.voidDistortedFrac = s.cell.nvoid > 0 ? double(s.en.voidDistorted) / double(s.cell.nvoid) : 0.0;

		const bool c1 = po.crit1Every > 0 && k % po.crit1Every == 0;
		if (c1 || po.crit2 || po.crit3 || po.onLoadPoint) buildAtState();
		if (c1) {
			rec.lam1 = s.lambdaMin(s.eig, po.eigTol, po.eigMaxit, true);
			lamHist.push_back(rec.lam1);
			size_t h = lamHist.size();
			if (adaptive && prec == Prec::FP16 && h >= 2 && std::isfinite(res.lam0)) {
				double ext = 2 * lamHist[h - 1] - lamHist[h - 2];
				if (ext < s.opt.adaptThresh * res.lam0) {
					prec = Prec::FP32;
					if (s.opt.verbose) std::printf("  extrapolated lambda_min %.3e < %.2f lambda_min(0): FP32 stencils from here\n", ext, s.opt.adaptThresh);
					buildAtState();
				}
			}
		}
		if (po.crit2 && stab) rec.lam2 = stab->supercell(po.eigTol, po.eigMaxit, prec);
		if (po.crit3 && stab) rec.ell = stab->rankOne();
		if (po.onLoadPoint) po.onLoadPoint(k, s, rec);
		rec.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		if (s.opt.verbose)
			std::printf("  k = %2d  t = %.4f  P11 = %+.6e  nu12 = %+.5f  newton = %2d  krylov = %5d  substeps = %d  %s%s\n", k,
				tk, rec.P[0], rec.nu12, rec.newton, rec.krylov, rec.substeps, prec_name(rec.prec), rec.minres ? "  minres" : "");
		res.pts.push_back(rec);
		if (po.stopOnDistortion && rec.voidDistortedFrac > 0.01) {
			res.stopped = true;
			if (s.opt.verbose) std::printf("  path stopped: %.2f%% of the void voxels reach J < 0.1\n", 100 * rec.voidDistortedFrac);
			break;
		}
	}
	s.sp.release(backup);
	res.completed = !res.stopped && !res.failed;
	return res;
}

void write_path_csv(const std::string& file, const PathResult& r) {
	std::ofstream o(file);
	o << "k,t,eps,F11,F22,F33,F12,P11,P12,P13,P21,P22,P23,P31,P32,P33,nu12,nu13,newton,krylov,substeps,failures,minres,"
		 "prec,lambda_cell,lambda_supercell,ellipticity,void_distorted,seconds\n";
	o.precision(10);
	for (const auto& p : r.pts) {
		o << p.k << "," << p.t << "," << p.eps << "," << p.F[0] << "," << p.F[4] << "," << p.F[8] << "," << p.F[1];
		for (int c = 0; c < 9; c++) o << "," << p.P[c];
		o << "," << p.nu12 << "," << p.nu13 << "," << p.newton << "," << p.krylov << "," << p.substeps << "," << p.failures << ","
		  << (p.minres ? 1 : 0) << "," << prec_name(p.prec) << "," << p.lam1 << "," << p.lam2 << "," << p.ell << ","
		  << p.voidDistortedFrac << "," << p.seconds << "\n";
	}
}

}  // namespace fs
