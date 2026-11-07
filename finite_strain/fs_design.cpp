#include "fs_design.h"

#include "fs_io.h"
#include "fs_mma.h"
#include "fs_stability.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace fs {

namespace {

// Design variables on one forty-eighth of the cell: voxel (i, j, k) maps to the sorted triple of its
// folded coordinates min(i, n - 1 - i), which is invariant under the reflections of the cube.
struct SymMap {
	int n = 0, nred = 0;
	std::vector<int> map;
	std::vector<int> count;
};

SymMap make_symmap(int n) {
	SymMap s;
	s.n = n;
	int h = n / 2;
	std::vector<int> id(size_t(h) * h * h, -1);
	for (int a = 0; a < h; a++)
		for (int b = a; b < h; b++)
			for (int c = b; c < h; c++) id[a + size_t(h) * (b + size_t(h) * c)] = s.nred++;
	s.map.resize(size_t(n) * n * n);
	s.count.assign(s.nred, 0);
	for (int k = 0; k < n; k++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++) {
				int f[3] = {std::min(i, n - 1 - i), std::min(j, n - 1 - j), std::min(k, n - 1 - k)};
				std::sort(f, f + 3);
				int r = id[f[0] + size_t(h) * (f[1] + size_t(h) * f[2])];
				s.map[i + size_t(n) * (j + size_t(n) * k)] = r;
				s.count[r]++;
			}
	return s;
}

bool youngs_modulus(const double C[36], double& E, double W[36]) {
	Eigen::Matrix<double, 6, 6> Cm;
	for (int i = 0; i < 36; i++) Cm(i / 6, i % 6) = C[i];
	Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(Cm);
	if (ldlt.info() != Eigen::Success) return false;
	Eigen::Matrix<double, 6, 6> Sm = ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
	if (!(Sm(0, 0) > 0)) return false;
	E = 1.0 / Sm(0, 0);
	// dE = E^2 (S dC S)_11  ->  W_IJ = E^2 S_1I S_1J
	for (int I = 0; I < 6; I++)
		for (int J = 0; J < 6; J++) W[I * 6 + J] = E * E * Sm(0, I) * Sm(0, J);
	return true;
}

struct Evaluation {
	bool ok = false;
	double V = 0, E = 0;
	std::vector<double> nu, P;  // per load point
	double JS = 0;
	std::vector<double> gV, gE, gJS;
	std::vector<std::vector<double>> gnu;
	int newton = 0, krylov = 0;
};

class Designer {
public:
	Designer(Solver& s, const DesignOptions& o) : S(s), opt(o), sym(make_symmap(s.n)) {
		ne = S.cell.ne;
		r = opt.filterR > 0 ? float(opt.filterR) : float(S.n) / 32.f;
		xf = dalloc_d(ne);
		rt = dalloc_d(ne);
		rho = dalloc_d(ne);
		drho = dalloc_d(ne);
		t1 = dalloc_d(ne);
		t2 = dalloc_d(ne);
		gE = dalloc_d(ne);
		gJ = dalloc_d(ne);
		for (int k = 0; k < opt.steps; k++) gnu.push_back(dalloc_d(ne));
		prec = opt.prec == "fp32" ? Prec::FP32 : Prec::FP16;
		double emax = opt.steps * opt.inc;
		Pstar.resize(opt.steps);
		Pstar2 = 0;
		for (int k = 0; k < opt.steps; k++) {
			double e = (k + 1) * opt.inc;
			Pstar[k] = opt.Estar * (e + (opt.S - 1) * e * e / (2 * emax));
			Pstar2 += Pstar[k] * Pstar[k];
		}
	}
	~Designer() {
		for (double* p : {xf, rt, rho, drho, t1, t2, gE, gJ}) dfree(p);
		for (double* p : gnu) dfree(p);
	}

	std::vector<double> reduce_init(const std::vector<float>& x0) const {
		std::vector<double> xr(sym.nred, 0.0);
		for (size_t e = 0; e < ne; e++) xr[sym.map[e]] += x0[e];
		for (int i = 0; i < sym.nred; i++) xr[i] /= sym.count[i];
		return xr;
	}

	// physical density of the reduced variables (host copy in rhoHost)
	void forward(const std::vector<double>& xr, double beta) {
		std::vector<double> full(ne);
		for (size_t e = 0; e < ne; e++) full[e] = xr[sym.map[e]];
		host2d(full.data(), xf, ne);
		filter_periodic(S.n, r, xf, rt);
		project_heaviside(ne, float(beta), 0.5f, rt, rho, drho);
		std::vector<double> rh(ne);
		d2host(rho, rh.data(), ne);
		rhoHost.assign(rh.begin(), rh.end());
	}

	// gradient with respect to rho (device) -> gradient with respect to the reduced variables
	std::vector<double> chain(const double* g) {
		dmul(g, drho, t1, ne);
		filter_periodic(S.n, r, t1, t2);
		std::vector<double> h(ne), gr(sym.nred, 0.0);
		d2host(t2, h.data(), ne);
		for (size_t e = 0; e < ne; e++) gr[sym.map[e]] += h[e];
		return gr;
	}

	Evaluation evaluate(const std::vector<double>& xr, double beta) {
		Evaluation ev;
		forward(xr, beta);
		S.setDensity(rhoHost);
		// small-strain moduli and E^H with its sensitivity
		double C[36], W[36];
		bool okLin = S.linearModuli(C, nullptr, false, opt.linTol, prec);
		if (!youngs_modulus(C, ev.E, W)) return ev;
		dzero(gE, ne);
		S.moduliSensitivity(W, gE);
		double V = 0;
		for (float v : rhoHost) V += v;
		ev.V = V / double(ne);
		// load path with adjoints at the load points
		LoadPath path = make_path("L1", opt.steps, opt.inc);
		PathOptions po;
		po.precMode = opt.prec;
		po.crit1Every = opt.crit1Every;
		bool okAdj = true;
		ev.nu.assign(opt.steps, 0.0);
		ev.P.assign(opt.steps, 0.0);
		dzero(gJ, ne);
		po.onLoadPoint = [&](int k, Solver& s, PointRecord& rec) {
			ev.nu[k - 1] = rec.nu12;
			ev.P[k - 1] = rec.P[0];
			double a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, b[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
			if (opt.program == "nu") {
				b[0] = -1.0 / rec.eps;  // d nu12 / d Fbar_22
				dzero(gnu[k - 1], ne);
				if (!s.adjoint(a, b, gnu[k - 1], opt.linTol, prec)) okAdj = false;
			} else {
				a[0] = 2 * (rec.P[0] - Pstar[k - 1]) / Pstar2;
				if (!s.adjoint(a, nullptr, gJ, opt.linTol, prec)) okAdj = false;
			}
		};
		PathResult pr = run_path(S, path, po, nullptr);
		ev.newton = pr.newton;
		ev.krylov = pr.krylov;
		if (!pr.completed || int(pr.pts.size()) != opt.steps) return ev;
		ev.gE = chain(gE);
		{
			double* ones = dalloc_d(ne);
			dfill(1.0 / double(ne), ones, ne);
			ev.gV = chain(ones);
			dfree(ones);
		}
		if (opt.program == "nu") {
			for (int k = 0; k < opt.steps; k++) ev.gnu.push_back(chain(gnu[k]));
		} else {
			ev.JS = 0;
			for (int k = 0; k < opt.steps; k++) ev.JS += (ev.P[k] - Pstar[k]) * (ev.P[k] - Pstar[k]);
			ev.JS /= Pstar2;
			ev.gJS = chain(gJ);
		}
		if (!okLin || !okAdj) std::printf("  warning: corrector or adjoint solves did not reach the tolerance\n");
		ev.ok = true;
		return ev;
	}

	double nuStar(int k) const { return opt.nuStar + opt.nuSlope * (k + 1) * opt.inc; }

	Solver& S;
	DesignOptions opt;
	SymMap sym;
	size_t ne;
	float r;
	Prec prec;
	double *xf, *rt, *rho, *drho, *t1, *t2, *gE, *gJ;
	std::vector<double*> gnu;
	std::vector<double> Pstar;
	double Pstar2;
	std::vector<float> rhoHost;
};

}  // namespace

bool reference_values(Solver& S, const std::vector<float>& rho, Prec p, double tol, double& V, double& E) {
	S.setDensity(rho);
	double C[36], W[36];
	bool ok = S.linearModuli(C, nullptr, false, tol, p);
	double v = 0;
	for (float x : rho) v += x;
	V = v / double(rho.size());
	return youngs_modulus(C, E, W) && ok;
}

int run_design(Solver& S, const std::vector<float>& x0, const DesignOptions& o) {
	if (!(o.Vstar > 0) || !(o.Estar > 0)) {
		std::fprintf(stderr, "design: V* and E* must be positive\n");
		return 1;
	}
	if (o.program != "nu" && o.program != "stiff") {
		std::fprintf(stderr, "design: unknown program %s (nu or stiff)\n", o.program.c_str());
		return 1;
	}
	Designer D(S, o);
	const bool nuProg = o.program == "nu";
	const int nred = D.sym.nred;
	const int nvar = nuProg ? nred + 1 : nred;  // the bound variable z (scaled) comes last
	const int m = nuProg ? o.steps + 2 : 2;
	MMA mma(nvar, m);
	std::vector<double> x(nvar, 0.0), xmin(nvar, 0.0), xmax(nvar, 1.0);
	std::vector<double> xr0 = D.reduce_init(x0);
	for (int i = 0; i < nred; i++) x[i] = std::min(1.0, std::max(0.0, xr0[i]));
	double zs = 1.0;  // scale of the deviation bound, set from the first evaluation
	if (nuProg) xmax[nvar - 1] = 10.0;

	std::ofstream log(o.prefix + "design.csv");
	log << "iter,beta,move,objective,max_deviation,V_ratio,E_ratio,newton,krylov,rejected\n";
	log.precision(10);

	std::vector<double> df0(nvar), fval(m), dfdx(size_t(m) * nvar);
	auto assemble = [&](const Evaluation& ev, double& f0) {
		std::fill(df0.begin(), df0.end(), 0.0);
		std::fill(dfdx.begin(), dfdx.end(), 0.0);
		int c = 0;
		if (nuProg) {
			f0 = x[nvar - 1];
			df0[nvar - 1] = 1.0;
			for (int k = 0; k < o.steps; k++, c++) {
				double dev = ev.nu[k] - D.nuStar(k);
				fval[c] = dev * dev / zs - x[nvar - 1];
				for (int i = 0; i < nred; i++) dfdx[size_t(c) * nvar + i] = 2 * dev / zs * ev.gnu[k][i];
				dfdx[size_t(c) * nvar + nvar - 1] = -1.0;
			}
		} else {
			f0 = ev.JS;
			for (int i = 0; i < nred; i++) df0[i] = ev.gJS[i];
		}
		fval[c] = ev.V / o.Vstar - 1.0;
		for (int i = 0; i < nred; i++) dfdx[size_t(c) * nvar + i] = ev.gV[i] / o.Vstar;
		c++;
		fval[c] = 1.0 - ev.E / o.Estar;
		for (int i = 0; i < nred; i++) dfdx[size_t(c) * nvar + i] = -ev.gE[i] / o.Estar;
	};
	auto maxdev = [&](const Evaluation& ev) {
		double md = 0;
		for (int k = 0; k < o.steps; k++) md = std::max(md, std::fabs(ev.nu[k] - D.nuStar(k)));
		return md;
	};

	double move = o.move;
	double beta = 1.0;
	Evaluation ev = D.evaluate(x, beta);
	if (!ev.ok) {
		std::fprintf(stderr, "design: the load path of the initial design does not complete\n");
		return 2;
	}
	if (nuProg) {
		double md = maxdev(ev);
		zs = std::max(1e-6, md * md);
		x[nvar - 1] = 1.0;
	}
	int rejected = 0;
	for (int it = 1; it <= o.iters; it++) {
		double f0;
		assemble(ev, f0);
		std::printf("design %4d  beta = %4.0f  objective = %.6e  max|nu - nu*| = %.4f  V/V* = %.4f  E/E* = %.4f  move = %.3f\n", it - 1,
			beta, f0, nuProg ? maxdev(ev) : 0.0, ev.V / o.Vstar, ev.E / o.Estar, move);
		log << it - 1 << "," << beta << "," << move << "," << f0 << "," << (nuProg ? maxdev(ev) : 0.0) << "," << ev.V / o.Vstar << ","
			<< ev.E / o.Estar << "," << ev.newton << "," << ev.krylov << "," << rejected << "\n";
		log.flush();
		if (o.saveEvery > 0 && (it - 1) % o.saveEvery == 0)
			write_density_vdb(o.prefix + "rho_" + std::to_string(it - 1) + ".vdb", S.n, D.rhoHost);

		// MMA step; a design whose load path fails is rejected and the step repeated with half the move limit
		MMA saved = mma;
		std::vector<double> xprev = x;
		Evaluation evNew;
		double newBeta = std::min(o.betaMax, std::pow(2.0, std::floor(double(it) / o.betaEvery)));
		for (int attempt = 0; attempt < 6; attempt++) {
			mma = saved;
			x = xprev;
			mma.update(it, x, xmin, xmax, df0, fval, dfdx, move);
			evNew = D.evaluate(x, newBeta);
			if (evNew.ok) break;
			rejected++;
			move *= 0.5;
			std::printf("  load path failed: update rejected, move limit %.4f\n", move);
		}
		if (!evNew.ok) {
			x = xprev;
			std::fprintf(stderr, "design: no admissible update after six move-limit halvings\n");
			break;
		}
		ev = evNew;
		beta = newBeta;
	}
	D.forward(std::vector<double>(x.begin(), x.begin() + nred), beta);
	write_density_vdb(o.prefix + "rho_design.vdb", S.n, D.rhoHost);

	if (o.finalCheck) {
		std::vector<float> bin(D.rhoHost.size());
		for (size_t e = 0; e < bin.size(); e++) bin[e] = D.rhoHost[e] >= 0.5f ? 1.f : 0.f;
		write_density_vdb(o.prefix + "rho_thresholded.vdb", S.n, bin);
		Prec p = o.prec == "fp32" ? Prec::FP32 : Prec::FP16;
		double V, E;
		reference_values(S, bin, p, o.linTol, V, E);
		std::printf("thresholded design: V/V* = %.4f  E/E* = %.4f\n", V / o.Vstar, E / o.Estar);
		LoadPath path = make_path("L1", int(std::lround(o.steps * o.inc / 0.005)), 0.005);
		PathOptions po;
		po.precMode = o.prec;
		po.crit1Every = 1;
		po.crit2 = true;
		po.crit3 = true;
		Stability stab(S);
		PathResult pr = run_path(S, path, po, &stab);
		write_path_csv(o.prefix + "final_L1.csv", pr);
		std::ofstream sm(o.prefix + "final_summary.txt");
		sm << "V_ratio " << V / o.Vstar << "\nE_ratio " << E / o.Estar << "\ncompleted " << (pr.completed ? 1 : 0) << "\n";
	}
	return 0;
}

}  // namespace fs
