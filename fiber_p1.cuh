#include <memory>
// P1 pilot: push diverse random starting cells onto one fixed cubic stiffness target
// (C11, C12, C44 and volume fraction), i.e. onto the "fiber" of linear homogenization.
//
// Run through the custom-objective path:
//   HOMO_CUSTOM=p1 P1_MODE=<target|fdcheck|project> homo3d -obj custom -reso 32 -sym reflect6 -vol 0.3 -relthres 1e-4 -prefix <dir>/
// Environment:
//   P1_SEEDS   "0-15" or "1,3,5"  seeds of the random starting cells
//   P1_TARGET  "c11,c12,c44"      target invariants (project mode)
//   P1_BETAS   "4,8,16,32"        sharpness continuation of the smooth Heaviside
//   P1_ITERS   per-stage Gauss-Newton iterations (default 15)
//   P1_PERIOD  max wave number of the random starting field (default 3)
#pragma once
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace p1 {

static std::string env(const char* k, const char* d) {
	const char* v = getenv(k);
	return v ? std::string(v) : std::string(d);
}

static std::vector<double> parse_list(const std::string& s) {
	std::vector<double> out;
	std::stringstream ss(s);
	std::string tok;
	while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stod(tok));
	return out;
}

static std::vector<int> parse_seeds(const std::string& s) {
	std::vector<int> out;
	auto dash = s.find('-');
	if (dash != std::string::npos && s.find(',') == std::string::npos) {
		int a = std::stoi(s.substr(0, dash)), b = std::stoi(s.substr(dash + 1));
		for (int i = a; i <= b; i++) out.push_back(i);
	} else {
		for (double v : parse_list(s)) out.push_back(int(v));
	}
	return out;
}

// Solve the small dense system A x = b (n <= 4) by Gaussian elimination with partial pivoting.
static bool solve_small(int n, double A[4][4], double b[4], double x[4]) {
	double M[4][5];
	for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) M[i][j] = A[i][j]; M[i][n] = b[i]; }
	for (int c = 0; c < n; c++) {
		int p = c;
		for (int r = c + 1; r < n; r++) if (fabs(M[r][c]) > fabs(M[p][c])) p = r;
		if (fabs(M[p][c]) < 1e-30) return false;
		for (int j = 0; j <= n; j++) std::swap(M[c][j], M[p][j]);
		for (int r = c + 1; r < n; r++) {
			double f = M[r][c] / M[c][c];
			for (int j = c; j <= n; j++) M[r][j] -= f * M[c][j];
		}
	}
	for (int i = n - 1; i >= 0; i--) {
		double s = M[i][n];
		for (int j = i + 1; j < n; j++) s -= M[i][j] * x[j];
		x[i] = s / M[i][i];
	}
	return true;
}

// Random periodic trigonometric field on the host, standardized to mean 0, std 1.
static void random_field(int nx, int ny, int nz, int seed, int period, std::vector<float>& f) {
	std::mt19937 rng(seed * 7919 + 13);
	std::normal_distribution<double> nd(0.0, 1.0);
	std::uniform_int_distribution<int> ki(-period, period);
	std::uniform_real_distribution<double> ph(0.0, 2 * M_PI);
	const int M = 40;
	std::vector<int> k(3 * M);
	std::vector<double> amp(M), phase(M);
	for (int m = 0; m < M; m++) {
		do { k[3 * m] = ki(rng); k[3 * m + 1] = ki(rng); k[3 * m + 2] = ki(rng); }
		while (k[3 * m] == 0 && k[3 * m + 1] == 0 && k[3 * m + 2] == 0);
		double kn = sqrt(double(k[3 * m] * k[3 * m] + k[3 * m + 1] * k[3 * m + 1] + k[3 * m + 2] * k[3 * m + 2]));
		amp[m] = nd(rng) / kn;
		phase[m] = ph(rng);
	}
	f.assign(size_t(nx) * ny * nz, 0.f);
	for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
		double px = (x + 0.5) / nx, py = (y + 0.5) / ny, pz = (z + 0.5) / nz, s = 0;
		for (int m = 0; m < M; m++)
			s += amp[m] * cos(2 * M_PI * (k[3 * m] * px + k[3 * m + 1] * py + k[3 * m + 2] * pz) + phase[m]);
		f[x + size_t(nx) * (y + size_t(ny) * z)] = float(s);
	}
	double mean = 0, var = 0;
	for (float v : f) mean += v;
	mean /= f.size();
	for (float v : f) var += (v - mean) * (v - mean);
	double sd = sqrt(var / f.size()) + 1e-12;
	for (float& v : f) v = float((v - mean) / sd);
}

static double sigm(double t) { return 1.0 / (1.0 + exp(-t)); }

// Periodic trilinear upsampling of a cell-centred field by integer factor f (keeps cubic symmetry).
static void upsample_periodic(int nx, int ny, int nz, const std::vector<float>& c, int f, std::vector<float>& out) {
	const int mx = nx * f, my = ny * f, mz = nz * f;
	out.assign(size_t(mx) * my * mz, 0.f);
	auto at = [&](int i, int j, int k) {
		i = (i % nx + nx) % nx; j = (j % ny + ny) % ny; k = (k % nz + nz) % nz;
		return c[i + size_t(nx) * (j + size_t(ny) * k)];
	};
	for (int K = 0; K < mz; K++) for (int J = 0; J < my; J++) for (int I = 0; I < mx; I++) {
		// fine centre (I + 0.5) / f in coarse cell units, minus 0.5 to index coarse centres
		double x = (I + 0.5) / f - 0.5, y = (J + 0.5) / f - 0.5, z = (K + 0.5) / f - 0.5;
		int i0 = int(floor(x)), j0 = int(floor(y)), k0 = int(floor(z));
		double fx = x - i0, fy = y - j0, fz = z - k0;
		double v = 0;
		for (int dk = 0; dk < 2; dk++) for (int dj = 0; dj < 2; dj++) for (int di = 0; di < 2; di++)
			v += (di ? fx : 1 - fx) * (dj ? fy : 1 - fy) * (dk ? fz : 1 - fz) * at(i0 + di, j0 + dj, k0 + dk);
		out[I + size_t(mx) * (J + size_t(my) * K)] = float(v);
	}
}

// Remove solid pieces that do not span the periodic cell (floating islands) from the level set {h > thr}.
// Connectivity includes edge and corner contacts, as in the voxel FE model. A piece spans if it touches
// every slice along all three axes; all spanning pieces are kept (symmetric interlocked lattices survive).
// Removed voxels get h = lowval. Returns the number of removed voxels.
static size_t remove_floating(int nx, int ny, int nz, std::vector<float>& h, float thr, float lowval) {
	const size_t N = size_t(nx) * ny * nz;
	std::vector<int> comp(N, -1);
	std::vector<size_t> stack;
	size_t removed = 0;
	int ncomp = 0;
	for (size_t s0 = 0; s0 < N; s0++) {
		if (h[s0] <= thr || comp[s0] >= 0) continue;
		std::vector<size_t> members;
		std::vector<char> tx(nx, 0), ty(ny, 0), tz(nz, 0);
		stack.assign(1, s0);
		comp[s0] = ncomp;
		while (!stack.empty()) {
			size_t v = stack.back(); stack.pop_back();
			members.push_back(v);
			int i = int(v % nx), j = int((v / nx) % ny), k = int(v / (size_t(nx) * ny));
			tx[i] = ty[j] = tz[k] = 1;
			for (int dk = -1; dk <= 1; dk++) for (int dj = -1; dj <= 1; dj++) for (int di = -1; di <= 1; di++) {
				if (!di && !dj && !dk) continue;
				int a = (i + di + nx) % nx, b = (j + dj + ny) % ny, c = (k + dk + nz) % nz;
				size_t u = a + size_t(nx) * (b + size_t(ny) * c);
				if (h[u] > thr && comp[u] < 0) { comp[u] = ncomp; stack.push_back(u); }
			}
		}
		bool spans = true;
		for (char t : tx) spans &= t != 0;
		for (char t : ty) spans &= t != 0;
		for (char t : tz) spans &= t != 0;
		if (!spans) { for (size_t v : members) h[v] = lowval; removed += members.size(); }
		ncomp++;
	}
	return removed;
}

// Values and Jacobian rows of the three cubic invariants for physical density rhop(rho).
// One homogenization solve; the three rows reuse it through holdOn().
template<typename RhoP>
static void eval_cubic(Homogenization& hom, TensorVar<float>& rho, RhoP rhop, cfg::Symmetry sym,
	double C[3], std::vector<std::vector<float>>* rows) {
	auto Ch = genCH(hom, rhop);
	auto e0 = (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) / 3.f;
	auto e1 = (Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) / 3.f;
	auto e2 = (Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) / 3.f;
	Ch.holdOff();
	C[0] = e0.eval();
	if (rows) { rho.diff().reset(); e0.backward(1); symmetrizeField(rho.diff(), sym); rho.diff().toHost((*rows)[0]); }
	// holdOn() skips the solve, but it also skips re-evaluating the density expression, and each
	// node clears its diff buffer only in eval(). Re-evaluate it (no solve) so rows do not accumulate.
	Ch.holdOn();
	Ch.densityField.eval();
	C[1] = e1.eval();
	if (rows) { rho.diff().reset(); e1.backward(1); symmetrizeField(rho.diff(), sym); rho.diff().toHost((*rows)[1]); }
	Ch.densityField.eval();
	C[2] = e2.eval();
	if (rows) { rho.diff().reset(); e2.backward(1); symmetrizeField(rho.diff(), sym); rho.diff().toHost((*rows)[2]); }
	Ch.holdOff();
}

template<typename VolExp>
static double eval_vol(TensorVar<float>& rho, VolExp ve, cfg::Symmetry sym, std::vector<float>* row) {
	double v = ve.eval();
	if (row) { rho.diff().reset(); ve.backward(1); symmetrizeField(rho.diff(), sym); rho.diff().toHost(*row); }
	return v;
}

} // namespace p1

void fiber_p1(cfg::HomoConfig config) {
	using namespace p1;
	setPathPrefix(config.outprefix);
	Homogenization hom(config);
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	const int nx = config.reso[0], ny = config.reso[1], nz = config.reso[2];
	const size_t N = size_t(nx) * ny * nz;
	const float r = config.filterRadius;
	const double vgoal = config.volRatio;
	const cfg::Symmetry sym = config.sym;
	const std::string mode = env("P1_MODE", "project");
	const std::vector<int> seeds = parse_seeds(env("P1_SEEDS", "0-7"));
	std::vector<double> betas = parse_list(env("P1_BETAS", "4,8,16,32"));
	const int iters = std::stoi(env("P1_ITERS", "15"));
	const int period = std::stoi(env("P1_PERIOD", "3"));
	const double lo = 0.001;
	// Void stiffness relative to solid. Below ~1e-3 the mixed-precision MG tends to diverge.
	const float emin = std::stof(env("P1_EMIN", "1e-3"));
	const int dc_iters = std::stoi(env("P1_DC_ITERS", "12"));
	auto K = radial_convker_t<float, Spline4>(r, 0, true, true); // periodic density filter

	TensorVar<float> rho(nx, ny, nz);
	TensorVar<float> rhob(nx, ny, nz);
	// Optional iso-surface check: binary cell from the filtered field upsampled by P1_ISO (e.g. 2 -> 128^3 from 64^3).
	const int iso = std::stoi(env("P1_ISO", "0"));
	std::unique_ptr<Homogenization> homf;
	std::unique_ptr<TensorVar<float>> rhof;
	if (iso > 1) {
		cfg::HomoConfig cf = config;
		for (int i = 0; i < 3; i++) cf.reso[i] = config.reso[i] * iso;
		homf.reset(new Homogenization(cf));
		rhof.reset(new TensorVar<float>(nx * iso, ny * iso, nz * iso));
		printf("[iso] binary check on a %d^3 grid\n", nx * iso);
	}
	std::vector<float> fine;
	std::vector<float> h(N), tmp(N);
	std::vector<std::vector<float>> rows(3, std::vector<float>(N));
	std::vector<float> vrow(N);

	auto clock = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };

	// Starting cell for a seed: random field -> cubic symmetrization -> soft threshold, with the
	// threshold chosen so the physical density (filter + projection at beta) has the goal volume.
	auto make_start = [&](int seed, float beta) {
		random_field(nx, ny, nz, seed, period, h);
		rho.value().fromHost(h);
		symmetrizeField(rho.value(), sym);
		rho.value().toHost(h);
		const double a = 4.0;
		auto soft = [&](double t) {
			for (size_t i = 0; i < N; i++) tmp[i] = float(std::min(1.0, std::max(lo, sigm(a * (h[i] - t)))));
			rho.value().fromHost(tmp);
		};
		double t = 0;
		for (int rep = 0; rep < 5; rep++) {
			double tlo = -6, thi = 6;
			for (int it = 0; it < 40; it++) {
				t = 0.5 * (tlo + thi);
				soft(t);
				double m = (rho.conv(K).sgm(beta, 0.5f).sum() / float(N)).eval();
				if (m > vgoal) tlo = t; else thi = t;
			}
			t = 0.5 * (tlo + thi);
			size_t rem = remove_floating(nx, ny, nz, h, float(t), -20.f);
			printf("[start] seed %d pass %d  threshold %.3f  removed %zu floating voxels\n", seed, rep, t, rem);
			if (rem == 0) break;
		}
		soft(t);
	};

	// Binarize the filtered density at 0.5 and re-solve without filter or projection.
	auto binarized_vox = [&](double Cb[3], double& volb) {
		auto filt = rho.conv(K);
		filt.eval();
		filt.value().toHost(tmp);
		// Drop floating pieces from the binary cell, and from the design so the smooth model agrees.
		std::vector<float> fl(tmp);
		if (remove_floating(nx, ny, nz, fl, 0.5f, 0.f) > 0) {
			rho.value().toHost(h);
			for (size_t i = 0; i < N; i++) if (tmp[i] > 0.5f && fl[i] <= 0.5f) h[i] = float(lo);
			rho.value().fromHost(h);
		}
		size_t cnt = 0;
		for (size_t i = 0; i < N; i++) { bool s = fl[i] > 0.5f; tmp[i] = s ? 1.f : float(lo); cnt += s; }
		volb = double(cnt) / N;
		rhob.value().fromHost(tmp);
		eval_cubic(hom, rhob, rhob.pow(3) * (1.f - emin) + emin, sym, Cb, nullptr);
	};
	auto binarized_iso = [&](double Cb[3], double& volb) {
		auto filt = rho.conv(K);
		filt.eval();
		filt.value().toHost(tmp);
		upsample_periodic(nx, ny, nz, tmp, iso, fine);
		const size_t NF = fine.size();
		// Floating pieces are dropped from the checked cell only. Clearing them in the design made it jump
		// (thin necks that join at the design grid split on the fine grid); target shifting absorbs the change.
		std::vector<float> fl(fine);
		remove_floating(nx * iso, ny * iso, nz * iso, fl, 0.5f, 0.f);
		size_t cnt = 0;
		for (size_t q = 0; q < NF; q++) { bool sd = fl[q] > 0.5f; fl[q] = sd ? 1.f : float(lo); cnt += sd; }
		volb = double(cnt) / NF;
		rhof->value().fromHost(fl);
		eval_cubic(*homf, *rhof, rhof->pow(3) * (1.f - emin) + emin, sym, Cb, nullptr);
	};
	auto binarized = [&](double Cb[3], double& volb) {
		if (iso > 1) binarized_iso(Cb, volb); else binarized_vox(Cb, volb);
	};

	if (mode == "evaliso") {
		// Re-check saved designs: binary cell from the filtered field upsampled by P1_ISO, re-solved.
		// P1_FROM is the run dir holding p1_seedXXX_rho. Writes p1_evaliso.csv.
		if (iso < 2) { printf("set P1_ISO >= 2\n"); return; }
		const std::string from = env("P1_FROM", "");
		std::ofstream ofs(getPath("p1_evaliso.csv"));
		ofs << "seed,iso,c11b,c12b,c44b,volb\n";
		for (int seed : seeds) {
			char name[64];
			sprintf(name, "p1_seed%03d_rho", seed);
			try {
				rho.value().fromVdb(from + name, false);
				double Cb[3], volb;
				binarized_iso(Cb, volb);
				printf("[evaliso] seed %d iso %d  C = (%.4f %.4f %.4f) vol %.4f\n", seed, iso, Cb[0], Cb[1], Cb[2], volb);
				ofs << seed << "," << iso << "," << Cb[0] << "," << Cb[1] << "," << Cb[2] << "," << volb << "\n";
				ofs.flush();
			} catch (const std::exception& e) { printf("[evaliso-fail] seed %d: %s\n", seed, e.what()); }
		}
		return;
	}

	if (mode == "target") {
		// Invariants of the sharpened starting cells; pick the target from these.
		std::ofstream ofs(getPath("p1_target.csv"));
		ofs << "seed,c11,c12,c44,vol,c11b,c12b,c44b,volb\n";
		for (int seed : seeds) {
			float beta = float(betas.back());
			double C[3], Cb[3], volb;
			try {
			make_start(seed, beta);
			eval_cubic(hom, rho, rho.conv(K).sgm(beta, 0.5f).pow(3) * (1.f - emin) + emin, sym, C, nullptr);
			double vol = eval_vol(rho, rho.conv(K).sgm(beta, 0.5f).sum() / float(N), sym, nullptr);
			binarized(Cb, volb);
			printf("[target] seed %d  C = (%.4f %.4f %.4f) vol %.3f | binary C = (%.4f %.4f %.4f) vol %.3f\n",
				seed, C[0], C[1], C[2], vol, Cb[0], Cb[1], Cb[2], volb);
			ofs << seed << "," << C[0] << "," << C[1] << "," << C[2] << "," << vol << ","
				<< Cb[0] << "," << Cb[1] << "," << Cb[2] << "," << volb << "\n";
			ofs.flush();
			} catch (const std::exception& e) { printf("[target-fail] seed %d: %s\n", seed, e.what()); }
		}
		return;
	}

	if (mode == "fdcheck") {
		// Central finite differences of the invariants and volume along a random symmetric direction.
		float beta = 8.f;
		make_start(seeds[0], beta);
		double C[3], Cp[3], Cm[3];
		eval_cubic(hom, rho, rho.conv(K).sgm(beta, 0.5f).pow(3) * (1.f - emin) + emin, sym, C, &rows);
		double vol = eval_vol(rho, rho.conv(K).sgm(beta, 0.5f).sum() / float(N), sym, &vrow);
		std::vector<float> base(N), d(N);
		rho.value().toHost(base);
		random_field(nx, ny, nz, 12345, 4, d);
		rho.value().fromHost(d); symmetrizeField(rho.value(), sym); rho.value().toHost(d);
		double eps = std::stod(env("P1_EPS", "1e-3"));
		for (size_t i = 0; i < N; i++) {
			// keep the perturbation inside the box
			if (base[i] - eps * fabs(d[i]) < lo || base[i] + eps * fabs(d[i]) > 1) d[i] = 0;
		}
		double lin[4] = { 0, 0, 0, 0 };
		for (size_t i = 0; i < N; i++) {
			for (int k = 0; k < 3; k++) lin[k] += double(rows[k][i]) * d[i];
			lin[3] += double(vrow[i]) * d[i];
		}
		for (size_t i = 0; i < N; i++) tmp[i] = base[i] + float(eps) * d[i];
		rho.value().fromHost(tmp);
		eval_cubic(hom, rho, rho.conv(K).sgm(beta, 0.5f).pow(3) * (1.f - emin) + emin, sym, Cp, nullptr);
		double vp = eval_vol(rho, rho.conv(K).sgm(beta, 0.5f).sum() / float(N), sym, nullptr);
		for (size_t i = 0; i < N; i++) tmp[i] = base[i] - float(eps) * d[i];
		rho.value().fromHost(tmp);
		eval_cubic(hom, rho, rho.conv(K).sgm(beta, 0.5f).pow(3) * (1.f - emin) + emin, sym, Cm, nullptr);
		double vm = eval_vol(rho, rho.conv(K).sgm(beta, 0.5f).sum() / float(N), sym, nullptr);
		const char* name[4] = { "C11", "C12", "C44", "vol" };
		double fd[4] = { (Cp[0] - Cm[0]) / (2 * eps), (Cp[1] - Cm[1]) / (2 * eps), (Cp[2] - Cm[2]) / (2 * eps), (vp - vm) / (2 * eps) };
		printf("[fdcheck] base C = (%.5f %.5f %.5f) vol %.4f\n", C[0], C[1], C[2], vol);
		for (int k = 0; k < 4; k++)
			printf("[fdcheck] %s  adjoint %.6e  fd %.6e  ratio %.4f\n", name[k], lin[k], fd[k], lin[k] / (fd[k] + 1e-30));
		return;
	}

	// ---- project mode ----
	std::vector<double> tgt = parse_list(env("P1_TARGET", ""));
	if (tgt.size() != 3) { printf("set P1_TARGET=c11,c12,c44\n"); return; }
	std::ofstream ofs(getPath("p1_project.csv"));
	ofs << "seed,solves,seconds,c11,c12,c44,vol,res_inf,c11b,c12b,c44b,volb,resb_inf,dc_steps,resvox_inf\n";
	const double scale[4] = { tgt[0], tgt[1], tgt[2], vgoal };

	// One Gauss-Newton minimum-norm step on the free set (box-active variables frozen),
	// using the smooth Jacobian in rows/vrow and the residual res. The largest change is capped at move.
	auto gn_step = [&](const double res[4], double move) {
		rho.value().toHost(h);
		std::vector<char> freev(N, 1);
		std::vector<double> delta(N, 0.0);
		const float* R[4] = { rows[0].data(), rows[1].data(), rows[2].data(), vrow.data() };
		for (int pass = 0; pass < 3; pass++) {
			double A[4][4] = {}, lam[4], bb[4];
			for (size_t i = 0; i < N; i++) {
				if (!freev[i]) continue;
				double g[4];
				for (int k = 0; k < 4; k++) g[k] = R[k][i] / scale[k];
				for (int a = 0; a < 4; a++) for (int c = 0; c < 4; c++) A[a][c] += g[a] * g[c];
			}
			for (int k = 0; k < 4; k++) { bb[k] = res[k]; A[k][k] *= (1 + 1e-8); }
			if (!solve_small(4, A, bb, lam)) break;
			bool changed = false;
			for (size_t i = 0; i < N; i++) {
				if (!freev[i]) { delta[i] = 0; continue; }
				double sum = 0;
				for (int k = 0; k < 4; k++) sum += R[k][i] / scale[k] * lam[k];
				delta[i] = sum;
				if ((h[i] <= lo + 1e-6 && sum < 0) || (h[i] >= 1 - 1e-6 && sum > 0)) { freev[i] = 0; changed = true; }
			}
			if (!changed) break;
		}
		double dmax = 0;
		for (size_t i = 0; i < N; i++) dmax = std::max(dmax, fabs(delta[i]));
		double alpha = dmax > move ? move / dmax : 1.0;
		for (size_t i = 0; i < N; i++) h[i] = float(std::min(1.0, std::max(lo, h[i] + alpha * delta[i])));
		rho.value().fromHost(h);
		symmetrizeField(rho.value(), sym);
	};
	auto smooth_rows = [&](float beta, double C[3]) {
		eval_cubic(hom, rho, rho.conv(K).sgm(beta, 0.5f).pow(3) * (1.f - emin) + emin, sym, C, &rows);
		return eval_vol(rho, rho.conv(K).sgm(beta, 0.5f).sum() / float(N), sym, &vrow);
	};
	auto residual = [&](const double C[3], double vol, double res[4]) {
		double rinf = 0;
		for (int k = 0; k < 3; k++) res[k] = (tgt[k] - C[k]) / scale[k];
		res[3] = (vgoal - vol) / scale[3];
		for (int k = 0; k < 4; k++) rinf = std::max(rinf, fabs(res[k]));
		return rinf;
	};

	for (int seed : seeds) {
		double t0 = clock();
		try {
		// P1_FROM set: warm start from a saved design and skip stage 1 (refine against a new binary check).
		const std::string from = env("P1_FROM", "");
		if (from.empty()) make_start(seed, float(betas.front()));
		else { char nm[64]; sprintf(nm, "p1_seed%03d_rho", seed); rho.value().fromVdb(from + nm, false); }
		int solves = 0;
		double C[3] = { 0, 0, 0 }, vol = 0, rinf = 1, res[4];
		// Stage 1: beta continuation on the smooth (filtered, projected, SIMP) model.
		for (double bd : betas) {
			if (!from.empty()) break;
			float beta = float(bd);
			double move = 0.1, prev = 1e30;
			for (int it = 0; it < iters; it++) {
				vol = smooth_rows(beta, C);
				solves++;
				rinf = residual(C, vol, res);
				printf("[p1] seed %d beta %4.0f it %2d  C = (%.4f %.4f %.4f) vol %.4f  |r|inf %.2e\n",
					seed, bd, it, C[0], C[1], C[2], vol, rinf);
				if (rinf > prev) move *= 0.5; else move = std::min(0.2, move * 1.2);
				prev = rinf;
				if (rinf < 1e-3 && it > 0) break;
				gn_step(res, move);
			}
		}
		// Stage 2: target shifting. The binary cell differs from the smooth model by a gap g = C_bin - C_smooth
		// that changes slowly. Shift the smooth target by the measured binary error, re-converge the smooth
		// model (warm start), binarize again; keep the best binary cell.
		const float bfin = float(betas.back());
		const double damp = std::stod(env("P1_SHIFT_DAMP", "0.8"));
		double Cb[3], volb, resb[4], rb = 1e30, best = 1e30;
		double ts[3] = { tgt[0], tgt[1], tgt[2] }, vs = vgoal;
		std::vector<float> hbest(N);
		int dc = 0;
		for (; dc <= dc_iters; dc++) {
			binarized(Cb, volb);
			solves++;
			rb = residual(Cb, volb, resb);
			printf("[p1-dc] seed %d step %2d  binary C = (%.4f %.4f %.4f) vol %.4f  |r|inf %.2e\n",
				seed, dc, Cb[0], Cb[1], Cb[2], volb, rb);
			if (rb < best) { best = rb; rho.value().toHost(hbest); }
			if (best < 2e-3 || dc == dc_iters) break;
			for (int k = 0; k < 3; k++) ts[k] += damp * (tgt[k] - Cb[k]);
			vs += damp * (vgoal - volb);
			double move = 0.05, prev = 1e30;
			for (int it = 0; it < iters; it++) {
				vol = smooth_rows(bfin, C);
				solves++;
				double rs[4] = { (ts[0] - C[0]) / scale[0], (ts[1] - C[1]) / scale[1], (ts[2] - C[2]) / scale[2], (vs - vol) / scale[3] };
				double ri = 0;
				for (int k = 0; k < 4; k++) ri = std::max(ri, fabs(rs[k]));
				if (ri > prev) move *= 0.5; else move = std::min(0.1, move * 1.2);
				prev = ri;
				if (ri < 2e-4) break;
				gn_step(rs, move);
			}
		}
		rho.value().fromHost(hbest);
		vol = smooth_rows(bfin, C);
		rinf = residual(C, vol, res);
		binarized(Cb, volb);
		rb = residual(Cb, volb, resb);
		double secs = clock() - t0;
		printf("[p1-done] seed %d  solves %d  %.1fs  smooth |r| %.2e  binary C = (%.4f %.4f %.4f) vol %.4f  |r| %.2e  dc %d\n",
			seed, solves, secs, rinf, Cb[0], Cb[1], Cb[2], volb, rb, dc);
		// voxel-binary (design grid) numbers of the same cell, for comparison
		double Cv[3], volv, resv[4];
		std::vector<float> hsave(N);
		rho.value().toHost(hsave);           // binarized_vox may edit the design; keep the scored one
		binarized_vox(Cv, volv);
		rho.value().fromHost(hsave);
		double rv = residual(Cv, volv, resv);
		printf("[p1-vox] seed %d  voxel binary C = (%.4f %.4f %.4f) vol %.4f  |r| %.2e\n", seed, Cv[0], Cv[1], Cv[2], volv, rv);
		ofs << seed << "," << solves << "," << secs << "," << C[0] << "," << C[1] << "," << C[2] << "," << vol << "," << rinf << ","
			<< Cb[0] << "," << Cb[1] << "," << Cb[2] << "," << volb << "," << rb << "," << dc << "," << rv << "\n";
		ofs.flush();
		char name[64];
		sprintf(name, "p1_seed%03d_rho", seed); rho.value().toVdb(getPath(name));
		sprintf(name, "p1_seed%03d_vox", seed); rhob.value().toVdb(getPath(name));
		sprintf(name, "p1_seed%03d_bin", seed);
		if (iso > 1) rhof->value().toVdb(getPath(name)); else rhob.value().toVdb(getPath(name));
		} catch (const std::exception& e) {
			printf("[p1-fail] seed %d: %s\n", seed, e.what());
			ofs << seed << ",-1," << clock() - t0 << ",nan,nan,nan,nan,nan,nan,nan,nan,nan,inf,-1,inf\n";
			ofs.flush();
		}
	}
}
