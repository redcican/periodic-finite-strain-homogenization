#include "fs_mma.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace fs {

namespace {

using Vec = std::vector<double>;

struct Sub {
	int m, n;
	const Vec &low, &upp, &alfa, &beta, &p0, &q0, &P, &Q, &a, &b, &c, &d;
	double a0;
	Vec x, y, lam, xsi, eta, mu, s;
	double z = 1, zet = 1;

	// residual norm and max of the perturbed KKT conditions
	void residual(double epsi, double& norm, double& mx) const {
		double s2 = 0, smax = 0;
		auto acc = [&](double r) {
			s2 += r * r;
			smax = std::max(smax, std::fabs(r));
		};
		Vec plam(p0), qlam(q0), gvec(m, 0.0);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++) {
				plam[j] += P[size_t(i) * n + j] * lam[i];
				qlam[j] += Q[size_t(i) * n + j] * lam[i];
			}
		for (int j = 0; j < n; j++) {
			double ux1 = upp[j] - x[j], xl1 = x[j] - low[j];
			for (int i = 0; i < m; i++) gvec[i] += P[size_t(i) * n + j] / ux1 + Q[size_t(i) * n + j] / xl1;
			double dpsidx = plam[j] / (ux1 * ux1) - qlam[j] / (xl1 * xl1);
			acc(dpsidx - xsi[j] + eta[j]);              // rex
			acc(xsi[j] * (x[j] - alfa[j]) - epsi);       // rexsi
			acc(eta[j] * (beta[j] - x[j]) - epsi);       // reeta
		}
		double rez = a0 - zet;
		for (int i = 0; i < m; i++) {
			acc(c[i] + d[i] * y[i] - mu[i] - lam[i]);    // rey
			rez -= a[i] * lam[i];
			acc(gvec[i] - a[i] * z - y[i] + s[i] - b[i]);  // relam
			acc(mu[i] * y[i] - epsi);                    // remu
			acc(lam[i] * s[i] - epsi);                   // res
		}
		acc(rez);
		acc(zet * z - epsi);
		norm = std::sqrt(s2);
		mx = smax;
	}

	void solve(double epsimin) {
		x.assign(n, 0);
		xsi.assign(n, 0);
		eta.assign(n, 0);
		for (int j = 0; j < n; j++) {
			x[j] = 0.5 * (alfa[j] + beta[j]);
			xsi[j] = std::max(1.0, 1.0 / (x[j] - alfa[j]));
			eta[j] = std::max(1.0, 1.0 / (beta[j] - x[j]));
		}
		y.assign(m, 1.0);
		lam.assign(m, 1.0);
		s.assign(m, 1.0);
		mu.assign(m, 0);
		for (int i = 0; i < m; i++) mu[i] = std::max(1.0, 0.5 * c[i]);
		z = 1;
		zet = 1;
		double epsi = 1;
		while (epsi > epsimin) {
			double rnorm, rmax;
			residual(epsi, rnorm, rmax);
			int ittt = 0;
			while (rmax > 0.9 * epsi && ittt < 200) {
				ittt++;
				Vec plam(p0), qlam(q0);
				for (int i = 0; i < m; i++)
					for (int j = 0; j < n; j++) {
						plam[j] += P[size_t(i) * n + j] * lam[i];
						qlam[j] += Q[size_t(i) * n + j] * lam[i];
					}
				Vec gvec(m, 0.0), delx(n), diagx(n), GG(size_t(m) * n);
				for (int j = 0; j < n; j++) {
					double ux1 = upp[j] - x[j], xl1 = x[j] - low[j];
					double ux2 = ux1 * ux1, xl2 = xl1 * xl1;
					for (int i = 0; i < m; i++) {
						gvec[i] += P[size_t(i) * n + j] / ux1 + Q[size_t(i) * n + j] / xl1;
						GG[size_t(i) * n + j] = P[size_t(i) * n + j] / ux2 - Q[size_t(i) * n + j] / xl2;
					}
					double dpsidx = plam[j] / ux2 - qlam[j] / xl2;
					delx[j] = dpsidx - epsi / (x[j] - alfa[j]) + epsi / (beta[j] - x[j]);
					diagx[j] = 2 * (plam[j] / (ux2 * ux1) + qlam[j] / (xl2 * xl1)) + xsi[j] / (x[j] - alfa[j]) + eta[j] / (beta[j] - x[j]);
				}
				Vec dely(m), dellam(m), diagy(m), diaglamyi(m);
				double delz = a0 - epsi / z;
				for (int i = 0; i < m; i++) {
					dely[i] = c[i] + d[i] * y[i] - lam[i] - epsi / y[i];
					delz -= a[i] * lam[i];
					dellam[i] = gvec[i] - a[i] * z - y[i] - b[i] + epsi / lam[i];
					diagy[i] = d[i] + mu[i] / y[i];
					diaglamyi[i] = s[i] / lam[i] + 1.0 / diagy[i];
				}
				Eigen::MatrixXd AA = Eigen::MatrixXd::Zero(m + 1, m + 1);
				Eigen::VectorXd bb(m + 1);
				for (int i = 0; i < m; i++) {
					double v = dellam[i] + dely[i] / diagy[i];
					for (int j = 0; j < n; j++) v -= GG[size_t(i) * n + j] * delx[j] / diagx[j];
					bb(i) = v;
					AA(i, i) = diaglamyi[i];
					for (int k = 0; k <= i; k++) {
						double t = 0;
						for (int j = 0; j < n; j++) t += GG[size_t(i) * n + j] * GG[size_t(k) * n + j] / diagx[j];
						AA(i, k) += t;
						if (k != i) AA(k, i) += t;
					}
					AA(i, m) = a[i];
					AA(m, i) = a[i];
				}
				AA(m, m) = -zet / z;
				bb(m) = delz;
				Eigen::VectorXd sol = AA.colPivHouseholderQr().solve(bb);
				Vec dlam(m), dx(n), dy(m), dxsi(n), deta(n), dmu(m), ds(m);
				for (int i = 0; i < m; i++) dlam[i] = sol(i);
				double dz = sol(m);
				for (int j = 0; j < n; j++) {
					double gl = 0;
					for (int i = 0; i < m; i++) gl += GG[size_t(i) * n + j] * dlam[i];
					dx[j] = -delx[j] / diagx[j] - gl / diagx[j];
					dxsi[j] = -xsi[j] + epsi / (x[j] - alfa[j]) - xsi[j] * dx[j] / (x[j] - alfa[j]);
					deta[j] = -eta[j] + epsi / (beta[j] - x[j]) + eta[j] * dx[j] / (beta[j] - x[j]);
				}
				for (int i = 0; i < m; i++) {
					dy[i] = -dely[i] / diagy[i] + dlam[i] / diagy[i];
					dmu[i] = -mu[i] + epsi / y[i] - mu[i] * dy[i] / y[i];
					ds[i] = -s[i] + epsi / lam[i] - s[i] * dlam[i] / lam[i];
				}
				double dzet = -zet + epsi / z - zet * dz / z;
				// largest step that keeps the variables positive
				double stm = 1.0;
				auto upd = [&](double v, double dv) { stm = std::max(stm, -1.01 * dv / v); };
				for (int i = 0; i < m; i++) {
					upd(y[i], dy[i]);
					upd(lam[i], dlam[i]);
					upd(mu[i], dmu[i]);
					upd(s[i], ds[i]);
				}
				upd(z, dz);
				upd(zet, dzet);
				for (int j = 0; j < n; j++) {
					upd(xsi[j], dxsi[j]);
					upd(eta[j], deta[j]);
					stm = std::max(stm, -1.01 * dx[j] / (x[j] - alfa[j]));
					stm = std::max(stm, 1.01 * dx[j] / (beta[j] - x[j]));
				}
				double steg = 1.0 / stm;
				Vec x0 = x, y0 = y, lam0 = lam, xsi0 = xsi, eta0 = eta, mu0 = mu, s0 = s;
				double z0 = z, zet0 = zet;
				int itto = 0;
				double resinew = 2 * rnorm;
				while (resinew > rnorm && itto < 50) {
					itto++;
					for (int j = 0; j < n; j++) {
						x[j] = x0[j] + steg * dx[j];
						xsi[j] = xsi0[j] + steg * dxsi[j];
						eta[j] = eta0[j] + steg * deta[j];
					}
					for (int i = 0; i < m; i++) {
						y[i] = y0[i] + steg * dy[i];
						lam[i] = lam0[i] + steg * dlam[i];
						mu[i] = mu0[i] + steg * dmu[i];
						s[i] = s0[i] + steg * ds[i];
					}
					z = z0 + steg * dz;
					zet = zet0 + steg * dzet;
					double rm;
					residual(epsi, resinew, rm);
					rmax = rm;
					steg *= 0.5;
				}
				rnorm = resinew;
			}
			epsi *= 0.1;
		}
	}
};

}  // namespace

MMA::MMA(int n_, int m_) : n(n_), m(m_), a(m_, 0.0), c(m_, 1000.0), d(m_, 1.0), xold1(n_), xold2(n_), low(n_), upp(n_) {}

void MMA::update(int iter, std::vector<double>& x, const std::vector<double>& xmin, const std::vector<double>& xmax,
	const std::vector<double>& df0dx, const std::vector<double>& fval, const std::vector<double>& dfdx, double move) {
	const double epsimin = 1e-7, raa0 = 1e-5, albefa = 0.1, asyinit = 0.5, asyincr = 1.2, asydecr = 0.7;
	Vec alfa(n), beta(n), p0(n), q0(n), P(size_t(m) * n), Q(size_t(m) * n), b(m, 0.0);
	for (int j = 0; j < n; j++) {
		double w = xmax[j] - xmin[j];
		if (iter < 3) {
			low[j] = x[j] - asyinit * w;
			upp[j] = x[j] + asyinit * w;
		} else {
			double zzz = (x[j] - xold1[j]) * (xold1[j] - xold2[j]);
			double factor = zzz > 0 ? asyincr : (zzz < 0 ? asydecr : 1.0);
			low[j] = x[j] - factor * (xold1[j] - low[j]);
			upp[j] = x[j] + factor * (upp[j] - xold1[j]);
			low[j] = std::min(std::max(low[j], x[j] - 10 * w), x[j] - 0.01 * w);
			upp[j] = std::max(std::min(upp[j], x[j] + 10 * w), x[j] + 0.01 * w);
		}
		alfa[j] = std::max(std::max(low[j] + albefa * (x[j] - low[j]), x[j] - move * w), xmin[j]);
		beta[j] = std::min(std::min(upp[j] - albefa * (upp[j] - x[j]), x[j] + move * w), xmax[j]);
		double xmamiinv = 1.0 / std::max(w, 1e-5);
		double ux1 = upp[j] - x[j], xl1 = x[j] - low[j];
		double ux2 = ux1 * ux1, xl2 = xl1 * xl1;
		double pp = std::max(df0dx[j], 0.0), qq = std::max(-df0dx[j], 0.0);
		double pq = 0.001 * (pp + qq) + raa0 * xmamiinv;
		p0[j] = (pp + pq) * ux2;
		q0[j] = (qq + pq) * xl2;
		for (int i = 0; i < m; i++) {
			double g = dfdx[size_t(i) * n + j];
			double Pi = std::max(g, 0.0), Qi = std::max(-g, 0.0);
			double PQ = 0.001 * (Pi + Qi) + raa0 * xmamiinv;
			P[size_t(i) * n + j] = (Pi + PQ) * ux2;
			Q[size_t(i) * n + j] = (Qi + PQ) * xl2;
			b[i] += P[size_t(i) * n + j] / ux1 + Q[size_t(i) * n + j] / xl1;
		}
	}
	for (int i = 0; i < m; i++) b[i] -= fval[i];
	Sub sub{m, n, low, upp, alfa, beta, p0, q0, P, Q, a, b, c, d, a0};
	sub.solve(epsimin);
	xold2 = xold1;
	xold1 = x;
	x = sub.x;
}

}  // namespace fs
