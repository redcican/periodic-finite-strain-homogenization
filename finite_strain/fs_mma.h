// Method of moving asymptotes (Svanberg 1987; mmasub and subsolv of Svanberg's 2007 implementation)
// for   min f0(x) + a0 z + sum_i (c_i y_i + d_i y_i^2 / 2)
//       s.t. f_i(x) - a_i z - y_i <= 0,  xmin <= x <= xmax,  y, z >= 0.
#pragma once

#include <vector>

namespace fs {

class MMA {
public:
	MMA(int n, int m);
	// One MMA step: x is the current point on entry and the new point on return.
	// dfdx is row-major m x n. iter starts at 1.
	void update(int iter, std::vector<double>& x, const std::vector<double>& xmin, const std::vector<double>& xmax,
		const std::vector<double>& df0dx, const std::vector<double>& fval, const std::vector<double>& dfdx, double move);

	int n, m;
	double a0 = 1;
	std::vector<double> a, c, d;
	std::vector<double> xold1, xold2, low, upp;
};

}  // namespace fs
