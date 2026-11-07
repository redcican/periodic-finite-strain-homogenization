#include "fs_io.h"

#include "voxelIO/openvdb_wrapper_t.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace fs {

namespace {

bool ends_with(const std::string& s, const std::string& e) {
	return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}

std::vector<float> read_npy(const std::string& path, int& n) {
	std::ifstream in(path, std::ios::binary);
	if (!in) throw std::runtime_error("cannot open " + path);
	char magic[6];
	in.read(magic, 6);
	if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + " is not a .npy file");
	unsigned char ver[2];
	in.read(reinterpret_cast<char*>(ver), 2);
	size_t hlen = 0;
	if (ver[0] == 1) {
		unsigned char b[2];
		in.read(reinterpret_cast<char*>(b), 2);
		hlen = b[0] | (size_t(b[1]) << 8);
	} else {
		unsigned char b[4];
		in.read(reinterpret_cast<char*>(b), 4);
		hlen = b[0] | (size_t(b[1]) << 8) | (size_t(b[2]) << 16) | (size_t(b[3]) << 24);
	}
	std::string hdr(hlen, ' ');
	in.read(&hdr[0], std::streamsize(hlen));
	bool f8 = hdr.find("'<f8'") != std::string::npos, f4 = hdr.find("'<f4'") != std::string::npos;
	if (!f8 && !f4) throw std::runtime_error(path + ": only little-endian float32 or float64 arrays are supported");
	bool fortran = hdr.find("'fortran_order': True") != std::string::npos;
	size_t sp = hdr.find("'shape':");
	if (sp == std::string::npos) throw std::runtime_error(path + ": no shape in the header");
	int a = 0, b = 0, c = 0;
	if (std::sscanf(hdr.c_str() + hdr.find('(', sp), "(%d, %d, %d", &a, &b, &c) != 3 || a != b || b != c)
		throw std::runtime_error(path + ": expected a cubic array of shape (n, n, n)");
	if (n > 0 && n != a) throw std::runtime_error(path + ": resolution does not match -reso");
	n = a;
	size_t N = size_t(n) * n * n;
	std::vector<double> raw(N);
	if (f8) {
		in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(N * 8));
	} else {
		std::vector<float> rf(N);
		in.read(reinterpret_cast<char*>(rf.data()), std::streamsize(N * 4));
		for (size_t i = 0; i < N; i++) raw[i] = rf[i];
	}
	if (!in) throw std::runtime_error(path + " is incomplete");
	std::vector<float> rho(N);
	for (int z = 0; z < n; z++)
		for (int y = 0; y < n; y++)
			for (int x = 0; x < n; x++) {
				size_t lex = x + size_t(n) * (y + size_t(n) * z);
				size_t src = fortran ? lex : z + size_t(n) * (y + size_t(n) * x);
				rho[lex] = float(raw[src]);
			}
	return rho;
}

}  // namespace

std::vector<float> read_density(const std::string& path, int& n) {
	if (ends_with(path, ".npy")) return read_npy(path, n);
	if (n <= 0) throw std::runtime_error("the resolution (-reso) is required for OpenVDB input");
	std::vector<int> pos[3];
	std::vector<float> val;
	openvdb_wrapper_t<float>::openVDBfile2grid(path, pos, val);
	size_t N = size_t(n) * n * n;
	std::vector<float> rho(N, 0.f);
	size_t outside = 0;
	for (size_t i = 0; i < val.size(); i++) {
		int x = pos[0][i], y = pos[1][i], z = pos[2][i];
		if (x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n) {
			outside++;
			continue;
		}
		rho[x + size_t(n) * (y + size_t(n) * z)] = val[i];
	}
	if (outside) std::fprintf(stderr, "warning: %zu voxels of %s lie outside the %d^3 grid\n", outside, path.c_str(), n);
	return rho;
}

void write_density_vdb(const std::string& path, int n, const std::vector<float>& rho) {
	int gs[3] = {n, n, n};
	openvdb_wrapper_t<float>::lexicalGrid2openVDBfile(path, gs, rho);
}

}  // namespace fs
