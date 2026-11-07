// Density input and output: OpenVDB files of the GPU solver and .npy arrays of shape [nx, ny, nz].
#pragma once

#include <string>
#include <vector>

namespace fs {

// Reads a density field and returns it in the lexicographic order x + n (y + n z).
// For OpenVDB files n must be given; for .npy files n is taken from the shape (and checked if n > 0).
std::vector<float> read_density(const std::string& path, int& n);
// Periodic trilinear resampling of an ns^3 field (voxel centres) to nt^3.
std::vector<float> resample_periodic(const std::vector<float>& src, int ns, int nt);
void write_density_vdb(const std::string& path, int n, const std::vector<float>& rho);

}  // namespace fs
