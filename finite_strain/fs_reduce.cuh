// Block reduction of M doubles followed by one atomic add per block.
#pragma once

#include "fs_common.h"

namespace fs {

// All threads of the block must call this function; blockDim.x must be a multiple of 32.
template <int M>
__device__ inline void block_sum_atomic(double (&v)[M], double* out) {
	__shared__ double sh[M][32];
	int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
	for (int m = 0; m < M; m++) {
		double x = v[m];
		for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
		if (lane == 0) sh[m][wid] = x;
	}
	__syncthreads();
	if (wid == 0) {
		int nw = (blockDim.x + 31) >> 5;
		for (int m = 0; m < M; m++) {
			double x = lane < nw ? sh[m][lane] : 0.0;
			for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
			if (lane == 0 && out != nullptr) atomicAdd(&out[m], x);
		}
	}
}

__device__ inline int wrapi(int i, int n) { return i >= n ? i - n : (i < 0 ? i + n : i); }

__device__ inline size_t nid(int i, int j, int k, int n) {
	return size_t(i) + size_t(n) * (size_t(j) + size_t(n) * size_t(k));
}

}  // namespace fs
