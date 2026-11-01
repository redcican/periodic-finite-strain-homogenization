#pragma once
// Managed (unified) memory is only cheap when the device supports concurrent managed access.
// Without it (WSL2, Windows), the driver moves every managed allocation on each kernel launch,
// costing about 10 us per MB per launch. In that case large buffers are put in device memory.
// Override with HOMO_MANAGED=1 (always managed) or HOMO_MANAGED=0 (never managed).
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

namespace homo {
inline bool useManagedForLargeBuffers(void) {
	static int cached = -1;
	if (cached < 0) {
		const char* env = std::getenv("HOMO_MANAGED");
		if (env && (env[0] == '0' || env[0] == '1')) {
			cached = env[0] - '0';
		} else {
			int dev = 0, concurrent = 0;
			cudaGetDevice(&dev);
			cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev);
			cached = concurrent ? 1 : 0;
		}
		printf("[mem] large buffers use %s memory\n", cached ? "managed" : "device");
	}
	return cached == 1;
}
}
