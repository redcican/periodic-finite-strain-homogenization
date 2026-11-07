// Convert the first float grid of an OpenVDB file to a dense .npy array (float32, shape [nx, ny, nz]).
// Usage: vdb2npy in.vdb out.npy
#include <openvdb/openvdb.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
	if (argc != 3) { std::fprintf(stderr, "usage: %s in.vdb out.npy\n", argv[0]); return 1; }
	openvdb::initialize();
	openvdb::io::File file(argv[1]);
	file.open();
	openvdb::FloatGrid::Ptr grid;
	for (auto it = file.beginName(); it != file.endName(); ++it) {
		auto base = file.readGrid(it.gridName());
		grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
		if (grid) { std::printf("grid '%s'\n", it.gridName().c_str()); break; }
	}
	file.close();
	if (!grid) { std::fprintf(stderr, "no float grid in %s\n", argv[1]); return 1; }

	openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
	openvdb::Coord lo = box.min(), dim = box.dim();
	std::printf("bbox min (%d %d %d), dims %d x %d x %d, background %g\n",
		lo.x(), lo.y(), lo.z(), dim.x(), dim.y(), dim.z(), grid->background());

	std::vector<float> data((size_t)dim.x() * dim.y() * dim.z());
	auto acc = grid->getConstAccessor();
	size_t n = 0;
	for (int i = 0; i < dim.x(); i++)
		for (int j = 0; j < dim.y(); j++)
			for (int k = 0; k < dim.z(); k++)
				data[n++] = acc.getValue(lo.offsetBy(i, j, k));

	// .npy v1.0 header, C order
	std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
		std::to_string(dim.x()) + ", " + std::to_string(dim.y()) + ", " + std::to_string(dim.z()) + "), }";
	while ((10 + hdr.size() + 1) % 64) hdr += ' ';
	hdr += '\n';
	std::ofstream out(argv[2], std::ios::binary);
	out.write("\x93NUMPY\x01\x00", 8);
	unsigned short hlen = (unsigned short)hdr.size();
	out.write((const char*)&hlen, 2);
	out.write(hdr.data(), hdr.size());
	out.write((const char*)data.data(), data.size() * sizeof(float));
	return 0;
}
