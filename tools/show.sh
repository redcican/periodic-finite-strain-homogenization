#!/usr/bin/env bash
# Usage: tools/show.sh <result_dir> [iso=0.5] [tile=2]
set -e
T="$(cd "$(dirname "$0")" && pwd)"
eval "$(conda shell.bash hook)" && conda activate homo3d
for f in rho density initRho; do [ -f "$1/$f" ] && "${VDB2NPY:-$T/../build/vdb2npy}" "$1/$f" "$1/$f.npy" > /dev/null; done
python -I "$T/visualize.py" "$@"
