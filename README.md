## Periodic Finite-Strain Homogenization of Voxel Cells

This repository contains the code for the homogenization and design of periodic cells on voxel grids at small and finite strain:

* a GPU solver for inverse homogenization of periodic cells (CUDA C++), extended from the open-source solver of Zhang et al. (2023);
* a double-precision reference solver for periodic finite-strain homogenization (JAX), with uniaxial-stress paths, the Poisson's ratio along a load path, stability criteria and a Bloch-wave analysis;
* an independent finite element implementation in DOLFINx for cross-checking the reference solver;
* tools for converting and rendering density fields.

### Repository layout

| Path | Content |
|------|---------|
| `*.cu`, `*.cpp`, `AutoDiff/`, `culib/`, `gmem/`, `homogenization/`, `optimization/`, `voxelIO/` | GPU solver |
| `fiber_p1.cuh` | custom driver of the GPU solver: projection onto a stiffness target and re-evaluation of $C^H$ |
| `reference/jax/` | FP64 finite-strain reference solver and its scripts |
| `reference/dolfinx_check/` | DOLFINx cross-check of the reference solver |
| `tools/` | `vdb2npy` (OpenVDB to `.npy`), PyVista and matplotlib renders |



### Dependency

GPU solver:

* CUDA 11.8
* OpenVDB 8.2 (downloaded and built by CMake)
* gflags
* glm
* Eigen3

Reference solver: JAX with FP64 and CUDA 12, NumPy, SciPy.

DOLFINx cross-check: DOLFINx 0.11, dolfinx_mpc 0.11, PETSc.

We have packed the dependencies into [conda](https://docs.conda.io/en/latest/miniconda.html) environments. The environment of the GPU solver includes the CUDA toolkit and the compilers; create it by:

```bash
conda env create -f environment.yml
```

Then you activate it by:

```bash
conda activate homo3d
```

The Python parts have their own environments:

```bash
conda env create -f reference/environment.yml                 # fs-reference: JAX reference solver
conda env create -f reference/dolfinx_check/environment.yml   # fs-dolfinx: DOLFINx cross-check
conda env create -f tools/environment.yml                     # fs-viz: PyVista renders
```



### Compilation

After the dependency is installed, the code can be compiled using cmake:

```shell
conda activate homo3d
mkdir build
cd build
cmake .. -DCMAKE_CUDA_HOST_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc
make -j4
```

If the conda environment is activated, `cmake` will automatically checkout the dependencies in this environment. The build produces `build/homo3d` (the GPU solver) and `build/vdb2npy` (the converter used by the Python scripts; set `VDB2NPY=<path>` to use another one).



### Usage

#### GPU solver

* `-reso` : the resolution of the discretized domain, e.g., `-reso 128` defines an $128\times128\times128$ domain. Default value is 128.
* `-obj` : the objective to be optimized, options are `bulk`,`shear`,`npr` and `custom`, which optimizes the bulk modulus, shear modulus, Poisson's ratio and custom objective respectively. Default is `bulk`
* `-init`: the method for initializing the density field, the common and default option is `randc`, which set the initialization via a set of trigonometric function basis. You can set this option to `manual` to set the initialization from a OpenVDB file.
* `-sym`: symmetry requirement on the structure, only `reflect3`, `reflect6` and `rotate3` are supported. Default is `reflect6`.
* `-vol`: volume ratio for material usage ranging from $(0,1)$, default is `0.3`
* `-E`: Young's modulus of base material. Default is `1e1` (recommended, since FP16 has a limited range).
* `-mu`: Poisson's ratio of base material. Default is `0.3`
* `-prefix`: output path suffixed with `/`
* `-in`: variable input determined by other options, e.g., a OpenVDB file path when the argument of `-init` is `manual`.
* `-N`: maximal iteration number, default is `300`.
* `-relthres`: the relative residual tolerance on FEM equation, default is `1e-2`. Use `1e-3` to `1e-4` for accurate $C^H$; smaller values are not reached by the mixed-precision multigrid.

Optimizing the bulk modulus:

```shell
./build/homo3d -reso 128 -obj bulk -init randc -sym reflect6 -vol 0.3 -mu 0.3 -prefix out/
```

The optimized density field is stored in `<prefix>/rho` in OpenVDB format, and the elastic matrix in `<prefix>/C` as an array of 36 floats.

On GPUs without concurrent managed-memory access (WSL2, Windows), large buffers are placed in device memory, which is 5 to 10 times faster there; set `HOMO_MANAGED=1` or `HOMO_MANAGED=0` to force either choice.

The custom driver runs through `-obj custom` with `HOMO_CUSTOM=p1`; `P1_MODE` selects the task (see the header of `fiber_p1.cuh` for all options):

```shell
# C^H of a saved density with stiffness (1e-9 + rho^3) C0, written to <prefix>/evalvdb_C.txt
HOMO_CUSTOM=p1 P1_MODE=evalvdb P1_FROM=out/rho ./build/homo3d -obj custom -reso 64 -relthres 1e-4 -prefix out/
# random density fields and their C^H, used by reference/jax/validate_linear.py
HOMO_CUSTOM=p1 P1_MODE=evalfield ./build/homo3d -obj custom -reso 16 -relthres 1e-4 -prefix check16/
```

#### Finite-strain reference solver

`reference/jax/finite_strain.py` minimizes the cell energy of a neo-Hookean (or linear) solid on an $n^3$ periodic grid of trilinear hexahedra with $2\times2\times2$ Gauss points, in FP64, by Newton-CG with a backtracking line search. `Cell` prescribes the full macroscopic deformation gradient, `Mixed` prescribes some components and leaves the others free (e.g. uniaxial stress), and `C_linear` returns the small-strain $C^H$. Void elements use the energy interpolation of Wang et al. (2014) with `wls=(500, 0.01)`.

```python
import sys
import numpy as np
sys.path.insert(0, "reference/jax")
import finite_strain as fs

rho = np.load("rho.npy")                    # physical density in [0, 1], shape (n, n, n)
cell = fs.Cell(rho, E=10.0, nu=0.3, material="neohooke", wls=(500, 0.01))
F = np.diag([1.1, 1.0, 1.0])
w, log = cell.solve(F, rtol=1e-10)
P = cell.mean_stress(w, F)                  # macroscopic first Piola-Kirchhoff stress
```

Scripts in `reference/jax/`, run in `fs-reference`; the header of each script documents its arguments and outputs:

* `poisson_drift.py <result dir>`: Poisson's ratio along uniaxial-stress paths in tension and compression, from a GPU solver result (`rho` is cubed as in the solver; pass `--phys` for a physical density); writes `drift_*.csv`.
* `stability_path.py <result dir>`: smallest eigenvalue of the diagonally scaled tangent along the path (cell and, with `--tile 2`, $2\times2\times2$ supercell).
* `bloch_scan.py <result dir>`: Bloch-wave stability for chosen wave vectors and strains.
* `path_check.py <density .npy>`: small-strain consistency, Newton and CG counts along uniaxial-strain paths, and the adjoint gradient against central differences.
* `validate_linear.py <dir>`: patch test and $C^H$ against the GPU solver (`P1_MODE=evalfield` output).
* `voxelize_truss.py <data dir> <out dir> <index> ...`: voxelizes trusses of the Zheng et al. (2023) library at a prescribed volume fraction.

#### DOLFINx cross-check

```shell
conda activate fs-reference
python -I reference/dolfinx_check/cases.py check/ <Zheng et al. dataset data dir>
python -I reference/dolfinx_check/reference_side.py check/
conda activate fs-dolfinx
python -I reference/dolfinx_check/dolfinx_side.py check/
python -I reference/dolfinx_check/compare.py check/
```

Twelve load cases (tension, compression, shear and uniaxial stress on random and lattice fields at $8^3$ and $16^3$) are solved by both codes; `compare.py` checks that all cases are complete and converged and reports the largest relative difference of the macroscopic stress.

#### Rendering

```shell
conda activate fs-viz
python -I tools/render_pv.py out/            # PyVista render of out/rho
tools/show.sh out/                           # matplotlib isosurface and C^H summary (homo3d env)
```



## Citation

This code extends the GPU solver of Zhang et al. If you use it in your academic research, please include the following citation

```
@ARTICLE{Zhang2023-ti,
  title    = "An optimized, easy-to-use, open-source {GPU} solver for
              large-scale inverse homogenization problems",
  author   = "Zhang, Di and Zhai, Xiaoya and Liu, Ligang and Fu, Xiao-Ming",
  journal  = "Structural and Multidisciplinary Optimization",
  volume   =  66,
  pages    =  "Article 207",
  month    =  sep,
  year     =  2023
}
```



## License

GNU General Public License v2.0, as the solver it extends (see `LICENSE`).
