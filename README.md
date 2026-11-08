## Periodic Finite-Strain Homogenization of Voxel Cells

This repository contains the code for the homogenization and design of periodic cells on voxel grids at small and finite strain:

* a GPU solver for inverse homogenization of periodic cells (CUDA C++), extended from the open-source solver of Zhang et al. (2023);
* a GPU solver for periodic finite-strain homogenization (CUDA C++, `fs3d`): a matrix-free mixed-precision Newton-Krylov multigrid method with stability criteria;
* a double-precision reference solver for periodic finite-strain homogenization (JAX), with uniaxial-stress paths, the Poisson's ratio along a load path, stability criteria and a Bloch-wave analysis;
* an independent finite element implementation in DOLFINx for cross-checking the reference solver;
* tools for converting and rendering density fields.

### Repository layout

| Path | Content |
|------|---------|
| `*.cu`, `*.cpp`, `AutoDiff/`, `culib/`, `gmem/`, `homogenization/`, `optimization/`, `voxelIO/` | GPU solver |
| `fiber_p1.cuh` | custom driver of the GPU solver: projection onto a stiffness target and re-evaluation of $C^H$ |
| `finite_strain/` | finite-strain GPU solver `fs3d` |
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

If the conda environment is activated, `cmake` will automatically checkout the dependencies in this environment. The build produces `build/homo3d` (the GPU solver), `build/fs3d` (the finite-strain GPU solver) and `build/vdb2npy` (the converter used by the Python scripts; set `VDB2NPY=<path>` to use another one).



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

#### Finite-strain GPU solver

`fs3d` follows a load path of a cell with the energy of a compressible neo-Hookean solid and the void interpolation of Wang et al. (2014), $s=\rho_{\min}+\bar\rho^{3}$. The density is the physical density $\bar\rho\in[0,1]$, read from an OpenVDB file (e.g. the `rho` output of `homo3d`, with `-reso`) or a `.npy` array of shape $[n,n,n]$; $n$ must be $8\cdot2^L$ (16, 32, 64, 128, 256).

* Newton iterations with the Eisenstat-Walker forcing terms, a backtracking line search on the cell energy (with a residual merit function beyond an instability), mixed macroscopic control through a bordered Hessian; an iteration has converged at $\|\mathsf g\|\le10^{-6}\|\mathsf g_0\|$ (`-newton_tol`) or at the rounding level $64\,u_{32}\|\mathsf f_e\|$ of the FP32 residual, $\|\mathsf f_e\|$ being the norm of the element forces it is assembled from;
* PCG preconditioned by a multigrid V-cycle; MINRES with a shifted preconditioner once the tangent is indefinite;
* matrix-free FP32 fine level from a deformation-gradient cache, Galerkin coarse operators stored in FP16 in the symmetric scaling $Z^{-1/2}KZ^{-1/2}$, FP64 dense solve on the $8^3$ grid;
* step control: a failed step is repeated with FP32 coarse stencils, then divided into substeps (at most three divisions); `-prec adaptive` switches to FP32 stencils once the extrapolated smallest eigenvalue falls below 10 % of its initial value;
* stability criteria: smallest eigenvalue of the scaled tangent of the cell (LOBPCG), of a $2\times2\times2$ supercell carrying the tiled state, and rank-one ellipticity of the effective tangent. Beyond the first load point with a negative eigenvalue the path continues on the fundamental branch (MINRES on the indefinite tangent), which is then a saddle.

Options of the path mode:

* `-in`, `-reso`: density field and resolution.
* `-path`: `L1` uniaxial tension, `L2` uniaxial compression (both with free lateral stretches $\bar F_{22}$, $\bar F_{33}$), `L3` simple shear, `L4` volumetric expansion. Default `L1`.
* `-steps`, `-inc`: number of load steps and increment (default 10 steps of 0.02, 0.01 for `L4`).
* `-prec`: coarse stencil precision `fp16`, `fp32` or `adaptive` (default).
* `-crit`: stability criteria to evaluate, any of `1` (cell), `2` (supercell), `3` (rank-one); `-crit_every` evaluates criterion 1 every k-th load point. Default `1`.
* `-E`, `-nu`: base material, default 10 and 0.3.
* `-newton_tol`: relative Newton tolerance, default `1e-6`.
* `-prefix`: output path suffixed with `/`; `-verbose 2` prints every Newton iteration.

```shell
./build/fs3d -mode path -in out/rho -reso 128 -path L2 -crit 123 -prefix out/
```

The load points are written to `<prefix>/path_<L>.csv` with the columns `k, t, eps, F11, F22, F33, F12, P11 .. P33` (row-major $\bar P$), `nu12, nu13` (secant Poisson's ratios $-(\bar F_{22}-1)/\varepsilon$, $-(\bar F_{33}-1)/\varepsilon$), `newton, krylov, substeps, failures, minres, prec`, the stability measures `lambda_cell, lambda_supercell, ellipticity` (NaN where not evaluated), `void_distorted` and `seconds` (wall time of the load step). `-nostop_distortion` continues the path when void voxels invert.

Small-strain moduli and third-order elastic constants (`-mode toec`): six corrector solves at $\bar F=I$ give $C^H$ and, by the $2n+1$ rule, the 56 constants $T_{IJK}$ of $\bar W=\tfrac12 C^H[\bar E,\bar E]+\tfrac16 T[\bar E,\bar E,\bar E]$ (Voigt order 11, 22, 33, 23, 13, 12 with engineering shears), written to `<prefix>/moduli.txt`:

```shell
./build/fs3d -mode toec -in out/rho -reso 128 -prefix out/
```

Adjoint sensitivities (`-mode fdcheck`): the gradient of $\nu_{12}$ (`-obj nu12`) or $\bar P_{11}$ (`-obj P11`) at load point `-fd_k` of a path with respect to the voxel densities, from one bordered adjoint solve, against central differences on `-fd_n` voxels with step `-fd_h`:

```shell
./build/fs3d -mode fdcheck -in rho32.npy -path L1 -steps 5 -obj P11 -prec fp32 -fd_n 6 -fd_out grad.npy
```

`-fd_out` writes the adjoint gradient $\partial J/\partial\bar\rho$ as a float64 `.npy` array of shape $[n,n,n]$ (same layout as the input); `-fd_n 0` skips the finite differences. The central differences of the FP32 solver resolve single-voxel derivatives only with a large step (`-fd_h 0.1` or more); the reference check (`reference/fs3d_check`) compares the whole gradient with the FP64 adjoint instead.

Finite-strain design (`-mode design`): the design variables live on one forty-eighth of the cell and are mapped to the full cell by the reflections of the cube, filtered with a periodic hat filter of radius $n/32$ voxels (`-filter_r`) and projected with a Heaviside function whose sharpness doubles every 40 iterations up to 32. Each iteration evaluates $C^H$ from the six correctors, path `L1` with the load steps as load points (criterion 1 every second step) and one adjoint per load point, and updates the design by MMA; a design whose load path fails is rejected and the step repeated with half the move limit.

* `-prog nu`: bound formulation $\min z$ subject to $(\nu_{12}(\varepsilon_k)-\nu^*(\varepsilon_k))^2\le z$, with $\nu^*(\varepsilon)$ = `-nustar` + `-nuslope` $\varepsilon$.
* `-prog stiff`: least-squares fit of $\bar P_{11}$ to $E^*(\varepsilon+(S-1)\varepsilon^2/(2\varepsilon_{\max}))$ with the stiffening ratio `-S`.
* constraints $V\le V^*$ and $E^H\ge E^*$; $V^*$ and $E^*$ come from `-ref` (resampled to `-reso`, `-ref_reso` for OpenVDB input), from the initial design `-in`, or from `-Vstar` and `-Estar`.
* `-iters`, `-move`, `-save_every`: iterations (default 240), MMA move limit (0.1), output interval.

```shell
./build/fs3d -mode design -prog nu -nustar -0.3 -in out/rho -reso 128 -prefix design/
```

The run writes `<prefix>/design.csv` (objective, largest deviation, $V/V^*$, $E^H/E^*$ per iteration), the densities `<prefix>/rho_<iter>.vdb` and `<prefix>/rho_design.vdb`, and, for the design thresholded at 0.5, `<prefix>/rho_thresholded.vdb` with its path `L1` in increments of 0.005 and all three stability criteria (`<prefix>/final_L1.csv`).

Diagnostics: `FS_DEBUG=1` reports which stage of a preconditioner build is not positive definite; `-mode mgcheck -in <cell>` compares the first coarse Galerkin stencil with $P^T K P$ applied to unit vectors. `reference/fs3d_check/check.py` runs `fs3d` against the reference solver below (see its README).

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
