# fs3d against the FP64 reference solver

`check.py` runs `build/fs3d` on density cells and compares it with `reference/jax/finite_strain.py`
(FP64 JAX, validated against homo3d's small-strain moduli and an independent DOLFINx code). Both solvers use
E = 10, nu = 0.3, the neo-Hookean energy with the void interpolation of Wang et al. (beta 500, rho0 0.01) and the
stiffness scale 1e-9 + rho^3, rho being the physical density stored in the `.npy` cell (shape `(n, n, n)`, axis
order x, y, z; n = 16, 32, 64, ...).

| item | fs3d | reference | compared |
|---|---|---|---|
| 1 | `-mode toec` (`moduli.txt`) | `C_linear` | C^H, relative Frobenius error (target 1e-4) |
| 2 | `-mode path -path L1/L2 -crit 1` | `Mixed` (uniaxial stress) | Pbar_11(eps) relative, nu_12(eps) absolute (target 1e-3); Newton iterations and seconds per step |
| 3 | `-mode fdcheck -obj P11 -fd_out` | `Mixed.grad_rho` | dPbar_11/drho at eps = 0.1: fs3d adjoint against fs3d central differences and against the reference adjoint (relative L2) |
| 4 | `moduli.txt` (C^H and T) | `Cell` at Fbar = I + tH | W''' = T[h,h,h] + 3 C[h, H^T H] along four strain directions against the second derivative of <Pbar(I + tH), H> |
| 5 | `-mode path -path L2 -crit 1` | LOBPCG on D^-1/2 H D^-1/2 | lambda_min(eps)/lambda_min(0) and the first load point with lambda_min <= 0 |

```shell
conda activate homo3d-jax   # JAX FP64
python -I reference/fs3d_check/make_cells.py runs/cells 16 32          # rod lattice and Schwarz P cells
./build/vdb2npy <homo3d result>/rho runs/cells/npr32.npy               # a homo3d cell
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python -I reference/fs3d_check/check.py \
    --cells runs/cells/rod16.npy runs/cells/bulk32.npy --out runs/check --items 1 2 3 4
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python -I reference/fs3d_check/check.py \
    --cells runs/cells/npr32.npy --out runs/check --items 1 2 5 --de 0.01 --emax 0.2 --emin5 -0.16
```

Options: `--prec fp16|fp32|adaptive` (fs3d coarse stencils, default adaptive), `--emax`, `--de` (item 2 path),
`--fd_h`, `--fd_n` (item 3 finite differences), `--delta` (item 4 reference step), `--emin5`, `--de5` (item 5),
`-v` prints the fs3d command lines. Reference results are cached in `<out>/<cell>/oracle_*` (delete to recompute);
fs3d outputs go to `<out>/<cell>/<mode>/`. The summary table is printed and appended to `<out>/summary.csv`; the exit
code is 1 if any item fails.

The reference path solves at 32^3 take about 10 s per load point (FP64); fs3d takes milliseconds to a few tenths
of a second per load point at this size.
