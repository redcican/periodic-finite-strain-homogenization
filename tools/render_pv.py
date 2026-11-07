"""High-resolution PyVista renders of a homo3d result.

Usage (in the homo3d-viz env):
  python render_pv.py <result_dir> [--field rho] [--iso 0.5] [--tile 3] [--upsample 2]
                      [--res 3000] [--views cell,cut,tile] [--color "#4a90d9"] [--transparent]
                      [--zoom 0.92] [--azimuth 0]

Reads <result_dir>/<field>.npy; if missing, converts <result_dir>/<field> (OpenVDB) with vdb2npy.
Writes <result_dir>/pv_<view>.png for each view, plus pv_cell.stl / pv_cell.ply meshes.
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("result_dir", type=Path)
ap.add_argument("--field", default="rho", help="rho, density or initRho")
ap.add_argument("--iso", type=float, default=0.5, help="density level of the solid surface")
ap.add_argument("--tile", type=int, default=3, help="cells per side in the tiling view")
ap.add_argument("--upsample", type=int, default=2, help="smooth upsampling factor before contouring")
ap.add_argument("--res", type=int, default=3000, help="image width/height in pixels")
ap.add_argument("--views", default="cell,cut,tile")
ap.add_argument("--color", default="#4a90d9")
ap.add_argument("--transparent", action="store_true", help="transparent background")
ap.add_argument("--zoom", type=float, default=0.92, help=">1 zooms in, <1 adds margin")
ap.add_argument("--azimuth", type=float, default=0.0, help="extra camera rotation in degrees")
args = ap.parse_args()

d = args.result_dir
npy = d / f"{args.field}.npy"
if not npy.exists():
    subprocess.run([os.environ.get("VDB2NPY", str(Path(__file__).resolve().parents[1] / "build" / "vdb2npy")), str(d / args.field), str(npy)], check=True)
rho = np.load(npy).astype(np.float32)
n = rho.shape[0]


def upsample(vol):
    """Periodic cubic upsampling, so the surface stays smooth across cell faces."""
    if args.upsample <= 1:
        return vol
    k = args.upsample
    pad = 3
    v = np.pad(vol, pad, mode="wrap")
    v = ndimage.zoom(v, k, order=3)
    return v[pad * k: -pad * k, pad * k: -pad * k, pad * k: -pad * k]


def surface(vol, ncell=1):
    """Iso-surface of vol, scaled so one unit cell has side length 1. Zero padding caps open faces."""
    v = np.pad(vol, 1, mode="constant", constant_values=0)
    h = ncell / vol.shape[0]
    grid = pv.ImageData(dimensions=v.shape, spacing=(h, h, h), origin=(-h, -h, -h))
    grid.point_data["rho"] = v.ravel(order="F")
    mesh = grid.contour([args.iso], scalars="rho", method="flying_edges")
    mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.05)
    return mesh.compute_normals(auto_orient_normals=True, split_vertices=False)


def render(mesh, name, show_box):
    p = pv.Plotter(off_screen=True, window_size=(args.res, args.res))
    p.set_background("white")
    p.add_mesh(mesh, color=args.color, smooth_shading=True, specular=0.4, specular_power=20,
               ambient=0.15, diffuse=0.85)
    if show_box:
        p.add_mesh(pv.Box(bounds=(0, 1, 0, 1, 0, 1)).extract_all_edges(), color="#888888", line_width=2)
    p.enable_anti_aliasing("ssaa")
    p.view_isometric()
    p.camera.azimuth = args.azimuth
    p.reset_camera()
    p.camera.zoom(args.zoom)
    out = d / f"pv_{name}.png"
    p.screenshot(str(out), transparent_background=args.transparent)
    p.close()
    print("wrote", out)


fine = upsample(rho)
views = args.views.split(",")

cell = surface(fine)
cell.save(d / "pv_cell.stl")
cell.save(d / "pv_cell.ply")
print(f"cell mesh: {cell.n_points} points, {cell.n_cells} triangles -> pv_cell.stl / pv_cell.ply")

if "cell" in views:
    render(cell, "cell", show_box=True)
if "cut" in views:
    half = fine.copy()
    half[:, half.shape[1] // 2:, :] = 0  # remove the y > 1/2 half to show the inside
    render(surface(half), "cut", show_box=True)
if "tile" in views:
    render(surface(np.tile(fine, (args.tile,) * 3), ncell=args.tile), "tile", show_box=False)
