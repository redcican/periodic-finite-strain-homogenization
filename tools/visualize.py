"""Visualize a homo3d result folder.

Usage: python visualize.py <result_dir> [iso=0.5] [tile=2]

Reads <result_dir>/rho.npy (made by vdb2npy) and <result_dir>/C, then writes:
  rho_slices.png   - three mid-plane slices of the density
  rho_iso.png      - 3D surface of one cell and of a tile x tile x tile array
  rho_cell.stl     - surface mesh of one cell (open in ParaView, Blender, MeshLab...)
"""
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

d = Path(sys.argv[1])
iso = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
tile = int(sys.argv[3]) if len(sys.argv) > 3 else 2

rho = np.load(d / "rho.npy")
n = rho.shape[0]
print(f"rho: shape {rho.shape}, min {rho.min():.3f}, max {rho.max():.3f}, mean (volume) {rho.mean():.3f}")
print(f"voxels > {iso}: {(rho > iso).mean():.3f}")

C = np.array(struct.unpack("36f", (d / "C").read_bytes())).reshape(6, 6)
np.set_printoptions(precision=4, suppress=True)
print("C (Voigt 6x6):\n", C)
K = (C[:3, :3].sum()) / 9
print(f"bulk modulus K = {K:.4f}")

# --- slices
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for i, (name, sl) in enumerate([("x = mid", rho[n // 2]), ("y = mid", rho[:, n // 2]), ("z = mid", rho[:, :, n // 2])]):
    im = ax[i].imshow(sl.T, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax[i].set_title(name)
fig.colorbar(im, ax=ax, shrink=0.8, label="density")
fig.savefig(d / "rho_slices.png", dpi=120)


# --- iso-surfaces. Pad with the periodic neighbour so the surface closes across cell faces.
def surface(vol):
    v = np.pad(vol, 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(v, iso)
    return verts - 1, faces


def write_stl(path, verts, faces):
    tri = verts[faces].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as f:
        f.write(b"homo3d".ljust(80, b"\0"))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())


def draw(ax, verts, faces, title):
    mesh = Poly3DCollection(verts[faces], linewidth=0, alpha=1.0)
    nrm = np.cross(verts[faces][:, 1] - verts[faces][:, 0], verts[faces][:, 2] - verts[faces][:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    shade = 0.3 + 0.7 * np.abs(nrm @ np.array([0.4, 0.5, 0.75]) / np.linalg.norm([0.4, 0.5, 0.75]))
    mesh.set_facecolor(np.c_[0.35 * shade + 0.1, 0.6 * shade + 0.1, 0.9 * shade + 0.1])
    ax.add_collection3d(mesh)
    m = verts.max()
    ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_zlim(0, m)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(25, 35)
    ax.set_title(title)


v1, f1 = surface(rho)
write_stl(d / "rho_cell.stl", v1, f1)
vt, ft = surface(np.tile(rho, (tile, tile, tile)))

half = rho.copy()
half[:, n // 2:, :] = 0  # cut the cell in half to show the inside
vh, fh = surface(half)

fig = plt.figure(figsize=(18, 6))
draw(fig.add_subplot(131, projection="3d"), v1, f1, f"one cell ({n}^3), iso = {iso}")
draw(fig.add_subplot(132, projection="3d"), vh, fh, "cutaway (half cell)")
draw(fig.add_subplot(133, projection="3d"), vt, ft, f"{tile}x{tile}x{tile} tiling")
fig.tight_layout()
fig.savefig(d / "rho_iso.png", dpi=120)
print("wrote", d / "rho_slices.png", d / "rho_iso.png", d / "rho_cell.stl")
