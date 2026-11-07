"""JAX reference side of the DOLFINx cross-check. Usage (reference env): python -I reference_side.py <dir>
Writes <dir>/reference.json: per field and load, the full Fbar and Pbar after every step."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jax"))
import cases as C  # noqa: E402
import finite_strain as O  # noqa: E402

d = Path(sys.argv[1])
res = {}
for name, f in C.FIELDS.items():
    rho = np.load(d / f"{name}.npy", allow_pickle=False)
    cell = O.Cell(rho, E=C.E, nu=C.NU, material="neohooke", wls=f["wls"])
    for ln, L in C.loads().items():
        rows = []
        if L["free"] is None:
            w = None
            for Fb in C.path(L["F"]):
                w, log = cell.solve(Fb, w0=w, rtol=1e-12, maxnewton=50, cg_tol=1e-12, cg_maxit=20000)
                P = np.asarray(cell.mean_stress(w, Fb))
                rows.append(dict(F=np.asarray(Fb).tolist(), P=P.tolist(), newton=log["newton"], converged=bool(log["converged"])))
        else:
            mix = O.Mixed(cell, L["free"])
            z = None
            for Fb in C.path(L["F"]):
                w, F, log = mix.solve(Fb, z0=z, rtol=1e-12, maxnewton=50, cg_tol=1e-12)
                z = log["z"]
                P = np.asarray(cell.mean_stress(w, F))
                rows.append(dict(F=np.asarray(F).tolist(), P=P.tolist(), newton=log["newton"], converged=bool(log.get("converged", log["newton"] < 50))))
        res[f"{name}/{ln}"] = rows
        print(f"{name}/{ln}: P11 {rows[-1]['P'][0][0]:+.6e}  F22 {rows[-1]['F'][1][1]:.6f}", flush=True)
json.dump(res, open(d / "reference.json", "w"), indent=1)
