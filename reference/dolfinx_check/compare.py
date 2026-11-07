"""Compare the JAX reference solver with the DOLFINx implementation (bar < 1e-3 relative).

Per case and step: relative error of Pbar, max |dP_ij| / max |P_ij| (all 9 components), and for the mixed-control
case the error of the free Fbar components (|dF22|, |dF33|). Usage: python -I compare.py <dir>; writes compare.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

STEPS = 5          # load steps per case (cases.path)
MAXNEWTON = 50     # reference_side.py: a JAX step without a convergence flag converged if it used fewer Newton iterations

d = Path(sys.argv[1])
a, b = json.load(open(d / "reference.json")), json.load(open(d / "dolfinx.json"))
# the comparison is valid only on matching, complete, finite and converged paths
problems = []
if set(a) != set(b):
    problems.append(f"case sets differ: only JAX {sorted(set(a) - set(b))}, only DOLFINx {sorted(set(b) - set(a))}")
for k in sorted(set(a) & set(b)):
    if not (len(a[k]) == len(b[k]) == STEPS):
        problems.append(f"{k}: {len(a[k])} JAX and {len(b[k])} DOLFINx steps, expected {STEPS}")
    for side, rows in (("JAX", a[k]), ("DOLFINx", b[k])):
        for i, r in enumerate(rows):
            if not (np.all(np.isfinite(r["P"])) and np.all(np.isfinite(r["F"]))):
                problems.append(f"{k} step {i + 1}: non-finite {side} output")
            ok = r.get("converged", r.get("newton", MAXNEWTON) < MAXNEWTON) if side == "JAX" else r.get("converged", True)
            if not ok:
                problems.append(f"{k} step {i + 1}: {side} not converged")
if problems:
    print("FAIL: incomplete comparison\n  " + "\n  ".join(problems))
    sys.exit(1)
out, worst = {}, 0.0
print(f"{'case':18s} {'max rel dPbar':>14s} {'max |dF free|':>14s}  final P11 (JAX)   final P11 (DOLFINx)")
for k in sorted(a):
    eP, eF = [], []
    for ra, rb in zip(a[k], b[k]):
        Pa, Pb = np.array(ra["P"]), np.array(rb["P"])
        eP.append(float(np.abs(Pa - Pb).max() / np.abs(Pa).max()))
        eF.append(float(np.abs(np.array(ra["F"]) - np.array(rb["F"])).max()))
    out[k] = dict(rel_P=eP, abs_F=eF, steps=len(eP), converged=[r.get("converged", True) for r in b[k]])
    worst = max(worst, max(eP))
    print(f"{k:18s} {max(eP):14.2e} {max(eF):14.2e}  {a[k][-1]['P'][0][0]:+.10e}  {b[k][-1]['P'][0][0]:+.10e}")
print(f"\nworst relative Pbar difference over all cases and steps: {worst:.2e} (bar 1e-3: {'PASS' if worst < 1e-3 else 'FAIL'})")
json.dump(dict(cases=out, worst=worst), open(d / "compare.json", "w"), indent=1)
