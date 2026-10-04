"""Predictions for 1535 (3D shell capacity), written before any shell C value is read: reads only 'W' from the shell file."""

import json, math, os
import numpy as np

SP = r"scratch"
os.chdir(r"centre_scripts")
beta = 3.98160794709
d = 3
V3 = 4 * math.pi / 3
S3 = 4 * math.pi
h3 = 1.5 * math.log(2 * math.pi * math.e)
# 3D ball constants, strip fixed at beta/2: S(A) = beta/2 + s1/A + s2/A^2 + s3/A^3
S = {}
for f in ["q662_ball_d3.json", "q656_shells3d.json"]:
    for s in json.load(open(f))["states"]:
        A = round(float(s["A"]), 6)
        if "C" in s and A not in S:
            S[A] = (math.exp(s["C"] + h3) - V3 * A**3) / (S3 * A**2)
As = np.array(sorted(S))
y = np.array([S[a] for a in As])
FITS = {}
for lo in (8, 10, 12):
    m = As >= lo
    X = np.c_[1 / As[m], 1 / As[m] ** 2, 1 / As[m] ** 3]
    FITS[lo] = np.linalg.lstsq(X, y[m] - beta / 2, rcond=None)[0]
    print(
        f"ball d3 A >= {lo} ({m.sum()} states): s1 {FITS[lo][0]:.5f} s2 {FITS[lo][1]:.4f} s3 {FITS[lo][2]:.4f}"
    )
eps = json.load(open(os.path.join(SP, "eps1533.json")))


def P0(r1, r2, s1, s2, s3):
    return (
        V3 * (r2**3 - r1**3)
        + 2 * math.pi * beta * (r1**2 + r2**2)
        + 4 * math.pi * s1 * (r2 - r1)
        + 8 * math.pi * s2
        + 4 * math.pi * s3 * (1 / r2 - 1 / r1)
    )


out = []
a = 20
sh = json.load(open("q686_annulus_a20_d3_SEALED.json"))
assert float(sh["DIM"]) == 3.0 and float(sh["a"]) == 20.0
for k, s in enumerate(sh["states"]):
    W = round(float(s["W"]), 6)
    r1, r2 = a, a + W
    Rm = (r1 + r2) / 2
    p = [P0(r1, r2, *FITS[lo]) for lo in (8, 10, 12)]
    base = 3 * (max(p) - min(p)) + 4 * math.pi * (1 / r1**2 + 1 / r2**2)
    e = eps.get(str(W / 2))
    if e is None or not (e["res"] < 1e-10 and e["kkt"] < 1e-9):
        out.append({"k": k, "W": W, "note": "no accepted 1D state"})
        continue
    cpl = 4 * math.pi * Rm**2 * e["eps"]
    out.append(
        {
            "k": k,
            "W": W,
            "P0": float(p[1]),
            "base": float(base),
            "coupling": cpl,
            "P1": p[1] + cpl,
            "band1": base + 0.1 * abs(cpl),
            "decisive": bool(abs(cpl) > 3 * base),
        }
    )
json.dump(out, open(os.path.join(SP, "pred1535.json"), "w"), indent=1)
for o in out:
    if "P0" in o:
        print(
            f"W {o['W']:4.1f}: P0 {o['P0']:.5f} +-{o['base']:.2e}  coupling {o['coupling']:+.3e}  P1 {o['P1']:.5f} +-{o['band1']:.2e} {'DECISIVE' if o['decisive'] else ''}"
        )
print(sum(1 for o in out if o.get("decisive")), "decisive of", sum(1 for o in out if "P0" in o))
