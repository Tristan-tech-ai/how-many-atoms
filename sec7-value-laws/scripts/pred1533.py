"""Predictions for 1533 (written before any 2D C value is read): reads only 'W' from the 2D annulus files."""

import json, math, os

SP = r"scratch"
os.chdir(r"centre_scripts")
beta = 3.98160794709
FITS = {
    8: (-0.5116, -0.2884),
    10: (-0.5407, -0.0847),
    12: (-0.5465, -0.0409),
}  # (s2, s3) with s0 = beta, from q649/q650
eps = json.load(open(os.path.join(SP, "eps1533.json")))


def P0(r1, r2, s2, s3):
    return (
        math.pi * (r2**2 - r1**2)
        + beta * math.pi * (r1 + r2)
        + math.pi * s2 * (1 / r1 + 1 / r2)
        + math.pi * s3 * (1 / r2**2 - 1 / r1**2)
    )


out = []
for a in (10, 20):
    d = json.load(open(f"q686_annulus_a{a}_d2_SEALED.json"))
    for k, s in enumerate(d["states"]):
        W = float(s["W"])
        r1, r2 = a, a + W
        Rm = (r1 + r2) / 2
        p = [P0(r1, r2, *FITS[lo]) for lo in (8, 10, 12)]
        base = 3 * (max(p) - min(p)) + math.pi * (1 / r1**3 + 1 / r2**3)
        e = eps.get(str(round(W, 6) / 2))
        if e is None or not (e["res"] < 1e-10 and e["kkt"] < 1e-9):
            out.append({"a": a, "k": k, "W": W, "note": "no accepted 1D state"})
            continue
        cpl = 2 * math.pi * Rm * e["eps"]
        out.append(
            {
                "a": a,
                "k": k,
                "W": W,
                "P0": p[1],
                "base": base,
                "coupling": cpl,
                "P1": p[1] + cpl,
                "band1": base + 0.1 * abs(cpl),
                "decisive": abs(cpl) > 3 * base,
            }
        )
json.dump(out, open(os.path.join(SP, "pred1533.json"), "w"), indent=1)
for o in out:
    if "P0" in o:
        print(
            f"a {o['a']} W {o['W']:4.1f}: P0 {o['P0']:.6f} +-{o['base']:.1e}  coupling {o['coupling']:+.3e}  P1 {o['P1']:.6f} +-{o['band1']:.1e}  {'DECISIVE' if o['decisive'] else ''}"
        )
    else:
        print(o)
