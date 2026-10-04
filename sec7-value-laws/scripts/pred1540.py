"""Predictions for 1540 (NPMLE 3D shell), from the 1523 ball fits; no shell has been solved."""

import glob, os, re, math, json
import numpy as np

SP = r"scratch"
os.chdir(r"centre_scripts")
D = -0.304944111846
G = {}
for f in sorted(glob.glob("q848_R*.log")):
    for line in open(f):
        m = re.match(r"R ([0-9.]+): rings (\d+).*?; (NOT ACCEPTED|ACCEPTED);.*Gt ([-+0-9.]+);", line)
        if m and m.group(3) == "ACCEPTED":
            G.setdefault(float(m.group(1)), float(m.group(4)))
Rs = np.array(sorted(G))
Gt = np.array([G[r] for r in Rs])
FIT = {}
for lo in (15, 20):
    s = Rs >= lo
    R = Rs[s]
    y = Gt[s] - 4 * math.pi * R**2 * D
    X = np.c_[R, np.ones_like(R), 1 / R, 1 / R**2]
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    FIT[lo] = (c[0] / (8 * math.pi), c[1], c[2], c[3])
    print(f"ball fit R >= {lo}: a3 {FIT[lo][0]:.7f} E0_3 {c[1]:.5f} E1 {c[2]:.4f} E2 {c[3]:.3f}")
d0 = {5.0: 2.2761e-4, 7.5: -8.5998e-5, 9.0: -1.2276e-5}


def P0(r1, r2, a3, E0, E1, E2):
    return (
        4 * math.pi * D * (r1**2 + r2**2)
        + 8 * math.pi * a3 * (r2 - r1)
        + 2 * E0
        + E1 * (1 / r2 - 1 / r1)
        + E2 * (1 / r1**2 + 1 / r2**2)
    )


out = []
for r1, r2 in [(12, 38), (14, 40), (16, 40), (20, 30), (20, 35), (20, 38)]:
    p = [P0(r1, r2, *FIT[lo]) for lo in (15, 20)]
    c = sum(p) / 2
    base = 3 * abs(p[0] - p[1]) + 20 * (1 / r1**3 + 1 / r2**3)
    E0 = sum(FIT[lo][1] for lo in (15, 20)) / 2
    E1 = sum(FIT[lo][2] for lo in (15, 20)) / 2
    W = r2 - r1
    Rm = (r1 + r2) / 2
    cpl = 8 * math.pi * Rm**2 * d0[W / 2] if W / 2 in d0 else 0.0
    noflip = c - E1 * (1 / r2 - 1 / r1) + E1 * (1 / r2 + 1 / r1)
    o = {
        "R1": r1,
        "R2": r2,
        "W": W,
        "P0": c,
        "base": base,
        "centre_alt": c - 2 * E0,
        "one_E0_alt": c - E0,
        "noflip_alt": noflip,
        "coupling": cpl,
        "P1": c + cpl,
        "band1": base + 0.1 * abs(cpl),
    }
    out.append(o)
    print(
        f"({r1}, {r2}) W {W}: P0 {c:.6f} +-{base:.2e}  [centre alt {c - 2*E0:.6f}; one E0 {c - E0:.6f}; no E1 flip {noflip:.6f}]"
        f"  coupling {cpl:+.4e}  P1 {c + cpl:.6f} +-{base + 0.1*abs(cpl):.2e}"
    )
json.dump(out, open(os.path.join(SP, "pred1540.json"), "w"), indent=1)
