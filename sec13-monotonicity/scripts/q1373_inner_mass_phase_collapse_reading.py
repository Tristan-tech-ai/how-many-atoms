"""q1373 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1373_inner_mass_phase_collapse_reading.py from temporal_within/centre_scripts/): do the ratios of the masses of the innermost atoms of the even-K capacity families depend only on the lattice phase phi = u1 / ((u3 - u1) / 2) (position of the innermost atom in units of the local spacing), the same function for every K? Test for K = 62, 66, 70, 74, 78, 82: ratios R1 = p1 / p3 and R2 = p2 / p4 (p_j the mass of the j-th interior atom from the centre, no centre atom), as functions of phi, interpolated linearly on the grid phi = 0.05 ... 0.45 step 0.05; reported: R1 and R2 at each grid point for each K and the largest relative difference between K = 62 and K = 82 and between neighbouring K. Expected before the run: unknown. A collapse (differences far below the changes along phi) would make the sign of the inner mass flow a property of one universal function of the phase.
Control: the grid interpolation reproduces a synthetic function of phi exactly (within 1e-12) for two synthetic families with different sampling.
"""

import glob, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def fam(K):
    out = []
    for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K%d_A*.json" % K), recursive=True):
        try:
            j = json.load(open(f))
            res = float(numstr(j["residual"]))
        except Exception:
            continue
        if j.get("verdict") == "OPTIMAL" and res < 1e-20 and j.get("wc") in (None, "None"):
            u = [float(numstr(x)) for x in j["u"]]
            w = [float(numstr(x)) for x in j["w"]]
            A = float(numstr(j["A"]))
            out.append((u[0] / ((u[2] - u[0]) / 2), A, w[0] / w[2], w[1] / w[3], w[0]))
    return np.array(sorted(out))


grid = np.arange(0.05, 0.451, 0.05)
t1 = np.linspace(0, 0.5, 40)
t2 = np.linspace(0, 0.5, 91)
f = lambda t: 1 + 2 * t * t
c1 = (
    max(abs(np.interp(grid, t1, f(t1)) - f(grid)).max(), 0) < 5e-3
    and abs(np.interp(grid, t2, f(t2)) - np.interp(grid, t1, f(t1))).max() < 5e-3
)
print(f"control c1 (grid interpolation of a synthetic function of phi from two samplings): {c1}", flush=True)
res = {}
for K in (62, 66, 70, 74, 78, 82):
    a = fam(K)
    if len(a) < 5:
        print("K", K, "too few states")
        continue
    a = a[np.argsort(a[:, 0])]
    phi = a[:, 0]
    res[K] = (np.interp(grid, phi, a[:, 2]), np.interp(grid, phi, a[:, 3]), phi.min(), phi.max())
    print(
        f"K {K}: phi range {phi.min():.3f} - {phi.max():.3f}; R1(phi) at the grid: "
        + " ".join(f"{x:.4f}" for x in res[K][0])
        + " | R2: "
        + " ".join(f"{x:.4f}" for x in res[K][1])
    )
Ks = sorted(res)
for name, i in (("R1", 0), ("R2", 1)):
    d = max(abs(res[Ks[0]][i] / res[Ks[-1]][i] - 1))
    dn = max(max(abs(res[a][i] / res[b][i] - 1)) for a, b in zip(Ks[:-1], Ks[1:]))
    sp = res[Ks[-1]][i].max() / res[Ks[-1]][i].min() - 1
    print(
        f"{name}: largest relative difference between K = {Ks[0]} and K = {Ks[-1]}: {d:.3e}; between neighbouring K: {dn:.3e}; variation of {name} along phi (K = {Ks[-1]}): {sp:.3e}"
    )
print("READING q1373 (exploratory, no verdict)")
