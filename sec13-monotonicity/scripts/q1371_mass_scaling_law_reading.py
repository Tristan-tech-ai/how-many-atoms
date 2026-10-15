"""q1371 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1371_mass_scaling_law_reading.py from temporal_within/centre_scripts/): the masses of the edge-indexed atoms of the capacity-achieving input scale as 1 / (A + delta): p_j(A) (A + delta) does not depend on A inside a family, j counted from the edge (j = 0 the edge atom at A). Reason (stated before the run): the KKT conditions are invariant under the scaling of the whole mass measure (D changes by a constant, absorbed in C), the edge region is the half-line solution up to that scale, and the scale is fixed by the flat bulk density 1 / (2 (A + delta)). For one atom j and two states of one family, delta_j = -p_j / (dp_j/dA) - A.
Reading: for all consecutive same-family pairs of the stored states with K >= 60 and 0.002 < A2 - A1 < 0.2 (2285 pairs) and j = 0..12: the median delta_j, its spread (max - min) per j, the global value from j = 0 to 3, and the number of the atoms j for which delta_j is within 1e-6 and 1e-4 of the global value, for K = 62, 70, 80, 82. Expected before the run: a constant of order 2, the same for all families, exact for the outer atoms and drifting inward (the centre region is not edge-indexed).
Controls: a synthetic family p_j(A) = c_j / (A + 1.99) with noise 1e-12 returns 1.99 within 1e-6 for all j; a family with p_j(A) = c_j / (A + 1.99 + 0.001 j) shows a drift of 0.001 per atom (the check can fail).
"""

import glob, json, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def masses_from_edge(j):  # index 0 = edge atom, then the interior atoms inward, centre atom last if present
    w = [float(numstr(x)) for x in j["w"]]
    we = float(numstr(j["we"]))
    wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
    return np.array([we] + w[::-1] + ([wc] if wc is not None else []))


def deltas(A1, p1, A2, p2):
    Am = (A1 + A2) / 2
    pm = (p1 + p2) / 2
    dp = (p2 - p1) / (A2 - A1)
    return -pm / dp - Am


# controls
rng = np.random.default_rng(1)
c = np.sort(rng.uniform(0.5, 1.5, 15))[::-1]
A1, A2 = 40.0, 40.01
d_ok = deltas(
    A1,
    c / (A1 + 1.99) * (1 + 1e-12 * rng.standard_normal(15)),
    A2,
    c / (A2 + 1.99) * (1 + 1e-12 * rng.standard_normal(15)),
)
d_bad = deltas(A1, c / (A1 + 1.99 + 0.001 * np.arange(15)), A2, c / (A2 + 1.99 + 0.001 * np.arange(15)))
c1 = (
    np.max(np.abs(d_ok - 1.99)) < 1e-6
    and np.max(np.abs(d_bad - 1.99 - 0.001 * np.arange(15))) < 1e-6
    and np.max(d_bad) - np.min(d_bad) > 0.01
)
print(f"control c1 (synthetic law recovered; a drifting law is detected): {c1}", flush=True)
if not c1:
    print("READING q1371: VOID (control)", flush=True)
    os._exit(2)
states = []
for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
    try:
        j = json.load(open(f))
        res = float(numstr(j["residual"]))
        K = int(j["K"])
        A = float(numstr(j["A"]))
    except Exception:
        continue
    if j.get("verdict") == "OPTIMAL" and res < 1e-20 and K >= 60:
        states.append((K, j.get("wc") not in (None, "None"), A, masses_from_edge(j)))
states.sort(key=lambda t: (t[0], t[1], t[2]))
NJ = 13
rows = []
for s1, s2 in zip(states[:-1], states[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 0.002 < s2[2] - s1[2] < 0.2:
        rows.append((s1[0], deltas(s1[2], s1[3][:NJ], s2[2], s2[3][:NJ])))
allr = np.array([r[1] for r in rows])
glob_d = float(np.median(allr[:, :4]))
print(f"pairs {len(rows)}; global delta (median over j = 0..3) {glob_d:.8f}", flush=True)
print("j | median delta_j over all pairs | max - min over all pairs")
for jj in range(NJ):
    print(f"  {jj:2d} | {np.median(allr[:, jj]):.8f} | {allr[:, jj].max() - allr[:, jj].min():.2e}")
for K in (62, 70, 80, 82):
    m = np.array([r[1] for r in rows if r[0] == K])
    dev = np.abs(m - glob_d)
    n6 = int(np.sum(np.max(dev, axis=0) < 1e-6))
    n4 = int(np.sum(np.max(dev, axis=0) < 1e-4))
    print(
        f"K {K}: pairs {len(m)}, atoms j (of {NJ}) with |delta_j - global| < 1e-6 at every pair: {n6}, < 1e-4: {n4}; delta_0 spread {m[:, 0].max() - m[:, 0].min():.2e}, median delta_0 {np.median(m[:, 0]):.9f}"
    )
print("READING q1371 (exploratory, no verdict)")
