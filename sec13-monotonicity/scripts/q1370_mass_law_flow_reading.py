"""q1370 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1370_mass_law_flow_reading.py from temporal_within/centre_scripts/): decompose the mass flow dp_j/dA of the capacity into the flow of the local density mu_j = p_j / s_j and of the cell s_j = (x_(j+1) - x_(j-1)) / 2 (T31: the mass of an atom is the flat density times its cell, median error 0.1 percent). With p_j = mu_j s_j, dp_j/dA = s_j dmu_j/dA + mu_j ds_j/dA, so dp_j/dA > 0 iff dln s_j/dA > -dln mu_j/dA. If -dln mu_j/dA is nearly the same for all j (the density falls with A as 1/(A + delta)), the sign change of the mass flow is the crossing of the single function dln s_j/dA through a constant.
Printed for the middle pair of consecutive same-family states of K = 62, 70 and 80: for selected atoms j (counted from the edge: j = 1 is the first interior atom below the edge atom) x_j' = dx_j/dA, dln s_j/dA, -dln mu_j/dA, dp_j/dA, and over all interior atoms j the spread of -dln mu_j/dA (min, max, mean), whether dln s_j/dA is monotone in j, and the position of the sign change of dp_j/dA.
Expected before the run: unknown (a reading to find the structure)."""

import glob, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def load(K):
    out = []
    for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K%d_A*.json" % K), recursive=True):
        try:
            j = json.load(open(f))
            res = float(numstr(j["residual"]))
        except Exception:
            continue
        if j.get("verdict") == "OPTIMAL" and res < 1e-20:
            out.append((float(numstr(j["A"])), j))
    return sorted(out, key=lambda t: t[0])


def seq(j):
    A = float(numstr(j["A"]))
    u = [float(numstr(x)) for x in j["u"]]
    w = [float(numstr(x)) for x in j["w"]]
    we = float(numstr(j["we"]))
    wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
    x = ([0.0] if wc is not None else []) + u + [A]
    p = ([wc] if wc is not None else []) + w + [we]
    return A, np.array(x), np.array(p), wc is not None


for K in (62, 70, 80):
    L = load(K)
    i = len(L) // 2
    (A1, j1), (A2, j2) = L[i], L[i + 1]
    while A2 - A1 < 1e-4 and i + 2 < len(L):
        i += 1
        (A1, j1), (A2, j2) = L[i], L[i + 1]
    _, x1, p1, c1 = seq(j1)
    _, x2, p2, c2 = seq(j2)
    dA = A2 - A1
    if len(x1) != len(x2) or c1 != c2:
        print("K", K, "families differ")
        continue
    xm = (x1 + x2) / 2
    pm = (p1 + p2) / 2
    xp = (x2 - x1) / dA
    pp = (p2 - p1) / dA
    n = len(xm)
    lo = 2 if not c1 else 2
    idx = list(range(lo, n - 1))  # skip the two innermost and the edge atom
    # mirror the first atom when there is no centre atom: x_0 = -x_1
    xe = np.concatenate([[-xm[0]], xm]) if not c1 else np.concatenate([[-xm[1]], xm])
    xpe = np.concatenate([[-xp[0]], xp]) if not c1 else np.concatenate([[-xp[1]], xp])
    off = 1
    s = np.array([(xe[k + off + 1] - xe[k + off - 1]) / 2 for k in range(n - 1)])
    sp = np.array([(xpe[k + off + 1] - xpe[k + off - 1]) / 2 for k in range(n - 1)])
    mu = pm[: n - 1] / s
    mup = np.gradient(mu) * 0  # placeholder
    dlns = sp / s
    dlnp = pp[: n - 1] / pm[: n - 1]
    dlnmu = dlnp - dlns
    ii = np.array(idx)
    print(f"K {K}, A {A1:.5f}->{A2:.5f}, centre {c1}, atoms {n}")
    print("  j(from edge) x_j' dln s_j/dA  -dln mu_j/dA  dp_j/dA(1e-4)")
    for jj in range(len(ii) - 1, -1, -max(1, len(ii) // 10)):
        k = ii[jj]
        print(f"  {n - 1 - k:3d}  {xp[k]:.4f}  {dlns[k]:+.5f}  {-dlnmu[k]:+.5f}  {pp[k] * 1e4:+.4f}")
    print(
        f"  over {len(ii)} atoms: -dln mu/dA min {np.min(-dlnmu[ii]):.5f} max {np.max(-dlnmu[ii]):.5f} mean {np.mean(-dlnmu[ii]):.5f}; 1/(A+0.2) = {1 / (A1 + 0.2):.5f}; dln s/dA monotone in j: {bool(np.all(np.diff(dlns[ii]) >= -1e-9) or np.all(np.diff(dlns[ii]) <= 1e-9))}; x' monotone: {bool(np.all(np.diff(xp[ii]) >= -1e-9) or np.all(np.diff(xp[ii]) <= 1e-9))}"
    )
print("READING q1370 (exploratory, no verdict)")
