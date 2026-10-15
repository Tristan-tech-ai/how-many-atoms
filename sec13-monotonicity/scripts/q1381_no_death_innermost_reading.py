"""q1381 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1381_no_death_innermost_reading.py from temporal_within/centre_scripts/): the minimal reduction of the monotonicity of K(A) (stated here, derived in amendment 2139). By Barletta and Dytso (arXiv 2609.10039, Theorem 3 and Corollary 2) the support size can change only at the origin: the point 0 appears or disappears, or the centre atom splits into a pair +-x or a pair merges into the centre. Inside a family K is constant. A DECREASE of K as A grows needs (i) the centre atom of an odd family to lose all its mass (its mass p_c going to 0 while A grows) or (ii) the innermost pair of an even family to merge at the origin (its position u_1 going to 0 while A grows). If p_c is increasing along every odd family and u_1 is increasing along every even family, neither can happen, and K cannot decrease. Tested here on the stored families of four problems: for every consecutive same-family pair (same K, same centre structure, 0 < dA < 0.25) the sign of dp_c/dA (odd K, the centre mass) and of du_1/dA (even K, the innermost interior atom), and the sign of du_1/dA also in odd families (first pair outside the centre atom). Printed: pairs, the smallest value of the derivative (relative to the mean of the quantity per unit A), the number of pairs with a non-positive derivative. Data: capacity (results_q2*_gauss files, K >= 25), immune (results_q584_ext.json), NPMLE (results_q601_ext.json), LFP (c80 states). Expected before the run: no pair with a non-positive derivative (all innermost quantities increase)."""

import glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def cap():
    out = []
    for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
        try:
            j = json.load(open(f))
            res = float(numstr(j["residual"]))
            K = int(j["K"])
            A = float(numstr(j["A"]))
        except Exception:
            continue
        if j.get("verdict") == "OPTIMAL" and res < 1e-20:
            u = np.array([float(numstr(x)) for x in j["u"]])
            w = [float(numstr(x)) for x in j["w"]]
            wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
            out.append((K, wc is not None, A, wc, u))
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def weighted(fn):
    rows = sorted(json.load(open(os.path.join(ROOT, fn)))["rows"], key=lambda r: float(r["a"]))
    out = []
    for r in rows:
        v = [float(x) for x in r["v"]]
        centre = bool(r["centre"])
        out.append(
            (
                int(r["K"]),
                centre,
                float(r["a"]),
                (1 - sum(v)) if centre else None,
                np.array([float(x) for x in r["p"]]),
            )
        )
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def lfp():
    allst = {}
    for fn in ("c80_lfp_states.json", "c80_lfp_states_b.json"):
        for s in json.load(open(os.path.join(HERE, fn))):
            allst[(round(float(s["m"]), 6), int(s["K"]), bool(s["centre"]))] = s
    out = []
    for m, K, c in sorted(allst):
        s = allst[(m, K, c)]
        out.append(
            (
                K,
                c,
                m,
                float(s["centre_mass"]) if c and s.get("centre_mass") not in (None, "None") else None,
                np.array([float(x) for x in s["positions"]]),
            )
        )
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def run(name, st):
    rows = {"odd p_c": [], "even u_1": [], "odd u_1 (first pair)": []}
    for s1, s2 in zip(st[:-1], st[1:]):
        if s1[0] == s2[0] and s1[1] == s2[1] and 0 < s2[2] - s1[2] < 0.25:
            dA = s2[2] - s1[2]
            if s1[1]:
                rows["odd p_c"].append(((s2[3] - s1[3]) / dA / ((s1[3] + s2[3]) / 2)))
                rows["odd u_1 (first pair)"].append(
                    ((s2[4][0] - s1[4][0]) / dA / ((s1[4][0] + s2[4][0]) / 2))
                )
            else:
                rows["even u_1"].append(((s2[4][0] - s1[4][0]) / dA / ((s1[4][0] + s2[4][0]) / 2)))
    for k, v in rows.items():
        v = np.array(v)
        if len(v):
            print(
                f"{name}: {k}: pairs {len(v)}, smallest relative derivative (per unit A) {v.min():.4e}, pairs with derivative <= 0: {int(np.sum(v <= 0))}",
                flush=True,
            )


for name, st in (
    ("capacity", cap()),
    ("immune", weighted("results_q584_ext.json")),
    ("npmle", weighted("results_q601_ext.json")),
    ("lfp", lfp()),
):
    run(name, st)
print("READING q1381 (exploratory, no verdict)")
