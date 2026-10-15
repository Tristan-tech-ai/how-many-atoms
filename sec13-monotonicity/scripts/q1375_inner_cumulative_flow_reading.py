"""q1375 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1375_inner_cumulative_flow_reading.py from temporal_within/centre_scripts/): the statement left to prove in amendments 2130-2134: for every m the mass T_m(A) of the m innermost right-half atoms (centre atom first if present, the edge atom last) is nondecreasing in A. For every consecutive same-family pair (same K, same centre structure, 1e-6 < A2 - A1 < 0.2) of the stored states of the three problems compute dT_m = sum_{i<=m} (p2_i - p1_i) / (A2 - A1) for m = 1..n and report: the smallest dT_m over m and pairs (the statement holds if it is >= 0 up to rounding 1e-14), the margin dT_m divided by the scale S = 1 / (2 (A + delta)) (delta = 1.990803974 for the capacity; for the weighted problems the scale 1 / (2 a) with a the stored amplitude), where the tightest m sits (as a fraction of n and as an absolute index), the dependence on K and on the age of the family, and for the capacity the identity dT_(n-r) = C_r / (A + delta)^2 (C_r = (A + delta) sum of the r outermost masses) for r = 1..4 (an exact consequence of the scaling law). Expected before the run: no negative value; the tightest m near the end of the positive block or near n - r_edge; unknown otherwise."""

import glob, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DELTA = 1.990803974


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def cap_states():
    out = []
    for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
        try:
            j = json.load(open(f))
            res = float(numstr(j["residual"]))
            K = int(j["K"])
            A = float(numstr(j["A"]))
        except Exception:
            continue
        if j.get("verdict") == "OPTIMAL" and res < 1e-20 and K >= 60:
            w = [float(numstr(x)) for x in j["w"]]
            we = float(numstr(j["we"]))
            wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
            out.append((K, wc is not None, A, np.array(([wc] if wc is not None else []) + w + [we])))
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def weighted_states(fn):
    rows = sorted(json.load(open(os.path.join(ROOT, fn)))["rows"], key=lambda r: float(r["a"]))
    out = []
    for r in rows:
        v = [float(x) for x in r["v"]]
        centre = bool(r["centre"])
        c = 1 - sum(v) if centre else None
        out.append(
            (int(r["K"]), centre, float(r["a"]), np.array(([c] if centre else []) + [x / 2 for x in v]))
        )
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def analyse(name, states, scale_fn):
    fam_first = {}
    for s in states:
        fam_first.setdefault((s[0], s[1]), s[2])
    mins = []
    neg = 0
    npairs = 0
    tight_m = []
    byK = {}
    ages = []
    margins_all = []
    for s1, s2 in zip(states[:-1], states[1:]):
        if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
            dA = s2[2] - s1[2]
            Am = (s1[2] + s2[2]) / 2
            dT = np.cumsum((s2[3] - s1[3]) / dA)[:-1]
            S = scale_fn(Am)
            npairs += 1  # m = 1..n-1 (m = n is the constant total)
            mi = int(np.argmin(dT))
            mins.append(dT[mi] / S)
            tight_m.append((mi + 1, len(dT) + 1))
            neg += int(dT[mi] < -1e-14 / dA)
            byK.setdefault(s1[0], []).append(dT[mi] / S)
            ages.append((Am - fam_first[(s1[0], s1[1])], dT[mi] / S))
    mins = np.array(mins)
    tm = np.array(tight_m)
    a = np.array(ages)
    print(
        f"{name}: pairs {npairs}; pairs with a negative cumulative flow (below -1e-14/dA): {neg}; smallest margin dT_m / S over all pairs and m: {mins.min():.4e}; median over pairs of the pair's smallest margin: {np.median(mins):.4e}",
        flush=True,
    )
    print(
        f"   tightest m: median {np.median(tm[:, 0]):.0f} of n = {np.median(tm[:, 1]):.0f}; fraction of pairs whose tightest m is 1: {np.mean(tm[:, 0] == 1):.3f}, is within 3 of n: {np.mean(tm[:, 0] >= tm[:, 1] - 3):.3f}; correlation of the margin with the family age: {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.3f}",
        flush=True,
    )
    ks = sorted(byK)
    print(
        "   median margin by K (every 5th): " + ", ".join(f"K {k}: {np.median(byK[k]):.3e}" for k in ks[::5]),
        flush=True,
    )


cap = cap_states()
analyse("capacity", cap, lambda A: 1 / (2 * (A + DELTA)))
# identity for the outer cumulative flow
errs = []
for s1, s2 in zip(cap[:-1], cap[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
        dA = s2[2] - s1[2]
        Am = (s1[2] + s2[2]) / 2
        d = (s2[3] - s1[3]) / dA
        pm = (s1[3] + s2[3]) / 2
        for r in range(1, 5):
            errs.append(abs(np.sum(d[-r:]) / (-np.sum(pm[-r:]) / (Am + DELTA)) - 1))
print(
    f"capacity: outer cumulative flow of the r = 1..4 outermost atoms over -sum p / (A + delta) - 1: largest {max(errs):.2e}, median {np.median(errs):.2e}",
    flush=True,
)
analyse("immune", weighted_states("results_q584_ext.json"), lambda a: 1 / (2 * a))
analyse("npmle", weighted_states("results_q601_ext.json"), lambda a: 1 / (2 * a))
print("READING q1375 (exploratory, no verdict)")


# second part: margins restricted to the inner range m <= n - 6 (beyond the edge-indexed atoms obeying the scaling law)
def inner_range(name, states, scale_fn, rcut=6):
    vals = []
    first = []
    for s1, s2 in zip(states[:-1], states[1:]):
        if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
            dA = s2[2] - s1[2]
            Am = (s1[2] + s2[2]) / 2
            dT = np.cumsum((s2[3] - s1[3]) / dA)[:-1]
            S = scale_fn(Am)
            n = len(dT) + 1
            vals.append(dT[: n - rcut].min() / S)
            first.append(dT[0] / S)
    v = np.array(vals)
    f = np.array(first)
    print(
        f"{name}: inner range m <= n - {rcut}: smallest margin dT_m / S over pairs {v.min():.4e}, median {np.median(v):.4e}; m = 1 (the innermost mass flow) smallest {f.min():.4e}, median {np.median(f):.4e}",
        flush=True,
    )


inner_range("capacity", cap, lambda A: 1 / (2 * (A + DELTA)))
inner_range("immune", weighted_states("results_q584_ext.json"), lambda a: 1 / (2 * a))
inner_range("npmle", weighted_states("results_q601_ext.json"), lambda a: 1 / (2 * a))
print("READING q1375b (exploratory, no verdict)")
