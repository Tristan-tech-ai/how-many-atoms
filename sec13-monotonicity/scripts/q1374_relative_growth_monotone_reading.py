"""q1374 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1374_relative_growth_monotone_reading.py from temporal_within/centre_scripts/): with the scaling law of amendment 2132 (p_j (A + delta) constant for the edge-indexed atoms, delta = 1.990803974), write the mass flow as dp_j/dA = p_j [g_j - 1/(A + delta)] with g_j = dln p_j/dA + 1/(A + delta), the growth rate of the mass of atom j relative to the scale law (g_j = 0 for the outer atoms). The one-sign-change of the mass flow follows at once if g_j is nonincreasing from the centre outward (the flow is positive where g_j > 1/(A + delta)). Test: for every consecutive same-family pair of the stored capacity states (K >= 40, 1e-6 < A2 - A1 < 0.2, 2637 pairs) compute g_j with the finite differences (p2 - p1)/(A2 - A1) / mean p, in position order (centre or innermost pair first, edge atom last), and check g_(j) >= g_(j+1) - tol with tol = 1e-12 (rounding of p about 1e-16 relative to dA). Printed: the number of pairs with g nonincreasing, the number of violations and the largest violation relative to the scale 1/(A + delta), the number of pairs with g_edge within 1e-6 of 0, and the first atom index where g drops below 1/(A + delta) (the sign change) summarised. Expected before the run: unknown (the sufficient condition may be too strong)."""

import glob, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DELTA = 1.990803974


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def vec(j):
    w = [float(numstr(x)) for x in j["w"]]
    we = float(numstr(j["we"]))
    wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
    return ([wc] if wc is not None else []) + w + [we], wc is not None


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
        m, c = vec(j)
        states.append((K, c, A, np.array(m)))
states.sort(key=lambda t: (t[0], t[1], t[2]))
# control: a monotone synthetic g gives no violation, a non-monotone one is detected
gs = np.array([0.3, 0.2, 0.1, 0.0])
gb = np.array([0.3, 0.1, 0.2, 0.0])
ctl = bool(np.all(np.diff(gs) <= 1e-12)) and not bool(np.all(np.diff(gb) <= 1e-12))
print(f"control c1: {ctl}", flush=True)
if not ctl:
    print("READING q1374: VOID (control)", flush=True)
    os._exit(2)
npairs = 0
nmono = 0
viol = []
edge_ok = 0
cross = []
for s1, s2 in zip(states[:-1], states[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
        dA = s2[2] - s1[2]
        Am = (s1[2] + s2[2]) / 2
        pm = (s1[3] + s2[3]) / 2
        dp = (s2[3] - s1[3]) / dA
        g = dp / pm + 1 / (Am + DELTA)
        sc = 1 / (Am + DELTA)
        npairs += 1
        d = np.diff(g)
        worst = float(np.max(d)) if len(d) else 0.0
        if worst <= 1e-12:
            nmono += 1
        else:
            viol.append((s1[0], Am, worst / sc, int(np.argmax(d))))
        edge_ok += int(abs(g[-1]) < 1e-6)
        above = np.where(g > sc)[0]
        cross.append(len(above))
print(
    f"pairs {npairs}; g_j nonincreasing from the centre outward in {nmono}; violations {len(viol)}; edge atom |g| < 1e-6 in {edge_ok}",
    flush=True,
)
if viol:
    v = np.array([x[2] for x in viol])
    print(
        f"largest violation relative to 1/(A + delta): {v.max():.3e}; median {np.median(v):.3e}; examples (K, A, relative violation, index): {[(x[0], round(x[1], 3), float('%.2e' % x[2]), x[3]) for x in sorted(viol, key=lambda x: -x[2])[:6]]}",
        flush=True,
    )
c = np.array(cross)
print(f"number of atoms with g_j above the scale (positive flow): min {c.min()}, max {c.max()}", flush=True)
print("READING q1374 (exploratory, no verdict)")
# part 2 (added 9 Oct 00:33, correction of amendment 2134): where do the violations of the monotonicity of g_j sit?
idx = {}
allviol = {}
for s1, s2 in zip(states[:-1], states[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
        dA = s2[2] - s1[2]
        Am = (s1[2] + s2[2]) / 2
        pm = (s1[3] + s2[3]) / 2
        g = (s2[3] - s1[3]) / dA / pm + 1 / (Am + DELTA)
        d = np.diff(g)
        bad = np.where(d > 1e-12)[0]
        for b in bad:
            allviol[int(b)] = allviol.get(int(b), 0) + 1
        if len(bad):
            idx[int(np.argmax(d))] = idx.get(int(np.argmax(d)), 0) + 1
print(
    "index (from the centre, 0 = first step between the first two atoms) of the largest violation per pair:",
    sorted(idx.items())[:12],
    "...; total violating pairs",
    sum(idx.values()),
    flush=True,
)
print(
    "all violating steps by index (first 12):",
    sorted(allviol.items())[:12],
    "; largest index with a violation",
    max(allviol),
    flush=True,
)
print("READING q1374b (exploratory, no verdict)")
# part 3: noise-aware tolerance. The finite-difference noise of g_j is about 4e-16 / (A2 - A1) (rounding of the stored masses divided by the step); the fixed tolerance 1e-12 of part 1 is below it for steps under 4e-4.
nm = 0
nv = 0
real = []
idx2 = {}
for s1, s2 in zip(states[:-1], states[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
        dA = s2[2] - s1[2]
        Am = (s1[2] + s2[2]) / 2
        pm = (s1[3] + s2[3]) / 2
        g = (s2[3] - s1[3]) / dA / pm + 1 / (Am + DELTA)
        d = np.diff(g)
        tol = 1e-12 + 4e-16 / dA * 1.0
        bad = np.where(d > tol)[0]
        if len(bad) == 0:
            nm += 1
        else:
            nv += 1
            real.append(float(np.max(d[bad]) * (Am + DELTA)))
            k = int(bad[np.argmax(d[bad])])
            idx2[k] = idx2.get(k, 0) + 1
print(
    f"noise-aware tolerance: g_j nonincreasing in {nm} of {nm + nv} pairs; violations in {nv}; relative size of the largest violation per pair (times A + delta): median {np.median(real):.3e}, max {np.max(real):.3e}; index of the largest violation: {sorted(idx2.items())[:10]} ... largest index {max(idx2)}",
    flush=True,
)
print("READING q1374c (exploratory, no verdict)")
