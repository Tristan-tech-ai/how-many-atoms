"""q900 (6 Oct 2026): T50, the flow of the capacity atoms and masses with A from central differences of the stored certified states, written before any output was read (registered in amendment 1672).
Data and families as in q899 (results_q206/q207/q220_gauss_K*_A*.json, verdict OPTIMAL; K 50: A 35.5-35.74, K 60: 40.9-41.2, K 70: 46.3837-46.4337; where two files share an A the smaller residual is used). Inside a family the states are
0.01 apart (K 70: 0.01 as well), so the central difference of the neighbours (A_(i+1) - A_(i-1) <= 0.0201 of two adjacent files; steps that differ from 0.01 by more than 20 percent are skipped) approximates d/dA at every interior state with an error of order 1e-4.
For each interior state: dx_j/dA of the interior atoms (indexed from the edge), dm_j/dA of their per-atom masses, and dwe/dA of the wall atom.
Criteria: (a) dx_j/dA > 0.05 and at most 1.001 for every interior atom at every interior state; (b) the derivative of the innermost interior mass (j = 1 from the centre) is positive at every interior state; (c) at every interior state the derivatives
of the interior masses have at most one sign change, from + (inner) to - (outer). PASS: (a), (b), (c) in all three families. FAIL lists the family, the state and the item. Reported without criterion: the smallest dx_j/dA per family, the derivative of the
innermost mass at the first and last interior state, the number of positive derivatives of interior masses, and the sign of dwe/dA.
Declared prior: (a) and (b) PASS; (c) PASS (the last rows of the immune, NPMLE and least favourable prior chains show one sign change, + inner and - outer); the sign of dwe/dA undecided.
Usage: py q900_capacity_flows.py   (from temporal_within/)"""

import glob, json
import numpy as np

states = {}
for f in glob.glob("results_q2*_gauss_K*_A*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if (
        not (isinstance(d, dict) and all(k in d for k in ("A", "u", "w", "we")))
        or str(d.get("verdict")) != "OPTIMAL"
    ):
        continue
    A = float(str(d["A"]).replace("np.float64(", "").rstrip(")"))
    K = 2 * (len(d["u"]) + 1) + (1 if d.get("wc") is not None else 0)
    res = float(d.get("residual", 1))
    key = (K, round(A, 6))
    if key not in states or res < states[key][1]:
        states[key] = (d, res)
fam = {}
for (K, A), (d, res) in states.items():
    fam.setdefault(K, []).append((A, d))
allok = True
for K in sorted(fam):
    if len(fam[K]) < 4:
        continue
    L = sorted(fam[K], key=lambda t: t[0])
    As = np.array([a for a, d in L])
    U = [np.array([float(x) for x in d["u"]]) for a, d in L]
    Wm = [np.array([float(x) for x in d["w"]]) for a, d in L]
    WE = np.array([float(d["we"]) for a, d in L])
    bad_a = bad_b = bad_c = 0
    used = 0
    smallest = 9.0
    inn = []
    pos = []
    dwes = []
    for i in range(1, len(L) - 1):
        h1 = As[i] - As[i - 1]
        h2 = As[i + 1] - As[i]
        if abs(h1 - 0.01) > 0.002 or abs(h2 - 0.01) > 0.002:
            continue
        used += 1
        dA = As[i + 1] - As[i - 1]
        dx = (U[i + 1] - U[i - 1]) / dA
        dm = (Wm[i + 1] - Wm[i - 1]) / dA
        dwe = (WE[i + 1] - WE[i - 1]) / dA
        smallest = min(smallest, dx.min())
        if not ((dx > 0.05).all() and (dx <= 1.001).all()):
            bad_a += 1
        if not dm[0] > 0:
            bad_b += 1
        sg = np.sign(dm)
        ch = sum(1 for p, q in zip(sg, sg[1:]) if p != q)
        if not (ch <= 1 and (sg[0] > 0 or ch == 0)):
            bad_c += 1
        inn.append(dm[0])
        pos.append(int((dm > 0).sum()))
        dwes.append(dwe)
    ok = used > 0 and bad_a == 0 and bad_b == 0 and bad_c == 0
    allok = allok and ok
    print(
        f"K {K}: {len(L)} states, interior states used {used}; (a) violations {bad_a}, smallest dx/dA {smallest:.4f}; (b) violations {bad_b}, d(innermost mass)/dA {inn[0]:+.4e} (first) {inn[-1]:+.4e} (last); "
        f"(c) violations {bad_c}, positive interior-mass derivatives {pos[0]} (first) {pos[-1]} (last) of {len(U[0])}; dwe/dA {dwes[0]:+.3e} -> {dwes[-1]:+.3e} => {'PASS' if ok else 'FAIL'}"
    )
print("CAPACITY FLOW VERDICT:", "PASS" if allok else "FAIL")
