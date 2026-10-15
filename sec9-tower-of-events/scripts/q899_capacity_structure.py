"""q899 (6 Oct 2026): C1-C4, the structure of amendments 1643-1669 on the amplitude-constrained Gaussian capacity itself, written before any output was read (registered in amendment 1670).
Data: the stored certified Gaussian capacity states (the files read by cap_beta.py: results_q206/q207/q220_gauss_K*_A*.json with fields A, u, w, we, wc, residual, verdict). 'u' are the interior positive atoms (ascending), 'w' their per-atom masses,
'we' the mass of the atom on the wall at x = A, 'wc' the mass of the atom at 0 (None here: every family below has no centre atom). Total mass: 2 (sum of w + we) + wc = 1. A family is the set of states with verdict OPTIMAL of one K
(50: A = 35.5 to 35.74; 60: A = 40.9 to 41.2; 70: A = 46.3837 to 46.4337) sorted by A; where two files exist for the same A, the one with the smaller residual is used. The next state of each family (verdict 'centre positive') is just past
the birth of a centre atom, so the last OPTIMAL state sits at the end of the family. The edge atom is not included in the interior list.
C1 (A2): between consecutive states of a family every interior atom moves outward, u_j(A') > u_j(A), indexed from the edge.
C2: the mass of the innermost interior atom (w[0]) increases strictly from state to state in each family.
C3 (cell law): at the last state of each family, the median over the interior atoms 2..10 counted from the centre of |m_j / (cell_j/(2A + beta)) - 1| is below 0.003, with beta = 3.98160794709 (the capacity strip of the record, amendment 726),
cell_j = (u_(j+1) - u_(j-1))/2 and u_0 = -u_1 (the mirror image of the first atom).
C4: at the last state of each family w = 2 u_1/(u_2 - u_1) lies in [0.97, 1.03].
PASS: C1 to C4 in all three families. FAIL lists the family and the item. Reported without criterion: the same C3 median at the first state of each family and mu a at the last state (mu = mass/cell for the inner atoms).
Declared prior: C1, C2, C4 PASS; C3 undecided (the capacity has a flat output density and an input density that should be flat on the widened interval; the cell law has not been tested on it).
Usage: py q899_capacity_structure.py   (from temporal_within/)"""

import glob, json
import numpy as np

BETA = 3.98160794709
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
        states[key] = (d, res, f)
fam = {}
for (K, A), (d, res, f) in states.items():
    fam.setdefault(K, []).append((A, d))
allok = True
for K in sorted(fam):
    if len(fam[K]) < 4:
        continue
    L = sorted(fam[K], key=lambda t: t[0])
    U = [np.array([float(x) for x in d["u"]]) for A, d in L]
    Wm = [np.array([float(x) for x in d["w"]]) for A, d in L]
    WE = [float(d["we"]) for A, d in L]
    As = [A for A, d in L]
    viol = sum(
        1
        for i in range(len(L) - 1)
        for j in range(1, len(U[i]) + 1)
        if not (U[i + 1][len(U[i + 1]) - j] > U[i][len(U[i]) - j])
    )
    c1 = viol == 0
    inn = [w[0] for w in Wm]
    c2 = all(b > a for a, b in zip(inn, inn[1:]))

    def cell_med(i):
        u = U[i]
        w = Wm[i]
        A = As[i]
        xs = np.concatenate(([-u[0]], u))
        e = []
        for j in range(1, 10):  # atoms 2..10 counted from the centre (0-based 1..9)
            cell = (xs[j + 2] - xs[j]) / 2
            e.append(w[j] / (cell / (2 * A + BETA)) - 1)
        return float(np.median(np.abs(e)))

    c3 = cell_med(-1) < 0.003
    wl = 2 * U[-1][0] / (U[-1][1] - U[-1][0])
    c4 = 0.97 <= wl <= 1.03
    mass = 2 * (Wm[-1].sum() + WE[-1])
    ok = c1 and c2 and c3 and c4
    allok = allok and ok
    print(
        f"K {K}: {len(L)} states, A {As[0]:.4f}-{As[-1]:.4f}; C1 violations {viol} -> {'PASS' if c1 else 'FAIL'}; C2 innermost mass {inn[0]:.5f} -> {inn[-1]:.5f} {'PASS' if c2 else 'FAIL'}; "
        f"C3 median |e| last {cell_med(-1):.4f} (first {cell_med(0):.4f}) {'PASS' if c3 else 'FAIL'}; C4 w {wl:.4f} {'PASS' if c4 else 'FAIL'}; total mass of the last state {mass:.6f} => {'PASS' if ok else 'FAIL'}"
    )
print("CAPACITY VERDICT:", "PASS" if allok else "FAIL")
