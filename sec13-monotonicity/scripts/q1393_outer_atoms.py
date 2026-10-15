"""q1393 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1393_outer_atoms.py FILE... from temporal_within/small_gap_q206/): in the D1 basis of q1389 (row j of the D block minus row 1; for odd K the extra row D(0)-D(A) minus row 1) the contributions c_k of the equation rows to the innermost flow, grouped by atom: atom j (counted from the centre, j = m the outermost interior atom) has the contributions c_D(j) and c_D'(j). Printed: the share of the three outermost atoms in sum|c| and in the positive and negative parts, the flow they give by themselves, and the number of atoms with |c_D| + |c_D'| above 1e-6 of sum|c|. Control: the sum over all rows equals the stored-system flow (relative difference below 1e-26)."""

import sys, json
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 40
for fn in sys.argv[1:]:
    d = json.load(open(fn))
    m = d["m"]
    odd = d["odd"]
    n = 2 * m + 1 + (1 if odd else 0)
    idx = (2 * m + 1) if odd else 0
    J = matrix(n, n)
    for a in range(n):
        for c in range(n):
            J[a, c] = mpf(d["J"][a][c])
    b = [mpf(t) for t in d["b"]]
    ref = lu_solve(J, matrix([-t for t in b]))[idx]
    T = matrix(n, n)
    for a in range(n):
        T[a, a] = 1
    for a in range(1, m):
        T[a, 0] = -1
    if odd:
        T[2 * m + 1, 0] = -1
    Jd = T * J
    bd = T * matrix(b)
    e = matrix(n, 1)
    e[idx] = 1
    x = lu_solve(Jd.T, e)
    c = [-x[k] * bd[k] for k in range(n)]
    tot = sum(c)
    ok = abs(tot - ref) / abs(ref) < mpf("1e-26")
    S = sum(abs(t) for t in c)
    atom = [c[j] + c[m + j] for j in range(m)]
    atom_abs = [abs(c[j]) + abs(c[m + j]) for j in range(m)]
    if odd:
        atom[0] += 0
    top = list(range(m - 3, m))
    Pall = sum(t for t in c if t > 0)
    Nall = -sum(t for t in c if t < 0)
    Ptop = sum(max(c[j], 0) + max(c[m + j], 0) for j in top)
    Ntop = -sum(min(c[j], 0) + min(c[m + j], 0) for j in top)
    nsig = sum(1 for j in range(m) if atom_abs[j] > mpf("1e-6") * S)
    print(
        f"K {d['K']} {d['label']} A {d['A']}: control {'OK' if ok else 'FAILED'}; flow {mp.nstr(ref, 8)}; three outermost atoms: share of sum|c| {mp.nstr(sum(atom_abs[j] for j in top) / S, 8)}, P_top/P {mp.nstr(Ptop / Pall, 6)}, N_top/N {mp.nstr(Ntop / Nall, 6)}, flow from the three {mp.nstr(sum(atom[j] for j in top), 6)}; per atom net (outer three, from the third last) {[mp.nstr(atom[j], 5) for j in top]}; atoms with |c| above 1e-6 of sum|c|: {nsig} of {m}"
    )
print("READING q1393 done")
