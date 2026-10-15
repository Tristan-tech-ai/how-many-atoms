"""q1397 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1397_outer_reference_basis.py FILE... from temporal_within/small_gap_q206/): basis O of the equation rows: the rows D(u_j) - D(A) are replaced by D(u_j) - D(u_m) for j < m (u_m the outermost interior atom), the row j = m stays D(u_m) - D(A), and for odd K the row D(0) - D(A) becomes D(0) - D(u_m). All rows with a non-zero forcing b then lie in the edge region, so the flow of the innermost atom is a sum of edge-local terms c_k = -x_k b_k. Printed per dump of q1388: cancellation ratio sum|c| / |sum c|, P/N, number of rows with |c| above 1e-12 of sum|c|, the sign string of the nonzero contributions, and the share of the row j = m. Expected before the run: unknown (the row-1 term of basis D1 disappears into the rows of the edge, so N falls and P/N grows). Control: the sum of all contributions equals the stored-system flow to 1e-24 relative."""

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
    for a in range(0, m - 1):
        T[a, m - 1] = -1
    if odd:
        T[2 * m + 1, m - 1] = -1
    Jo = T * J
    bo = T * matrix(b)
    e = matrix(n, 1)
    e[idx] = 1
    x = lu_solve(Jo.T, e)
    c = [-x[k] * bo[k] for k in range(n)]
    tot = sum(c)
    S = sum(abs(t) for t in c)
    P = sum(t for t in c if t > 0)
    N = -sum(t for t in c if t < 0)
    ok = abs(tot - ref) / abs(ref) < mpf("1e-24")
    nz = [k for k in range(n) if abs(c[k]) > mpf("1e-12") * S]
    sg = "".join("+" if c[k] > 0 else "-" for k in nz)
    lab = lambda k: (
        f"D{k + 1}" if k < m else (f"D'{k - m + 1}" if k < 2 * m else ("norm" if k == 2 * m else "D0"))
    )
    print(
        f"K {d['K']} {d['label']} A {d['A']}: control {'OK' if ok else 'FAILED'}; flow {mp.nstr(ref, 8)}; sum|c|/|sum c| {mp.nstr(S / abs(tot), 5)}; P/N {mp.nstr(P / N, 6)}; rows above 1e-12: {len(nz)} [{', '.join(lab(k) for k in nz)}]; signs {sg}; share of row D{m} {mp.nstr(abs(c[m - 1]) / S, 4)}",
        flush=True,
    )
print("READING q1397 done")
