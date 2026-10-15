"""q1389 reading (9 Oct 2026, EXPLORATORY, no item, no verdict; py q1389_basis_cancellation.py FILE... from temporal_within/small_gap_q206/): offline analysis of the dumps of q1388. For the innermost entry (u_1 for even K, wc for odd K) of dz/dA = -J^-1 b the contributions c_k = -(J^-T e)_k b_k are formed in four equation bases and the cancellation ratio sum|c_k| / |sum c_k| and the sign counts are printed: (S) the stored system; (R) J1, the principal 2m x 2m block of the (u, w) unknowns with we fixed (the scale gauge: the whole mass measure can be scaled), valid for u_1 only (even K); (D1) rows D(u_j)-D(A) replaced by D(u_j)-D(u_1) for j >= 2 (row j minus row 1); (T) telescoping differences D(u_j)-D(u_{j+1}), D(u_m)-D(A). Control: in every basis the sum of the contributions equals the entry of the stored-system solution to 1e-30 relative (the row operations do not change the solution)."""

import sys, json
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 40


def entries(J, b, idx):
    n = J.rows
    e = matrix(n, 1)
    e[idx] = 1
    x = lu_solve(J.T, e)
    c = [-x[k] * b[k] for k in range(n)]
    return x, c


def report(name, c, ref, extra="", info=False):
    tot = sum(c)
    a = sum(abs(t) for t in c)
    npos = sum(1 for t in c if t > 0)
    nneg = sum(1 for t in c if t < 0)
    rel = abs(tot - ref) / abs(ref)
    ok = rel <= mpf("1e-30") * (a / abs(tot))
    flag = "information" if info else ("OK" if ok else "FAILED")
    print(
        f"   {name:4s}: sum {mp.nstr(tot, 10)} (control {flag}, relative difference {mp.nstr(rel, 3)}); sum|c|/|sum c| = {mp.nstr(a / abs(tot), 5)}; positive part {mp.nstr(sum(t for t in c if t > 0), 8)}, negative part {mp.nstr(sum(t for t in c if t < 0), 8)}, ratio {mp.nstr(sum(t for t in c if t > 0) / -sum(t for t in c if t < 0), 6)}; positive {npos}, negative {nneg}; largest |c| {mp.nstr(max(abs(t) for t in c), 4)} {extra}"
    )


for fn in sys.argv[1:]:
    d = json.load(open(fn))
    m = d["m"]
    odd = d["odd"]
    n = 2 * m + 1 + (1 if odd else 0)
    J = matrix(n, n)
    for a in range(n):
        for c in range(n):
            J[a, c] = mpf(d["J"][a][c])
    b = [mpf(t) for t in d["b"]]
    idx = (2 * m + 1) if odd else 0
    dz = lu_solve(J, matrix([-t for t in b]))
    ref = dz[idx]
    print(
        f"K {d['K']} {d['label']} A {d['A']} ({'odd, wc' if odd else 'even, u_1'}), stored-system flow {mp.nstr(ref, 12)}, J {n}x{n}"
    )
    x, c = entries(J, b, idx)
    report("S", c, ref)
    if not odd:
        J1 = matrix(2 * m, 2 * m)
        for a in range(2 * m):
            for c_ in range(2 * m):
                J1[a, c_] = J[a, c_]
        x1, c1 = entries(J1, b[: 2 * m], 0)
        report("R", c1, ref, info=True)
    # D1 basis: rows 1..m-1 (0-based) minus row 0, in the D block
    T = matrix(n, n)
    for a in range(n):
        T[a, a] = 1
    for a in range(1, m):
        T[a, 0] = -1
    if odd:
        T[2 * m + 1, 0] = -1
    Jd = T * J
    bd = T * matrix(b)
    x2, c2 = entries(Jd, [bd[k] for k in range(n)], idx)
    report("D1", c2, ref)
    T2 = matrix(n, n)
    for a in range(n):
        T2[a, a] = 1
    for a in range(0, m - 1):
        T2[a, a + 1] = -1
    if odd:
        T2[2 * m + 1, 0] = -1
    Jt = T2 * J
    bt = T2 * matrix(b)
    x3, c3 = entries(Jt, [bt[k] for k in range(n)], idx)
    report("T", c3, ref)
    sgn = lambda v: "".join("+" if t > 0 else ("-" if t < 0 else "0") for t in v)
    print(
        "   signs of c in S, D block:",
        sgn(c[:m]),
        "| D' block:",
        sgn(c[m : 2 * m]) + (" | D(0) row: " + sgn(c[2 * m + 1 :]) if odd else ""),
    )
    print(
        "   signs of c in T, D block:",
        sgn(c3[:m]),
        "| D' block:",
        sgn(c3[m : 2 * m]) + (" | D(0) row: " + sgn(c3[2 * m + 1 :]) if odd else ""),
    )
    print(
        "   signs of c in D1, D block:",
        sgn(c2[:m]),
        "| D' block:",
        sgn(c2[m : 2 * m]) + (" | D(0) row: " + sgn(c2[2 * m + 1 :]) if odd else ""),
    )
print("READING q1389 done")
