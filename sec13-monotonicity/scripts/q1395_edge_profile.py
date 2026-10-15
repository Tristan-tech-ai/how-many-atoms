"""q1395 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1395_edge_profile.py from temporal_within/small_gap_q206/): is the ratio P/N an edge-region quantity? For the 21 states of q1388 at the stage f0.97 (K = 62..82) the D1-basis contributions c = -x b of the three outermost atoms are normalised by the contribution of the D' row of the outermost atom (the overall amplitude of the transfer to the centre cancels in P/N). Printed per state: depth of the three outermost atoms below the wall s_j = A - u_(m+1-j) (j = 1..3), and the normalised contributions of the rows D(u_j) - D(u_1) and D'(u_j) of the atoms m, m-1, m-2. Expected before the run: the normalised contributions agree between K = 62 and K = 82 to about 1 percent (they change as slowly as P/N does), while the depths s_j change by a few percent. Control: the sum of all contributions equals the stored-system flow to 1e-26 relative (OK per state, printed)."""

import json, sys
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 40
print(
    "K   par  A         s1       s2       s3     | c_D'(m) c_D(m)/c_D'(m)  c_D'(m-1)/c_D'(m) c_D(m-1)/c_D'(m)  c_D'(m-2)/c_D'(m) c_D(m-2)/c_D'(m) | P/N"
)
for K in range(62, 83):
    d = json.load(open(f"q1388_jb_K{K}_f0.97.json"))
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
    ok = abs(sum(c) - ref) / abs(ref) < mpf("1e-26")
    A = mpf(d["A"])
    u = [mpf(t) for t in d["u"]]
    P = sum(t for t in c if t > 0)
    N = -sum(t for t in c if t < 0)
    ref0 = c[m + m - 1]
    f = lambda t: float(t / ref0)
    print(
        f"{K} {'odd ' if odd else 'even'} {float(A):.4f} {float(A - u[-1]):.5f} {float(A - u[-2]):.5f} {float(A - u[-3]):.5f} | {mp.nstr(ref0, 5)} {f(c[m - 1]):.5f} {f(c[2 * m - 2]):.5f} {f(c[m - 2]):.5f} {f(c[2 * m - 3]):.5f} {f(c[m - 3]):.5f} | {float(P / N):.5f} {'ok' if ok else 'CONTROL FAILED'}",
        flush=True,
    )
print("READING q1395 done")
