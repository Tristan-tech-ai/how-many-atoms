"""q1398 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1398_row1_remainder.py FILE... from temporal_within/small_gap_q206/): in basis D1 of q1389 the rows j = 2..m (and the D(0) row for odd K) have forcing zero at 40 digits except in the edge region; row 1 (D(u_1) - D(A)) carries the common-mode forcing. Printed per dump: the flow, the row-1 contribution c_1, the edge sum E = flow - c_1 (all other rows), and c_1 / E. Expected before the run: c_1 is negative and between 2 and 4 percent of N, so |c_1| / E lies between 0.2 and 0.4. Control: E + c_1 = flow by construction is not a test, so the control is the sum of all contributions against the stored-system flow (1e-26 relative)."""

import sys, json
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 40
print("K par A flow c_1 E=flow-c_1 c_1/E  E/flow | outer-3 net | control")
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
    ok = abs(sum(c) - ref) / abs(ref) < mpf("1e-26")
    c1 = c[0]
    E = ref - c1
    mid = (
        sum(c[k] for k in range(1, m - 3))
        + sum(c[m + k] for k in range(0, m - 3))
        + (c[2 * m + 1] if odd else 0)
    )
    print(
        f"{d['K']} {'odd' if odd else 'even'} {d['A'][:8]} {mp.nstr(ref, 6)} {mp.nstr(c1, 6)} {mp.nstr(E, 6)} {mp.nstr(c1 / E, 5)} {mp.nstr(E / ref, 5)} | rest-between {mp.nstr(mid, 3)} | {'OK' if ok else 'FAILED'}",
        flush=True,
    )
print("READING q1398 done")
