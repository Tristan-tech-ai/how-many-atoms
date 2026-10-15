"""q1447 (9 Oct 2026, READING, no verdict; track B of PROOF_TRACKS_2026-10-09.md): the terms of the innermost-atom flow at the stage f0.97 for the training rows K = 83 to 108, term by term.
For each dump q1415_jb_K<K>_f0.97.json the code of q1416_stage_row.analyse is repeated, but all terms c_k = -x_k bd_k are kept: flow = sum c, P = sum of the positive terms, N = minus the sum of the negative terms, the six edge terms are c[m-3..m-1] and c[m+m-3..m+m-1] (as in q1416). Output: q1447_terms.json (one entry per K with the six edge terms divided by the flow, the sign pattern, the share of the six terms in P and N, the largest single terms and their indices).
Rows above K = 108 are refused: they belong to the blind gate q1446. Usage: py q1447_edge_terms_reading.py [K ...] (from temporal_within/small_gap_q206/).
"""

import sys, os, json
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 51
HERE = os.path.dirname(os.path.abspath(__file__))


def terms(fn):
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
    P = sum(t for t in c if t > 0)
    N = -sum(t for t in c if t < 0)
    rel = abs(tot - ref) / abs(ref)
    pos = [m - 3, m - 2, m - 1, m + m - 3, m + m - 2, m + m - 1]
    six = [c[j] for j in pos]
    return dict(
        K=d["K"],
        parity="odd" if odd else "even",
        m=m,
        n=n,
        flow=float(ref),
        control_rel=float(rel),
        six_over_flow=[float(t / ref) for t in six],
        six_index=pos,
        six_sign="".join("+" if t > 0 else "-" for t in six),
        six_pos_share_of_P=float(sum(t for t in six if t > 0) / P),
        six_neg_share_of_N=float(-sum(t for t in six if t < 0) / N),
        P_over_flow=float(P / ref),
        N_over_flow=float(N / ref),
        largest=sorted(((int(k), float(c[k] / ref)) for k in range(n)), key=lambda t: -abs(t[1]))[:6],
    )


if __name__ == "__main__":
    Ks = [int(a) for a in sys.argv[1:]] or list(range(83, 109))
    out = {}
    for K in Ks:
        if K > 108:
            raise SystemExit("REFUSED: rows above K = 108 belong to the blind gate q1446")
        fn = os.path.join(HERE, f"q1415_jb_K{K}_f0.97.json")
        out[str(K)] = terms(fn)
        print(K, out[str(K)]["six_sign"], "control", "%.1e" % out[str(K)]["control_rel"], flush=True)
        json.dump(out, open(os.path.join(HERE, "q1447_terms.json"), "w"), indent=1)
