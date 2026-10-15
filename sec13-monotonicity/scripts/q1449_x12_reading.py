"""q1449 (9 Oct 2026, READING, no verdict; track B step B3a): the same decomposition as q1448 but with the response x and the forcing bd of the TWELVE outermost atoms (positions and masses), each normalised at the outermost position index, to see how far from the edge the shape of x is K-independent and how it grows inward. Original docstring of q1448 (track B step 2): splits the six edge terms c_k = -x_k bd_k of the innermost flow into the adjoint response x_k (row of the inverse transposed Jacobian of the innermost atom, in the basis D1 of q1389) and the forcing bd_k, for the training rows K = 83 to 108, and stores x and bd at the six edge indices normalised by their value at the outermost position-type index. Question: does the forcing bd (a half-line, edge-local quantity) stay K-independent while the whole K^(-2/3) drift sits in the response x (the bulk transfer)? Rows above K = 108 are refused (blind gate q1446). Output q1448_xbd.json. Usage: py q1449_x12_reading.py (from temporal_within/small_gap_q206/)."""

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
    xs = [x[j] for j in pos]
    bs = [bd[j] for j in pos]
    ex = [m - 1 - k for k in range(12)]
    mx = [2 * m - 1 - k for k in range(12)]
    xr0 = x[m - 1]
    br0 = bd[m - 1]
    return dict(
        K=d["K"],
        parity="odd" if odd else "even",
        m=m,
        n=n,
        flow=float(ref),
        control_rel=float(rel),
        x_six=[float(v / xs[2]) for v in xs],
        bd_six=[float(v / bs[2]) for v in bs],
        x_ref=float(xs[2]),
        bd_ref=float(bs[2]),
        xpos12=[float(x[j] / xr0) for j in ex],
        xmass12=[float(x[j] / xr0) for j in mx],
        bdpos12=[float(bd[j] / br0) for j in ex],
        bdmass12=[float(bd[j] / br0) for j in mx],
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
        json.dump(out, open(os.path.join(HERE, "q1449_x12.json"), "w"), indent=1)
