"""q1387 reading (9 Oct 2026, EXPLORATORY, no item, no verdict; py q1387_ift_innermost.py [K ...] from temporal_within/small_gap_q206/): implicit-function flow of the innermost quantity of the capacity at stored 40-digit states. System: the mirror-reduced KKT system F(z; A) = 0 of q206_gauss_mp (z = u_1..u_m, w_1..w_m, we [, wc]; rows D(u_j) - D(A), D'(u_j), normalisation [, D(0) - D(A)]). dz/dA = -J^-1 dF/dA with J from residual_and_jacobian and dF/dA a central difference in the edge position A at fixed grid (delta = 1e-12, 40 digits). The innermost entry (u_1 for even K, wc for odd K) is decomposed as sum_k c_k, c_k = -(J^-T e_i)_k (dF/dA)_k, by row block. Control (independent earlier form): the same entry from the central difference of the two neighbouring stored states of the same family (dA = 0.005); tolerance 2e-3 relative. Expected before the run: the IFT value equals the finite-difference flow within the control tolerance; the contributions are NOT sign-definite by block."""

import sys, os, json, glob, re, time
import q206_gauss_mp as G
from mpmath import mp, mpf, log, pi, matrix, lu_solve

mp.dps = 40
G.mp.dps = 40
G.L2P = log(2 * pi) / 2
ROOT = ".."


def load(K):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "**", f"results_q2*_gauss_K{K}_A*.json"), recursive=True):
        if "eps" in f:
            continue
        try:
            j = json.load(open(f))
            a = mpf(j["A"])
            res = float(j["residual"])
        except Exception:
            continue
        if j.get("verdict", "OPTIMAL") != "OPTIMAL" or res > 1e-20 or int(j["K"]) != K:
            continue
        out[round(float(a), 6)] = j
    return [out[k] for k in sorted(out)]


def vec(j):
    return (
        [mpf(x) for x in j["u"]],
        [mpf(x) for x in j["w"]],
        mpf(j["we"]),
        (None if j["wc"] is None else mpf(j["wc"])),
    )


def flat(j):
    u, w, we, wc = vec(j)
    return u + w + [we] + ([] if wc is None else [wc])


def analyse(K, st, label):
    j = st[1]
    jm, jp = st[0], st[2]
    A = mpf(j["A"])
    u, w, we, wc = vec(j)
    m = len(u)
    odd = wc is not None
    y = G.grid(A)
    t0 = time.time()
    r0, J = G.residual_and_jacobian(A, y, u, w, we, wc)
    dl = mpf("1e-12")

    def F(Av):
        r, _ = G.residual_and_jacobian(Av, y, u, w, we, wc, want_J=False)
        return r

    rp, rm = F(A + dl), F(A - dl)
    b = [(rp[i] - rm[i]) / (2 * dl) for i in range(len(r0))]
    dz = lu_solve(J, matrix([-t for t in b]))
    idx = (2 * m + 1) if odd else 0
    e = matrix(len(r0), 1)
    e[idx] = 1
    x = lu_solve(J.T, e)
    c = [-x[k] * b[k] for k in range(len(r0))]
    ift = dz[idx]
    chk = sum(c)
    zp, zm = flat(jp), flat(jm)
    fd = (zp[idx] - zm[idx]) / (mpf(jp["A"]) - mpf(jm["A"]))
    rel = abs(ift - fd) / abs(fd)
    q = wc if odd else u[0]
    blocks = [("D(u_j)-D(A)", 0, m), ("D'(u_j)", m, 2 * m), ("norm", 2 * m, 2 * m + 1)] + (
        [("D(0)-D(A)", 2 * m + 1, 2 * m + 2)] if odd else []
    )
    print(
        f"K {K} {label} A {mp.nstr(A, 9)} ({'odd, wc' if odd else 'even, u_1'}): entry {mp.nstr(q, 10)}; IFT flow {mp.nstr(ift, 12)}; neighbour FD {mp.nstr(fd, 12)}; rel diff {mp.nstr(rel, 3)} {'OK' if rel < mpf('2e-3') else 'CONTROL FAILED'}; sum of contributions {mp.nstr(chk, 12)}; relative rate {mp.nstr(ift / q, 8)} [{time.time() - t0:.0f}s]",
        flush=True,
    )
    rec = dict(
        K=K, label=label, A=str(A), ift=mp.nstr(ift, 15), fd=mp.nstr(fd, 15), rel=mp.nstr(rel, 4), blocks=[]
    )
    for nm, a, bnd in blocks:
        cs = c[a:bnd]
        pos = sum(t for t in cs if t > 0)
        neg = sum(t for t in cs if t < 0)
        big = max(range(len(cs)), key=lambda i: abs(cs[i]))
        sb = [1 if t > 0 else -1 for t in b[a:bnd] if t != 0]
        sx = [1 if x[a + i] > 0 else -1 for i in range(bnd - a)]
        print(
            f"     block {nm:12s}: sum {mp.nstr(sum(cs), 8):>14}, positive part {mp.nstr(pos, 8):>14}, negative part {mp.nstr(neg, 8):>14}, largest |c| at index {big + 1} of {len(cs)} ({mp.nstr(cs[big], 6)}); signs of b: {sb.count(1)}+/{sb.count(-1)}-; signs of x: {sx.count(1)}+/{sx.count(-1)}-"
        )
        rec["blocks"].append(
            dict(
                name=nm,
                sum=mp.nstr(sum(cs), 12),
                pos=mp.nstr(pos, 12),
                neg=mp.nstr(neg, 12),
                big=big + 1,
                n=len(cs),
                c=[mp.nstr(t, 8) for t in cs],
                b=[mp.nstr(t, 8) for t in b[a:bnd]],
                x=[mp.nstr(x[a + i], 8) for i in range(bnd - a)],
            )
        )
    return rec


Ks = [int(a) for a in sys.argv[1:]] or [62, 61]
allrec = []
for K in Ks:
    st = load(K)
    if len(st) < 5:
        print(f"K {K}: only {len(st)} stored states")
        continue
    A = [float(s["A"]) for s in st]
    trip = [i for i in range(1, len(st) - 1) if A[i + 1] - A[i - 1] <= 0.0101]
    print(
        f"K {K}: {len(st)} stored OPTIMAL states A {A[0]} to {A[-1]}; centred triples {len(trip)}", flush=True
    )
    if not trip:
        continue
    for lab, i in (("start", trip[0]), ("middle", trip[len(trip) // 2]), ("end", trip[-1])):
        allrec.append(analyse(K, st[i - 1 : i + 2], lab))
        json.dump(allrec, open("q1387_rows_" + "_".join(map(str, Ks)) + ".json", "w"))
print("READING q1387 done", len(allrec))
