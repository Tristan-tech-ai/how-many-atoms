"""d22_order_n_numeric (8 Oct 2026): NUMERIC port of d20_order_n.py (the symbolic derivation of the centre-series coefficients e_n of amendments 1 Oct, e2 = 4/3 + 8 pi^2/45 ... e5 = -4254.456).
Same algorithm and same equations as d20_order_n.py; the exact sympy algebra (which reached 19.7 GB at order 6, amendment 1022) is replaced by polynomials in (z, C) and series in (delta, W) with mpmath complex coefficients (default 80 digits).
Control (before any e6): the program must reproduce e2 = 3.0879296713, e3 = 50.783470776, e4 = 535.47708947 and e5 = -4254.4559505 (d20_out_5.json / CENTRE_E2_DERIVATION.md section 0) to 1e-12 relative.
Usage: py d22_order_n_numeric.py NMAX [DPS]   Output d22_out_<NMAX>.json (e_n as strings with DPS digits)."""

import sys, json, time
import mpmath as mp

T0 = time.time()
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DPS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
mp.mp.dps = DPS
PI = mp.pi
I = mp.mpc(0, 1)
EPS = mp.mpf(10) ** (-(DPS - 25))
say = lambda *a: print(*a, flush=True)


# ---------------------------------------------------------------- polynomials in (z, C): dict {(a, b): coefficient}
def padd(p, q, s=1):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0) + s * v
    return {k: v for k, v in r.items() if abs(v) > EPS}


def pscale(p, c):
    return {k: c * v for k, v in p.items() if abs(c * v) > EPS}


def pmul(p, q):
    r = {}
    for (a, b), v in p.items():
        for (c, d), w in q.items():
            r[(a + c, b + d)] = r.get((a + c, b + d), 0) + v * w
    return {k: v for k, v in r.items() if abs(v) > EPS}


def D(p):
    """d/dz total derivative with C' = -pi (C^2 - 1)"""
    r = {}
    for (a, b), v in p.items():
        if a:
            r[(a - 1, b)] = r.get((a - 1, b), 0) + a * v
        if b:
            r[(a, b + 1)] = r.get((a, b + 1), 0) - PI * b * v
            r[(a, b - 1)] = r.get((a, b - 1), 0) + PI * b * v
    return {k: v for k, v in r.items() if abs(v) > EPS}


def zC(a, b, c=1):
    return {(a, b): mp.mpmathify(c)}


# ---------------------------------------------------------------- the saddle line: z = xi + i, C = c (xi, c real): real polynomial in (xi, c)
def saddle(p):
    re, im = {}, {}
    for (a, b), v in p.items():
        for k in range(a + 1):
            w = v * mp.binomial(a, k) * I ** (a - k)
            re[(k, b)] = re.get((k, b), 0) + w.real
            im[(k, b)] = im.get((k, b), 0) + w.imag
    return {k: v for k, v in re.items() if abs(v) > EPS}, {k: v for k, v in im.items() if abs(v) > EPS}


def Lz(k):
    """Re (xi + i)^k - xi^k as dict {power of xi: coefficient}"""
    r = {}
    for j in range(2, k + 1, 2):
        r[k - j] = r.get(k - j, 0) + mp.binomial(k, j) * (-1) ** (j // 2)
    return r


def solve_poly(p):
    """p: dict {power of xi: real coefficient}; returns Q(z) = dict {power: coefficient} with L[Q] = p"""
    rem = dict(p)
    Q = {}
    while any(abs(v) > EPS for v in rem.values()):
        d = max(k for k, v in rem.items() if abs(v) > EPS)
        lc = rem[d]
        k = d + 2
        lk = Lz(k)[d]
        coef = lc / lk
        Q[k] = Q.get(k, 0) + coef
        for kk, v in Lz(k).items():
            rem[kk] = rem.get(kk, 0) - coef * v
        rem = {kk: v for kk, v in rem.items() if abs(v) > EPS}
    return Q


def build(source):
    """source: real polynomial in (xi, c) as dict {(xi power, c power): coefficient}; eta = sum_n solve_poly(coef of c^n) C^n"""
    byc = {}
    for (k, n), v in source.items():
        byc.setdefault(n, {})[k] = v
    eta = {}
    for n, cf in byc.items():
        for kz, v in solve_poly(cf).items():
            if abs(v) > EPS:
                eta[(kz, n)] = eta.get((kz, n), 0) + v
    return eta


# ---------------------------------------------------------------- Laurent series of coth(pi w): dict {power: coefficient}
NS = 2 * NMAX + 8


def coth_series(N):
    s = {}
    for k in range(0, N // 2 + 3):
        pw = 2 * k - 1
        if pw > N:
            break
        s[pw] = mp.mpf(2) ** (2 * k) * mp.bernoulli(2 * k) * PI**pw / mp.factorial(2 * k)
    return s


def lmul(a, b, N):
    r = {}
    for i, u in a.items():
        for j, w in b.items():
            if i + j <= N:
                r[i + j] = r.get(i + j, 0) + u * w
    return r


NB = 6 * NMAX + 60  # series order of the coth powers (the powers needed stay far below NB - b)
CS = coth_series(NB)
CPOW = {0: {0: mp.mpf(1)}}


def cpow(m):
    while m not in CPOW:
        k = max(CPOW) + 1
        CPOW[k] = lmul(CPOW[k - 1], CS, NB)
    return CPOW[m]


def laurent_at0(eta, kmin):
    out = {k: 0 for k in range(kmin, 0)}
    for (a, b), v in eta.items():
        for pw, c in cpow(b).items():
            k = a + pw
            if kmin <= k <= -1:
                out[k] += v * c
    return out


def laurent_ati(eta, kmin, kmax):
    out = {k: 0 for k in range(kmin, kmax + 1)}
    for (a, b), v in eta.items():
        for j in range(a + 1):  # z^a = (i + t)^a = sum_j binom(a, j) i^(a-j) t^j
            w = v * mp.binomial(a, j) * I ** (a - j)
            for pw, c in cpow(b).items():
                k = j + pw
                if kmin <= k <= kmax:
                    out[k] += w * c
    return out


# ---------------------------------------------------------------- Cole-Hopf source (lists of polynomials in (z, C), index = power of the order parameter)
def lpadd(*ls):
    N = max(len(l) for l in ls)
    out = [dict() for _ in range(N)]
    for l in ls:
        for k, v in enumerate(l):
            out[k] = padd(out[k], v)
    return out


def lpmul(a, b, N):
    out = [dict() for _ in range(N + 1)]
    for j, aj in enumerate(a):
        if not aj:
            continue
        for k, bk in enumerate(b):
            if j + k <= N and bk:
                out[j + k] = padd(out[j + k], pmul(aj, bk))
    return out


def lpD(a):
    return [D(v) if v else dict() for v in a]


def lpscale(c, a):
    return [pscale(v, c) for v in a]


def source_n(etas_p, n_want):
    NM = n_want
    p = [pscale(e, -I * PI) for e in etas_p][: NM + 1]
    p += [dict()] * (NM + 1 - len(p))
    u, up = {}, {0: p}
    u[1] = lpadd(lpD(p), lpmul(p, p, NM - 1))[:NM]
    up[1] = lpD(u[1])
    for n in range(2, NM + 1):
        N = NM - n
        s = lpD(up[n - 1])[: N + 1]
        for j in range(0, n):
            s = lpadd(s, lpmul(up[j], up[n - 1 - j], N))
        u[n] = lpscale(mp.mpf(1) / n, s[: N + 1])
        up[n] = lpD(u[n])
    tot = {}
    for n in range(1, NM + 1):
        k = NM - n
        if k < len(u[n]):
            tot = padd(tot, u[n][k])
    re, im = saddle(tot)
    return {k: v / PI for k, v in im.items()}


# ---------------------------------------------------------------- series in delta (lists of complex numbers) and in (delta, W) (dicts)
LEN = 2 * NMAX + 2


def smul(a, b):
    r = [mp.mpc(0)] * LEN
    for i, u in enumerate(a):
        if u == 0:
            continue
        for j, w in enumerate(b):
            if i + j < LEN:
                r[i + j] += u * w
    return r


def sadd(a, b, s=1):
    return [x + s * y for x, y in zip(a, b)]


def spow(a, m):
    r = [mp.mpc(1)] + [mp.mpc(0)] * (LEN - 1)
    for _ in range(m):
        r = smul(r, a)
    return r


def bmul(p, q, order):
    r = {}
    for (d1, w1), u in p.items():
        for (d2, w2), w in q.items():
            if d1 + d2 <= order:
                r[(d1 + d2, w1 + w2)] = r.get((d1 + d2, w1 + w2), 0) + u * w
    return r


def badd(p, q, s=1):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0) + s * v
    return r


def bexp(p, order):
    """exp of a bivariate series with no delta^0 terms"""
    r = {(0, 0): mp.mpc(1)}
    term = {(0, 0): mp.mpc(1)}
    for m in range(1, order + 1):
        term = bmul(term, p, order)
        term = {k: v / m for k, v in term.items()}
        if not term:
            break
        r = badd(r, term)
    return r


# ---------------------------------------------------------------- state
eta_c = {1: padd(padd(pscale(zC(2, 1), 2), pscale(zC(3, 2), 2 * PI)), pscale(zC(3, 0), 2 * PI / 3))}
etap = {0: pscale(zC(1, 1), 2), 1: D(eta_c[1])}
A = {0: mp.mpf(2)}
Bc = {1: mp.mpf(-4)}
e_n = {1: mp.mpf(0)}
c0 = {}
L1c = laurent_ati(eta_c[1], -4, 2 * NMAX)
# eta0' = 2 (i + t) coth(pi t) near z = i: coefficient of t^(k-1), divided by k
e0t = {}
for pw, c in CS.items():
    e0t[pw] = e0t.get(pw, 0) + 2 * I * c
    e0t[pw + 1] = e0t.get(pw + 1, 0) + 2 * c
regc = {
    0: {k: e0t.get(k - 1, 0) / k for k in range(1, 2 * NMAX + 1)},
    1: {k: L1c[k] for k in range(1, 2 * NMAX)},
}


def a2_series():
    return [A.get(k, 0) for k in range(LEN)]


def b_series():
    return [Bc.get(k, 0) for k in range(LEN)]


def inner_solve(order):
    a2 = [mp.mpc(x) for x in a2_series()]
    bb = [mp.mpc(x) for x in b_series()]
    s = {}
    for n, ck in regc.items():
        for k, v in ck.items():
            if 2 * n + k <= order:
                s[(2 * n + k, k)] = s.get((2 * n + k, k), 0) + v
    ex = {kk: -I * PI * v for kk, v in s.items()}
    lin = [mp.mpc(0)] * LEN
    for (d, w), v in ex.items():
        if w == 1:
            lin[d] += v
    rest = {kk: v for kk, v in ex.items() if kk[1] != 1}
    # Wsh = W + 2 lin ; rest(Wsh)
    twolin = {(d, 0): 2 * v for d, v in enumerate(lin) if v != 0}
    restsh = {}
    for (d, w), v in rest.items():
        for j in range(w + 1):
            t = {(d, j): v * mp.binomial(w, j)}
            for _ in range(w - j):
                t = bmul(t, twolin, order)
            restsh = badd(restsh, t)
    ibb = {(d, 0): -I * v for d, v in enumerate(bb) if v != 0}
    base = badd({(0, 1): mp.mpc(1)}, badd(twolin, ibb))  # Wsh - i bb
    P = badd(bmul(base, base, order), {(d, 0): v for d, v in enumerate(a2) if v != 0}, -1)
    body = bmul(P, bexp({kk: v for kk, v in restsh.items()}, order), order)
    avg = {}
    for (d, w), v in body.items():
        if w % 2 == 0:
            avg[d] = avg.get(d, 0) + v * (mp.fac2(w - 1) if w > 0 else 1) * mp.mpf(2) ** (w // 2)
    l2 = smul(lin, lin)
    ex2 = [mp.mpc(1)] + [mp.mpc(0)] * (LEN - 1)
    term = ex2[:]
    for m in range(1, order + 1):
        term = [x / m for x in smul(term, l2)]
        ex2 = sadd(ex2, term)
    av = [avg.get(d, 0) for d in range(LEN)]
    g0 = smul(av, ex2)
    return g0[order], [g0[k] for k in range(order)]  # A_order = g0[order] with A_order = 0 in a2


def wsum(k2):
    a2 = [mp.mpc(x) for x in a2_series()]
    bb = [mp.mpc(x) for x in b_series()]
    ib = [I * x for x in bb]
    out = [mp.mpc(0)] * LEN
    for j in range(0, k2 + 1, 2):
        out = sadd(out, [2 * mp.binomial(k2, j) * x for x in smul(spow(a2, j // 2), spow(ib, k2 - j))])
    return [x / k2 for x in out]


def complete(eta_p, req, nC, kmin):
    """eta = eta_p + sum_k g_k C^(2k+1) + z sum_k al_k (C^2 - 1)^(k+1); equations: L0[-(2k+1)] = 0 (k < nC), Li[-o] = req[o]"""
    basis = []
    for k in range(nC):
        basis.append({(0, 2 * k + 1): mp.mpf(1)})
    for k in range(len(req)):
        q = {(0, 0): mp.mpf(1)}
        for _ in range(k + 1):
            q = pmul(q, {(0, 2): mp.mpf(1), (0, 0): mp.mpf(-1)})
        basis.append(pmul({(1, 0): mp.mpf(1)}, q))
    L0p = laurent_at0(eta_p, kmin)
    Lip = laurent_ati(eta_p, kmin, -1)
    rows = [[None] * len(basis) for _ in range(nC + len(req))]
    rhs = []
    L0b = [laurent_at0(b, kmin) for b in basis]
    Lib = [laurent_ati(b, kmin, -1) for b in basis]
    for r in range(nC):
        for c in range(len(basis)):
            rows[r][c] = L0b[c][-(2 * r + 1)]
        rhs.append(-L0p[-(2 * r + 1)])
    for r2, (o, val) in enumerate(sorted(req.items())):
        r = nC + r2
        for c in range(len(basis)):
            rows[r][c] = Lib[c][-o]
        rhs.append(val - Lip[-o])
    sol = mp.lu_solve(mp.matrix(rows), mp.matrix(rhs))
    eta = dict(eta_p)
    for c, b in enumerate(basis):
        eta = padd(eta, pscale(b, sol[c]))
    return eta


out = {"e1": "0"}
for n in range(2, NMAX + 1):
    etas_p = [etap[k] for k in range(0, n)]
    Sn = source_n(etas_p, n)
    ep = build(Sn)
    kmin = -(2 * n + 3)
    L0 = laurent_at0(ep, kmin)
    Li = laurent_ati(ep, kmin, 2 * NMAX)
    sing = {k: Li[k] - L0.get(k, 0) for k in range(kmin, 0)}
    far = {}
    for (a, b), v in ep.items():
        far[a] = far.get(a, 0) + v  # C = 1
    const = far.get(0, 0)
    r1 = L0[-1]
    dn = sing[-2]
    say(
        f"order {n}: S_n {len(Sn)} terms; Re d_n = {mp.nstr(dn.real, 6)}; bulk s_2n = {mp.nstr(far.get(2, 0), 15)} [{time.time() - T0:.0f} s]"
    )
    Bc[2 * n - 1] = PI / 2 * sing[-1]
    An, lower = inner_solve(2 * n - 2)
    A[2 * n - 2] = An
    say(
        f"   b_{2 * n - 1} = {mp.nstr(Bc[2 * n - 1], 15)}; A_{2 * n - 2} = {mp.nstr(An, 15)}; max |lower orders of Gt(0)| = {mp.nstr(max([abs(x) for x in lower] + [0]), 5)} [{time.time() - T0:.0f} s]"
    )
    a2s = a2_series()
    bs = b_series()
    amb = sadd([mp.mpc(x) for x in a2s], smul([mp.mpc(x) for x in bs], [mp.mpc(x) for x in bs]), -1)
    kap = amb[2 * n - 2]
    en = const - PI * r1 + kap + PI * dn.imag
    e_n[n] = en
    say(f"   kappa_{n} = {mp.nstr(kap, 15)}; e_{n} = {mp.nstr(en, 25)}")
    out[f"e{n}"] = mp.nstr(en.real if isinstance(en, mp.mpc) else en, DPS - 10)
    out[f"e{n}_imag"] = mp.nstr(en.imag if isinstance(en, mp.mpc) else 0, 5)
    if n == NMAX:
        break
    req = {}
    for k in range(1, n + 1):
        req[2 * k] = -(I / PI) * wsum(2 * k)[2 * n - 2 * k]
    eta_n = complete(ep, req, n + 1, kmin)
    chk = (
        sum(v for (a, b), v in eta_n.items() if a == 0)
        if False
        else sum(v for (a, b), v in eta_n.items() if a == 0)
    )
    say(
        f"   completed eta_{n}: far-field constant minus e_{n} = {mp.nstr(chk - en, 8)} (must be 0) [{time.time() - T0:.0f} s]"
    )
    eta_c[n] = eta_n
    etap[n] = D(eta_n)
    Ln = laurent_ati(eta_n, kmin, 2 * NMAX)
    regc[n] = {k: Ln[k] for k in range(1, 2 * NMAX - 2 * n + 1)}
json.dump(out, open(f"d22_out_{NMAX}.json", "w"), indent=1)
say("done", {k: v for k, v in out.items() if not k.endswith("imag")}, f"[{time.time() - T0:.0f} s]")
