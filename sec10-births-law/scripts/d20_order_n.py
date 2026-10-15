"""d20_order_n: the offset coefficients e_n (delta = sum_n e_n v^(-2n)) to order NMAX by one generic recursion (helpers are text
copies of d18_fourth_order.py). Per order n >= 2:
  S_n  = (1/pi) Im [E^n coefficient of the Cole-Hopf series U] on the saddle line, from eta_0 .. eta_{n-1} (complete);
  eta_n^p = polynomial-algebra particular solution; r1 (zeta^-1 at 0), d_n (t^-2 at i after regularisation), const at C = 1;
  b_{2n-1} = (pi/2) [t^-1 coefficient of eta_n at i]   (visible dipole = zero pair's vertical position);
  inner: Gt(0) = <P(W) exp(-i pi eta_reg(i + delta W))> = 0 to delta^(2n-2) gives A_{2n-2} (a^2 = 2 + sum A_k delta^k);
  kappa_n = [a^2 - b^2]_{delta^(2n-2)};  e_n = const - pi r1 + kappa_n + pi Im d_n;
  completion of eta_n: invisible t^-2k coefficients at i set to the zero pair's -(i/pi)[(w1^2k + w2^2k)/(2k)]_{delta^(2n-2k)}.
Usage: py d20_order_n.py NMAX"""

import sympy as sp, json, time, sys

T0 = time.time()
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
z, C, t, dl, E, Wv = sp.symbols("z C t delta E W")
xi_r, c_r = sp.symbols("xi_r c_r", real=True)
PI = sp.pi
I = sp.I
say = lambda *a: print(*a, flush=True)


def D(F):
    return sp.expand(sp.diff(F, z) + sp.diff(F, C) * (-PI * (C**2 - 1)))


def at_saddle2(F):
    G = sp.expand(F.subs(z, xi_r + I).subs(C, c_r))
    re, im = G.as_real_imag()
    return sp.expand(re), sp.expand(im)


def Lz(k):
    return sp.expand(sp.re(sp.expand((xi_r + I) ** k)) - xi_r**k)


def solve_poly(p):
    rem = sp.expand(p)
    Q = 0
    while rem != 0:
        pr = sp.Poly(rem, xi_r)
        d = pr.degree()
        lc = pr.LC()
        k = d + 2
        lk = sp.Poly(Lz(k), xi_r).LC()
        coef = sp.simplify(lc / lk)
        Q += coef * z**k
        rem = sp.expand(rem - coef * Lz(k))
    return sp.expand(Q)


def build(source):
    src = sp.Poly(sp.expand(source), c_r)
    eta = 0
    for (n,), coef in src.terms():
        eta += solve_poly(coef) * C**n
    return sp.expand(eta)


NS = 2 * NMAX + 8
Cser = lambda var, N: sp.series(sp.coth(PI * var), var, 0, N).removeO()
CS_z = Cser(z, NS)
CS_t = Cser(t, NS)


def laurent_at0(eta, kmin):
    e = sp.expand(eta.subs(C, CS_z))
    return {k: sp.simplify(e.coeff(z, k)) for k in range(kmin, 0)}


def laurent_ati(eta, kmin, kmax):
    e = sp.expand(eta.subs(z, I + t).subs(C, CS_t))
    return {k: sp.simplify(e.coeff(t, k)) for k in range(kmin, kmax + 1)}


def derivs(F, n):
    out = [F]
    for k in range(n):
        out.append(D(out[-1]))
    return out


def complete(eta_p, req, nC, kmin):
    gs = sp.symbols(f"g1:{nC + 1}")
    als = sp.symbols(f"al1:{len(req) + 1}")
    eta = sp.expand(
        eta_p
        + sum(gk * C ** (2 * k + 1) for k, gk in enumerate(gs))
        + z * sum(ak * (C**2 - 1) ** (k + 1) for k, ak in enumerate(als))
    )
    L0 = laurent_at0(eta, kmin)
    Li = laurent_ati(eta, kmin, -1)
    eqs = [L0[-(2 * k + 1)] for k in range(nC)] + [sp.expand(Li[-o] - val) for o, val in req.items()]
    sol = sp.solve(eqs, list(gs) + list(als), dict=True)[0]
    return sp.expand(eta.subs(sol)), sol


def mul(a, b, N):
    out = [0] * (N + 1)
    for j, aj in enumerate(a):
        if aj == 0:
            continue
        for k, bk in enumerate(b):
            if j + k <= N and bk != 0:
                out[j + k] += aj * bk
    return [sp.expand(o) for o in out]


def add(*ls):
    N = max(len(l) for l in ls)
    out = [0] * N
    for l in ls:
        for k, v in enumerate(l):
            out[k] += v
    return [sp.expand(o) for o in out]


def Dl(a):
    return [D(v) if v != 0 else 0 for v in a]


def scal(c, a):
    return [sp.expand(c * v) for v in a]


def source_n(etas_p, n_want):
    """(1/pi) Im [E^n_want coefficient of U] on the saddle line; etas_p = [eta0', eta1', ...]"""
    NM = n_want
    p = [sp.expand(-I * PI * e) for e in etas_p][: NM + 1]
    p += [0] * (NM + 1 - len(p))
    u, up = {}, {0: p}
    u[1] = add(Dl(p), mul(p, p, NM - 1))[:NM]
    up[1] = Dl(u[1])
    for n in range(2, NM + 1):
        N = NM - n
        s = Dl(up[n - 1])[: N + 1]
        for j in range(0, n):
            s = add(s, mul(up[j], up[n - 1 - j], N))
        u[n] = scal(sp.Rational(1, n), s[: N + 1])
        up[n] = Dl(u[n])
    tot = 0
    for n in range(1, NM + 1):
        k = NM - n
        if k < len(u[n]):
            tot += u[n][k]
    re, im = at_saddle2(sp.expand(tot))
    return sp.expand(im / PI)


# ------------------------------------------------ state
eta_c = {1: 2 * z**2 * C + 2 * PI * z**3 * C**2 + sp.Rational(2, 3) * PI * z**3}  # complete eta_n
etap = {
    0: 2 * z * C,
    1: D(eta_c[1]) if False else sp.expand(sp.diff(eta_c[1], z) + sp.diff(eta_c[1], C) * (-PI * (C**2 - 1))),
}
A = {0: sp.Integer(2)}
Bc = {1: sp.Integer(-4)}  # a^2 = sum A_k delta^k, b = sum B_k delta^k
e_n = {1: sp.Integer(0)}
e0p_t = sp.expand(2 * (I + t) * CS_t)
c0k = {k: sp.simplify(e0p_t.coeff(t, k - 1) / k) for k in range(1, 2 * NMAX + 1)}
regc = {0: c0k}  # regular Laurent coefficients at i: regc[n][k] (t^k), k >= 1
L1 = laurent_ati(eta_c[1], -4, 2 * NMAX)
regc[1] = {k: L1[k] for k in range(1, 2 * NMAX)}


def a2_series():
    return sum(v * dl**k for k, v in A.items())


def b_series():
    return sum(v * dl**k for k, v in Bc.items())


def inner_solve(order):
    """solve Gt(0) = 0 at delta^order for A_order (A_k for k < order already known)"""
    As = sp.Symbol("Aunk")
    a2 = a2_series() + As * dl**order
    bb = b_series()
    s = 0
    for n, ck in regc.items():
        for k, v in ck.items():
            if 2 * n + k <= order:
                s += dl ** (2 * n) * v * (dl * Wv) ** k
    ex = sp.expand(-I * PI * s)
    lin = ex.coeff(Wv, 1)
    rest = sp.expand(ex - lin * Wv)
    Wsh = Wv + 2 * lin
    P = (Wsh - I * bb) ** 2 - a2
    body = sp.expand(
        sp.series(
            sp.expand(P * sp.series(sp.exp(rest.subs(Wv, Wsh)), dl, 0, order + 1).removeO()), dl, 0, order + 1
        ).removeO()
    )
    poly = sp.Poly(body, Wv)
    avg = 0
    for (k,), cf in poly.terms():
        if k % 2 == 0:
            avg += cf * sp.factorial2(k - 1) * 2 ** (k // 2) if k > 0 else cf
    g0 = sp.expand(
        sp.series(
            avg * sp.series(sp.exp(sp.expand(lin**2)), dl, 0, order + 1).removeO(), dl, 0, order + 1
        ).removeO()
    )
    sol = sp.solve(g0.coeff(dl, order), As)
    return sp.simplify(sol[0]), [sp.simplify(g0.coeff(dl, k)) for k in range(0, order)]


def wsum(k2):
    """(w1^k2 + w2^k2)/k2 for w1,2 = +-a + i b, as a series in delta"""
    a2 = a2_series()
    bb = b_series()
    s = 0
    for j in range(0, k2 + 1, 2):
        s += 2 * sp.binomial(k2, j) * a2 ** (j // 2) * (I * bb) ** (k2 - j)
    return sp.expand(s / k2)


out = {"e1": "0"}
for n in range(2, NMAX + 1):
    # source from complete lower orders
    etas_p = [etap[k] for k in range(0, n)]
    Sn = source_n(etas_p, n)
    ep = build(Sn)
    kmin = -(2 * n + 3)
    L0 = laurent_at0(ep, kmin)
    Li = laurent_ati(ep, kmin, 2 * NMAX)
    sing = {k: sp.simplify(Li[k] - L0.get(k, 0)) for k in range(kmin, 0)}
    far = sp.expand(ep.subs(C, 1))
    const = far.coeff(z, 0)
    r1 = L0[-1]
    dn = sing[-2]
    say(
        f"order {n}: S_n {len(sp.Add.make_args(Sn))} terms; far field xi^2 coeff (bulk s_2n) = {sp.simplify(far.coeff(z, 2))} = {sp.N(far.coeff(z, 2), 12)}; Re d_n = {sp.simplify(sp.re(dn))} [{time.time() - T0:.0f} s]"
    )
    Bc[2 * n - 1] = sp.simplify(PI / 2 * sing[-1])
    An, lower = inner_solve(2 * n - 2)
    A[2 * n - 2] = An
    say(
        f"   b_{2*n - 1} = {Bc[2*n - 1]}; A_{2*n - 2} = {An}; lower orders of Gt(0) (must be 0): {[l for l in lower if l != 0]} [{time.time() - T0:.0f} s]"
    )
    amb = sp.expand(a2_series() - b_series() ** 2)
    kap = sp.simplify(amb.coeff(dl, 2 * n - 2))
    en = sp.simplify(const - PI * r1 + kap + PI * sp.im(dn))
    e_n[n] = en
    say(f"   kappa_{n} = {kap}; e_{n} = {en} = {sp.N(en, 15)}")
    out[f"e{n}"] = str(en)
    out[f"e{n}_num"] = float(sp.N(en, 20))
    if n == NMAX:
        break
    # complete eta_n
    req = {}
    for k in range(1, n + 1):
        req[2 * k] = sp.simplify(-(I / PI) * wsum(2 * k).coeff(dl, 2 * n - 2 * k))
    eta_n, sol = complete(ep, req, n + 1, kmin)
    chk = sp.simplify(sp.expand(eta_n.subs(C, 1)).coeff(z, 0) - en)
    say(
        f"   completed eta_{n}: far-field constant minus e_{n} = {chk} (must be 0) [{time.time() - T0:.0f} s]"
    )
    eta_c[n] = eta_n
    etap[n] = D(eta_n)
    Ln = laurent_ati(eta_n, kmin, 2 * NMAX)
    regc[n] = {k: Ln[k] for k in range(1, 2 * NMAX - 2 * n + 1)}
json.dump(out, open(f"d20_out_{NMAX}.json", "w"), indent=1)
say("done", out, f"[{time.time() - T0:.0f} s]")
