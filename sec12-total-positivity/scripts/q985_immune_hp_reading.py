"""q985 (7 Oct 2026): READING (no verdict), the repaired instrument for the immune rows 500 and 580 where T80 had X2 False: a text copy of the IMP class of q579 with the quadrature nodes and weights at 215 digits and the arithmetic at 700 bits,
Newton polish from the stored doubles with the finite-difference step 1e-90 to the residual 1e-120, Jacobian and dF/da with the step 1e-90, minors at 1400 bits. The discrete problem is the same Gauss-Legendre problem (degree 9) with nodes accurate to 215 digits instead of 60 (the stored states solve the 60-digit version; the two exact solutions differ by about 1e-60).
Prints per row: the polish residual, the number of initial minors of X (sigma = (-1)^(J+1)) that are not certainly positive, X1, X3. Usage: py q985_immune_hp_reading.py ROW [ROW ...]
"""

import json, sys, time
import mpmath as mp
from mpmath.calculus.quadrature import GaussLegendre
from flint import arb, arb_mat, ctx
import q973_extended_tp_test_hi as Q

PREC, DPS = 700, 215
mp.mp.dps = DPS
ctx.prec = PREC
t0 = time.time()
_GL = {}


def base_nodes(deg):
    if deg not in _GL:
        raw = GaussLegendre(mp.mp).calc_nodes(deg, mp.mp.prec)
        xs = [(x, w) for x, w in raw if x > 0]
        _GL[deg] = ([arb(mp.nstr(x, DPS)) for x, w in xs], [arb(mp.nstr(w, DPS)) for x, w in xs])
    return _GL[deg]


def f(d):
    return (-(d * d) / 2).exp()


class IMPH:
    def __init__(self, a, deg=9):
        self.a = arb(a)
        bx, bq = base_nodes(deg)
        self.x = [t * self.a for t in bx]
        self.Q = bq

    def P(self, p, v, c):
        return [
            sum((v[j] / 2) * (f(x - p[j]) + f(x + p[j])) for j in range(len(p)))
            + (c * f(x) if c is not None else arb(0))
            for x in self.x
        ]

    def gs(self, p, v, c, ys):
        P = self.P(p, v, c)
        C = sum(q / Pi for q, Pi in zip(self.Q, P))
        W = [q / (2 * Pi * Pi) for q, Pi in zip(self.Q, P)]
        out = []
        for y in ys:
            g0 = g1 = arb(0)
            for x, Wi in zip(self.x, W):
                dm, dp = x - y, x + y
                em, ep = f(dm), f(dp)
                g0 += Wi * (em + ep)
                g1 += Wi * (dm * em - dp * ep)
            out.append((g0 / C, g1 / C))
        return out, C

    def F(self, u, J, centre):
        p = u[:J]
        v = u[J : 2 * J] if centre else u[J : 2 * J - 1] + [arb(1) - sum(u[J : 2 * J - 1])]
        c = (arb(1) - sum(v)) if centre else None
        gp, C = self.gs(p, v, c, p)
        eq = [g1 for g0, g1 in gp] + [g0 - 1 for g0, g1 in gp[:-1]]
        if centre:
            eq.append(self.gs(p, v, c, [arb(0)])[0][0][0] - 1)
        return eq


def jac(I, u, J, centre, h):
    n = len(u)
    Jm = arb_mat(n, n)
    for k in range(n):
        up = list(u)
        um = list(u)
        up[k] += h
        um[k] -= h
        Fp = I.F(up, J, centre)
        Fm = I.F(um, J, centre)
        for i in range(n):
            Jm[i, k] = ((Fp[i] - Fm[i]) / (2 * h)).mid()
    return Jm


rows = json.load(open("results_q584.json"))["rows"]
H = arb("1e-90")
for i in [int(x) for x in sys.argv[1:]]:
    r = rows[i]
    a = arb(float(r["a"]))
    centre = bool(r["centre"])
    I = IMPH(a)
    J = len(r["p"])
    u = [arb(float(t)) for t in r["p"]] + [arb(float(t)) for t in (r["v"] if centre else r["v"][:-1])]
    n = len(u)
    m = n - J
    for it in range(8):
        Fu = I.F(u, J, centre)
        res = max(abs(t.mid()) for t in Fu)
        print(
            f"   row {i} iteration {it}: residual {float(res.mid()):.2e} [{time.time() - t0:.0f}s]",
            flush=True,
        )
        if res < arb("1e-120"):
            break
        Jm = jac(I, u, J, centre, H)
        step = Jm.solve(arb_mat([[(-t).mid()] for t in Fu]))
        u = [(u[k] + step[k, 0]).mid() for k in range(n)]
    Jm = jac(I, u, J, centre, H)
    Fp = IMPH(a + H).F(u, J, centre)
    Fm = IMPH(a - H).F(u, J, centre)
    b = [((Fp[k] - Fm[k]) / (2 * H)).mid() for k in range(n)]
    A = arb_mat(J, J)
    B = arb_mat(J, m)
    C = arb_mat(m, J)
    Dm = arb_mat(m, m)
    for p in range(J):
        for q in range(J):
            A[p, q] = Jm[p, q]
        for q in range(m):
            B[p, q] = Jm[p, J + q]
    for p in range(m):
        for q in range(J):
            C[p, q] = Jm[J + p, q]
        for q in range(m):
            Dm[p, q] = Jm[J + p, J + q]
    Di = Dm.inv()
    S = A - B * Di * C
    b1 = arb_mat([[b[k]] for k in range(J)])
    b2 = arb_mat([[b[J + k]] for k in range(m)])
    be = b1 - B * Di * b2
    ctx.prec = 1400
    sg = 1 if J % 2 == 1 else -1
    Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(J)] for p in range(J)]
    vv_ = [sg * be[p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(J)]
    g = Q.gates(Tm, vv_, J)
    ctx.prec = PREC
    print(
        f"row {i}: a {float(a.mid()):.6f}, K {int(r['K'])}, J {J}, residual {float(res.mid()):.2e}: X1 {g['X1']}, X2 {g['X2']} ({g['bad']} not certainly positive of {g['count']}), X3 {g['X3']} [{time.time() - t0:.0f}s]",
        flush=True,
    )
print("END q985 READING (no verdict)")
