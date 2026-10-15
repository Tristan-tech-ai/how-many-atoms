"""Q579: 200-bit (python-flint arb) polish and centre signals for the SYMMETRIC uniform immune optimum (F = sum Q/Ptilde, uniform Q
on [-a, a], kernel exp(-d^2/2)), written 30 Sep 17:16:41 (date stamp), from PLAN_192BIT_IMMUNE.md. Needed because for a >= 24 the
optimum's KKT margins (1e-8..1e-9) sit at the double polish floor (811).
Quadrature: Gauss-Legendre of degree DEG (3 * 2^(DEG-1) nodes on [-1, 1], mpmath at 60 digits), folded to [0, a] by symmetry.
State: pairs +-p_j with pair masses v_j, optional centre mass c = 1 - sum v. Equations (q508c spolish's): g'(p_j)/C = 0 (all j),
g(p_j)/C - 1 = 0 (all but the last pair), and g(0)/C - 1 = 0 when a centre atom is present; g(y) = sum_i (Q_i/2)(f(x_i - y) +
f(x_i + y))/P_i^2, C = sum_i Q_i/P_i, Q_i = w_i/a. Newton on arb midpoints with a central-difference Jacobian (step 1e-30) and
backtracking. Signals in arb: s0 = g(0)/C - 1, s2 = g''(0)/C, and the off-support maximum of g/C - 1 on a 0.02 grid of [0, a + 3].
PRE-REGISTERED CONTROLS (before any use): (M1) from the certified double states at a = 20 (K = 24) and a = 23 (K = 28) the polish
reaches max |F| < 1e-35 and the positions agree with the double ones to 1e-10, the signals are negative and agree in sign with the
double ones; (M2) at a = 23, DEG 9 vs 10 changes the positions by < 1e-20.
v2 (17:44): Levenberg-Marquardt fallback when no Newton step fraction lowers max|F| (young families just past a split).
Usage: py q579_immune_mp.py control | py q579_immune_mp.py A SEEDJSON [DEG]"""

import sys, json, time
import numpy as np
import mpmath as mp
from flint import arb, arb_mat, ctx
from mpmath.calculus.quadrature import GaussLegendre

ctx.prec = 200
mp.mp.dps = 60
say = lambda *a: print(*a, flush=True)
t0 = time.time()
_GL = {}


def nodes(a, deg):
    if deg not in _GL:
        _GL[deg] = GaussLegendre(mp.mp).calc_nodes(deg, mp.mp.prec)
    xs = [(x, w) for x, w in _GL[deg] if x > 0]
    return [arb(mp.nstr(x, 60)) * arb(a) for x, w in xs], [
        arb(mp.nstr(w, 60)) for x, w in xs
    ]  # Q_i = w_i (a*w/a)


def f(d):
    return (-(d * d) / 2).exp()


class IMP:
    def __init__(self, a, deg=9):
        self.a = arb(a)
        self.x, self.Q = nodes(a, deg)

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
            g0 = g1 = g2 = arb(0)
            for x, Wi in zip(self.x, W):
                dm, dp = x - y, x + y
                em, ep = f(dm), f(dp)
                g0 += Wi * (em + ep)
                g1 += Wi * (dm * em - dp * ep)
                g2 += Wi * ((dm * dm - 1) * em + (dp * dp - 1) * ep)
            out.append((g0 / C, g1 / C, g2 / C))
        return out, C

    def F(self, u, J, centre):
        p = u[:J]
        v = u[J : 2 * J] if centre else u[J : 2 * J - 1] + [arb(1) - sum(u[J : 2 * J - 1])]
        c = (arb(1) - sum(v)) if centre else None
        gp, C = self.gs(p, v, c, p)
        eq = [g1 for g0, g1, g2 in gp] + [g0 - 1 for g0, g1, g2 in gp[:-1]]
        if centre:
            eq.append(self.gs(p, v, c, [arb(0)])[0][0][0] - 1)
        return eq

    def polish(self, p, v, centre, tol=1e-35, itmax=30):
        J = len(p)
        u = [arb(float(t)) for t in p] + [arb(float(t)) for t in (v if centre else v[:-1])]
        Fu = self.F(u, J, centre)
        r = max(abs(t.mid()) for t in Fu)
        for it in range(itmax):
            if r < arb(tol):
                break
            h = arb("1e-30")
            n = len(u)
            Jm = arb_mat(n, n)
            for k in range(n):
                up = list(u)
                um = list(u)
                up[k] += h
                um[k] -= h
                Fp = self.F(up, J, centre)
                Fm = self.F(um, J, centre)
                for i in range(n):
                    Jm[i, k] = ((Fp[i] - Fm[i]) / (2 * h)).mid()
            step = Jm.solve(arb_mat([[(-t).mid()] for t in Fu]))
            lam = arb(1)
            moved = False
            for _ in range(30):
                un = [u[i] + lam * step[i, 0].mid() for i in range(n)]
                Fn = self.F(un, J, centre)
                rn = max(abs(t.mid()) for t in Fn)
                if rn < r:
                    u, Fu, r, moved = un, Fn, rn, True
                    break
                lam /= 2
            if not moved:  # v2: Levenberg-Marquardt fallback (as q491)
                JT = Jm.transpose()
                JTJ = JT * Jm
                g = JT * arb_mat([[t.mid()] for t in Fu])
                mu = arb("1e-6")
                for _ in range(40):
                    A = arb_mat(n, n)
                    for i in range(n):
                        for k in range(n):
                            A[i, k] = JTJ[i, k] + (mu * JTJ[i, i] if i == k else 0)
                    dv = A.solve(g)
                    un = [u[i] - dv[i, 0].mid() for i in range(n)]
                    Fn = self.F(un, J, centre)
                    rn = max(abs(t.mid()) for t in Fn)
                    if rn < r:
                        u, Fu, r, moved = un, Fn, rn, True
                        break
                    mu *= 10
            if not moved:
                break
        p = u[:J]
        v = u[J : 2 * J] if centre else u[J : 2 * J - 1] + [arb(1) - sum(u[J : 2 * J - 1])]
        return p, v, r

    def signals(self, p, v, centre):
        c = (arb(1) - sum(v)) if centre else None
        g0c = self.gs(p, v, c, [arb(0)])[0][0]
        s0 = g0c[0] - 1
        s2 = g0c[2]
        pf = [float(t.mid()) for t in p]
        grid = [
            y
            for y in np.arange(0, float(self.a.mid()) + 3, 0.02)
            if min([abs(y - q) for q in pf] + ([y] if centre else [1e9])) > 0.05
        ]
        gg, C = self.gs(p, v, c, [arb(y) for y in grid])
        off = max((t[0] - 1).mid() for t in gg)
        return s0, s2, off


def seed_from(js):
    r = json.load(open(js))
    ys = np.array(r["ys"])
    ws = np.array(r["ws"])
    centre = bool(np.any(np.abs(ys) < 1e-9))
    o = ys > 1e-9
    p = np.sort(ys[o])
    v = 2 * ws[o][np.argsort(ys[o])]
    return p, v, centre


if __name__ == "__main__":
    if sys.argv[1] == "control":
        for a, js in ((20.0, "results_q535_unif0.0_a20.0.json"), (23.0, "results_q556_unif0.0_a23.0.json")):
            p0, v0, centre = seed_from(js)
            M = IMP(a, 9)
            p, v, r = M.polish(list(p0), list(v0), centre)
            dp = max(abs(float(p[j].mid()) - p0[j]) for j in range(len(p0)))
            s0, s2, off = M.signals(p, v, centre)
            say(
                f"M1 a = {a}: K = {2*len(p0) + int(centre)}, max|F| = {float(r):.1e}, max |p_arb - p_double| = {dp:.1e}, s0 {float(s0.mid()):+.3e}, s2 {float(s2.mid()):+.3e}, off {float(off):+.3e}   [{time.time() - t0:.0f}s]"
            )
            if a == 23.0:
                M10 = IMP(a, 10)
                p10, v10, r10 = M10.polish([float(t.mid()) for t in p], [float(t.mid()) for t in v], centre)
                say(
                    f"M2 a = 23: DEG 9 vs 10: max position change {max(abs(float((p10[j] - p[j]).mid())) for j in range(len(p))):.1e}, max|F| (10) {float(r10):.1e}   [{time.time() - t0:.0f}s]"
                )
