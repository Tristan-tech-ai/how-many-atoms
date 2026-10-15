"""q986 (7 Oct 2026): verdict script of T80b (amendment 1757), the repaired replicate of T80 for the immune functional: T80 (q981) had X2 False at the rows 500 and 580 with the 200-bit polish (residual 4e-47 and 2e-47); the reading q985 shows both rows with X2 True after a polish to 1e-166.
Instrument: a text copy of the IMP class of q579 with the Gauss-Legendre nodes and weights at 215 digits and the arithmetic at 700 bits, Newton polish from the stored doubles (finite-difference step 1e-90) to the residual 1e-120 (VOID if the residual stays above 1e-100), Jacobian and dF/da by central differences with the step 1e-90, minors at 1400 bits.
Rows: the 15 rows of T80 (every 40th row of results_q584.json from index 20: 20, 60, ..., 580). Items per row: X1 all entries of v = sigma E b_eff certainly positive, sigma = (-1)^(J+1); X2 all J(J + 1) initial minors of X = [T | v] certainly positive (count checked); X3 the J maximal minors with v last certainly positive; X4 the opposite orientation fails X1.
Verdict (last line): PASS if X1 to X4 hold at all 15 rows; FAIL if a processed row fails an item; VOID if a row does not polish to 1e-100, a count differs or a control does not behave.
Controls (synthetic, J = 10, 1400 bits): PC Gaussian kernel with the continuation column passes X1 to X3; NC1 (sign of v[3] flipped) fails X2; NC2 (alternating v) fails X1 and X2.
Usage: py q986_immune_hp_test.py [--controls-only]   (from temporal_within/)"""

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


def controls(J=10):
    ctx.prec = 1400
    K = [[(arb(-1) / 10 * ((p - q) ** 2)).exp() for q in range(J)] for p in range(J)]
    v = [(arb(-1) / 10 * ((p - J) ** 2)).exp() for p in range(J)]
    pc = Q.gates(K, v, J)
    v1 = list(v)
    v1[3] = -v1[3]
    n1 = Q.gates(K, v1, J)
    n2 = Q.gates(K, [v[p] * (1 if p % 2 == 0 else -1) for p in range(J)], J)
    ctx.prec = PREC
    return {
        "PC": pc["X1"] and pc["X2"] and pc["X3"],
        "NC1": not n1["X2"],
        "NC2": (not n2["X1"]) and (not n2["X2"]),
    }


def main():
    c = controls()
    print("controls:", c, flush=True)
    if not all(c.values()):
        print("VERDICT q986 T80b: VOID (a control does not behave)")
        sys.exit(2)
    if "--controls-only" in sys.argv:
        print("VERDICT q986 T80b: INCOMPLETE (no row processed)")
        sys.exit(3)
    rows = json.load(open("results_q584.json"))["rows"]
    H = arb("1e-90")
    sel = list(range(20, len(rows), 40))
    allok = True
    for i in sel:
        ctx.prec = PREC
        r = rows[i]
        a = arb(float(r["a"]))
        centre = bool(r["centre"])
        I = IMPH(a)
        J = len(r["p"])
        u = [arb(float(t)) for t in r["p"]] + [arb(float(t)) for t in (r["v"] if centre else r["v"][:-1])]
        n = len(u)
        m = n - J
        res = None
        for it in range(8):
            Fu = I.F(u, J, centre)
            res = max(abs(t.mid()) for t in Fu)
            if res < arb("1e-120"):
                break
            Jm = jac(I, u, J, centre, H)
            step = Jm.solve(arb_mat([[(-t).mid()] for t in Fu]))
            u = [(u[k] + step[k, 0]).mid() for k in range(n)]
        if not (res < arb("1e-100")):
            print(f"VERDICT q986 T80b: VOID (row {i} does not polish to 1e-100)")
            sys.exit(2)
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
        opp = [-x for x in vv_]
        x4 = not all(t > 0 for t in opp)
        ctx.prec = PREC
        if g["count"] != J * (J + 1):
            print(
                f"VERDICT q986 T80b: VOID (initial minor count {g['count']} differs from {J * (J + 1)} at row {i})"
            )
            sys.exit(2)
        ok = g["X1"] and g["X2"] and g["X3"] and x4
        allok = allok and ok
        print(
            f"row {i}: a {float(a.mid()):.6f}, K {int(r['K'])}, centre {centre}, J {J}, residual {float(res.mid()):.2e}: X1 {g['X1']}, X2 {g['X2']} ({g['bad']} not certainly positive of {g['count']}), X3 {g['X3']}, X4 {x4} [{time.time() - t0:.0f}s]",
            flush=True,
        )
    print(f"VERDICT q986 T80b: {'PASS' if allok else 'FAIL'}; rows {len(sel)}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q986 T80b: VOID (exception)")
        sys.exit(2)
