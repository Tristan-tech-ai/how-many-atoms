"""q1024 (7 Oct 2026): READING (no verdict): does the geometric ripple formula of T93 (q1020) describe the second derivative of the KKT function of the immune functional?
Rows of results_q584.json polished to 1e-120 at 700 bits by the procedure of q986 (IMPH, Newton from the stored doubles). Measured: d_p = -g''(p_j) / C with g''(y) / C = sum_x W_x [((x - y)^2 - 1) f(x - y) + ((x + y)^2 - 1) f(x + y)] / C (the second derivative of the KKT function g of q986's gs).
Predicted: -R''(p_j) of q1020 from the polished positions p alone (window of 2 HW + 1 atoms, Taylor order 6). Printed: the ratio measured / predicted (1 would mean the same formula, 2 a factor from the weight 1 / P^2).
Usage: py q1024_immune_ripple_reading.py ROW [ROW ...]   (row indices of results_q584.json)"""

import sys, json
import mpmath as mp
import q986_immune_hp_test as H
from flint import arb, arb_mat, ctx

HW = 5
ORD = 6
PI2 = None


def polish(r):
    ctx.prec = H.PREC
    Hh = arb("1e-90")
    a = arb(float(r["a"]))
    centre = bool(r["centre"])
    I = H.IMPH(a)
    J = len(r["p"])
    u = [arb(float(t)) for t in r["p"]] + [arb(float(t)) for t in (r["v"] if centre else r["v"][:-1])]
    n = len(u)
    res = None
    for it in range(8):
        Fu = I.F(u, J, centre)
        res = max(abs(t.mid()) for t in Fu)
        if res < arb("1e-120"):
            break
        Jm = H.jac(I, u, J, centre, Hh)
        step = Jm.solve(arb_mat([[(-t).mid()] for t in Fu]))
        u = [(u[k] + step[k, 0]).mid() for k in range(n)]
    if not (res < arb("1e-100")):
        raise RuntimeError("row does not polish")
    return I, u, J, centre, res


def g2_over_C(I, u, J, centre):
    p = u[:J]
    v = u[J : 2 * J] if centre else u[J : 2 * J - 1] + [arb(1) - sum(u[J : 2 * J - 1])]
    c = (arb(1) - sum(v)) if centre else None
    P = I.P(p, v, c)
    C = sum(q / Pi for q, Pi in zip(I.Q, P))
    W = [q / (2 * Pi * Pi) for q, Pi in zip(I.Q, P)]
    out = []
    for y in p:
        s = arb(0)
        for x, Wi in zip(I.x, W):
            dm, dp = x - y, x + y
            s += Wi * ((dm * dm - 1) * H.f(dm) + (dp * dp - 1) * H.f(dp))
        out.append(s / C)
    return out, p


def ripple_pred(pos, p_index, psi, us):
    # q1020's formula with the window of 2 HW + 1 atoms around atom p_index (1-based) of the list pos
    n = 2 * HW + 1
    x0 = pos[p_index - 1]
    ts = [pos[j - 1] - x0 for j in range(p_index - HW, p_index + HW + 1)]
    ns = [mp.mpf(j - p_index) for j in range(p_index - HW, p_index + HW + 1)]
    V = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            V[i, j] = ts[i] ** j
    c = mp.lu_solve(V, mp.matrix(ns))

    def shift(delta):
        return [sum(c[i] * mp.binomial(i, j) * delta ** (i - j) for i in range(j, n)) for j in range(n)]

    def R(delta):
        cs = shift(delta)
        Jv = mp.mpc(0)
        for uu, ps in zip(us, psi):
            Jv += ps * mp.exp(1j * 2 * mp.pi * sum(cs[j] * uu**j for j in range(1, ORD + 1)))
        return -2 * mp.re(mp.exp(1j * 2 * mp.pi * cs[0]) * Jv)

    h = mp.mpf("0.02")
    vals = [R(k * h) for k in (-2, -1, 0, 1, 2)]
    return float(-(-vals[0] + 16 * vals[1] - 30 * vals[2] + 16 * vals[3] - vals[4]) / (12 * h * h))


def main():
    rows = json.load(open("results_q584.json"))["rows"]
    mp.mp.dps = 60
    us = [mp.mpf(i) / 25 for i in range(-350, 351)]
    psi = [mp.exp(-t * t / 4) / mp.sqrt(4 * mp.pi) * mp.mpf("0.04") for t in us]
    print(
        "READING q1024: immune functional, d_p = -g''(p) / C measured, -R''(p) predicted from the positions alone, ratio"
    )
    for i in [int(t) for t in sys.argv[1:]]:
        r = rows[i]
        I, u, Jn, centre, res = polish(r)
        d2, p = g2_over_C(I, u, Jn, centre)
        pos = [mp.mpf(t.str(80, radius=False)) for t in p]
        print(
            f"row {i}: a {float(r['a']):.6f}, K {int(r['K'])}, centre {centre}, J {Jn}, residual {float(res.mid()):.1e}",
            flush=True,
        )
        for j in range(HW + 1, Jn - HW + 1):
            meas = -float(d2[j - 1].mid())
            pred = ripple_pred(pos, j, psi, us)
            print(
                f"   atom {j}: p {float(pos[j - 1]):.4f}, d_p measured {meas:.4e}, predicted {pred:.4e}, ratio measured/predicted {meas / pred:.4f}",
                flush=True,
            )
    print("END q1024 READING (no verdict)")


if __name__ == "__main__":
    main()
