"""q1025 (7 Oct 2026): READING (no verdict): the geometric ripple formula of T93 for the NPMLE functional (uniform data on [-a, a]; KKT c(y) = int phi(x - y) q / P dx / sum q <= 1, weights W = q / (2 P), as in q601).
Same procedure as q1024 (immune): rows of results_q601_ext.json polished to 1e-120 at 700 bits with IMPH whose gs is replaced by the NPMLE gs; measured d_p = -g''(p_j) / C, predicted -R''(p_j) from the polished positions alone. For the weight 1 / P (instead of 1 / P^2) the ripple of g / C is minus the smoothed relative ripple of P once, so the ratio measured / predicted is expected to be about 1 (the immune ratio is about 2, q1024).
Usage: py q1025_npmle_ripple_reading.py ROW [ROW ...]"""

import sys, json
import mpmath as mp
import q986_immune_hp_test as H
import q1024_immune_ripple_reading as R24
from flint import arb, ctx


class IMPHN(H.IMPH):
    def gs(self, p, v, c, ys):
        P = self.P(p, v, c)
        Cn = sum(self.Q)
        W = [q / (2 * Pi) for q, Pi in zip(self.Q, P)]
        out = []
        for y in ys:
            g0 = g1 = arb(0)
            for x, Wi in zip(self.x, W):
                dm, dp = x - y, x + y
                em, ep = H.f(dm), H.f(dp)
                g0 += Wi * (em + ep)
                g1 += Wi * (dm * em - dp * ep)
            out.append((g0 / Cn, g1 / Cn))
        return out, Cn


def g2_over_C(I, u, J, centre):
    p = u[:J]
    v = u[J : 2 * J] if centre else u[J : 2 * J - 1] + [arb(1) - sum(u[J : 2 * J - 1])]
    c = (arb(1) - sum(v)) if centre else None
    P = I.P(p, v, c)
    Cn = sum(I.Q)
    W = [q / (2 * Pi) for q, Pi in zip(I.Q, P)]
    out = []
    for y in p:
        s = arb(0)
        for x, Wi in zip(I.x, W):
            dm, dp = x - y, x + y
            s += Wi * ((dm * dm - 1) * H.f(dm) + (dp * dp - 1) * H.f(dp))
        out.append(s / Cn)
    return out, p


def main():
    rows = json.load(open("results_q601_ext.json"))["rows"]
    H.IMPH = IMPHN
    R24.H.IMPH = IMPHN
    mp.mp.dps = 60
    us = [mp.mpf(i) / 25 for i in range(-350, 351)]
    psi = [mp.exp(-t * t / 4) / mp.sqrt(4 * mp.pi) * mp.mpf("0.04") for t in us]
    print(
        "READING q1025: NPMLE functional, d_p = -g''(p) / C measured, -R''(p) predicted from the positions alone, ratio"
    )
    for i in [int(t) for t in sys.argv[1:]]:
        r = rows[i]
        I, u, Jn, centre, res = R24.polish(r)
        d2, p = g2_over_C(I, u, Jn, centre)
        pos = [mp.mpf(t.str(80, radius=False)) for t in p]
        print(
            f"row {i}: a {float(r['a']):.6f}, K {int(r['K'])}, centre {centre}, J {Jn}, residual {float(res.mid()):.1e}",
            flush=True,
        )
        for j in range(R24.HW + 1, Jn - R24.HW + 1):
            meas = -float(d2[j - 1].mid())
            pred = R24.ripple_pred(pos, j, psi, us)
            print(
                f"   atom {j}: p {float(pos[j - 1]):.4f}, d_p measured {meas:.4e}, predicted {pred:.4e}, ratio measured/predicted {meas / pred:.4f}",
                flush=True,
            )
    print("END q1025 READING (no verdict)")


if __name__ == "__main__":
    main()
