"""q1855_strip_general_reading.py (13 Oct 2026, task PR1B, claim B2 at the exact states; generalizes q1834): a reading with no verdict word.

Why. q1834 writes Q(z) = w_1 phi(z - x_1)[1 + r_2 e^{i theta} + R(z)] with ONE partner atom x_2 and puts every other atom into |R| <= tau. In the batch
q1844 this failed near y = 0 at K = 61, A = 41.235 (1 - tau = -0.539): with an odd number of atoms the centre atom and the two atoms at +-u_1 have
comparable terms there, so one partner is not enough. The argument does not need ONE partner. For |b| <= a and every atom with a |x_k - x_1| <= pi/2,
the term r_k(y) e^{i b (x_k - x_1)} has a nonnegative real part, r_k(y) = (w_k/w_1) rho_k(y) > 0. So with the set S of all such atoms (other than x_1),
  Re[1 + sum_{k in S} r_k e^{i theta_k} + R] >= 1 - tau(y),   tau(y) = sum_{k not in S, k != 1} r_k(y),
  |1 + sum_S r_k e^{i theta_k} + R| <= 1 + r_S(y) + tau(y),   r_S = sum_S r_k,
and the rest of the q1834 argument is unchanged: if tau < 1 on every box, Q has no zero on |Im z| <= a and |arg(bracket)| < pi/2. One partner is the case
|S| = 1, so tau here is never larger than in q1834 at the same reference atom. The function split() is shared with q1856 and q1857.

Registered before the first run (the states are set by the driver q1858 through STATE; the default is the stored K = 40 state of q1834):
 P1 with a = 0.5, on every box of width 0.01 covering [0, A + 40], tau < 1 (ball arithmetic, 128 bits); the phase condition holds by the choice of S;
 P2 the smallest margin 1 - tau over the boxes is at least 0.3 (the bar of q1834 P2, kept);
 P3 at the same state, the margin is at least the margin of the one-partner split of q1834 (computed here on the same boxes).
Usage: py q1855_strip_general_reading.py  (from temporal_within/; writes nothing but its output)"""

import os
import sys

sys.dont_write_bytecode = True
TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TW)
os.chdir(TW)
from flint import arb, ctx  # noqa: E402
import q920_cap_flint as F  # noqa: E402

STATE = "results_q206_gauss_K40_A29.85.json"


def split(xs, ws, y, ymid, a):
    """Reference atom (largest term at the box midpoint, ranked by the logarithm), the set S of atoms within phase pi/2 of it for |b| <= a, and on the
    box y: rho_k for every k, r_S = sum_S (w_k/w_1) rho_k, tau = sum of the rest. Returns (k1, x1, w1, S, rho, rS, tau).
    """
    half_pi = arb.pi() / 2
    order = sorted(
        range(len(xs)), key=lambda k: -(float(ws[k].mid().log().mid()) - (ymid - float(xs[k].mid())) ** 2 / 2)
    )
    k1 = order[0]
    x1, w1 = xs[k1], ws[k1]
    S = [k for k in order[1:] if a * abs(xs[k] - x1) <= half_pi]
    rho = [((xs[k] - x1) * (2 * y - xs[k] - x1) / 2).exp() for k in range(len(xs))]
    inS = set(S)
    rS = sum((ws[k] / w1 * rho[k] for k in S), arb(0))
    tau = sum((ws[k] / w1 * rho[k] for k in order[1:] if k not in inS), arb(0))
    return k1, x1, w1, S, rho, rS, tau


def run(a_str, A, xs, ws, delta="0.01"):
    a = arb(a_str)
    d = arb(delta)
    Y = A + 40
    nbox = int(float(Y.mid()) / float(d.mid())) + 1
    worst = None
    worst_y = None
    worst_old = None
    tau_fail = 0
    maxS = 0
    p3 = True
    for i in range(nbox):
        lo = d * i
        y = (lo + d / 2).union(lo).union(lo + d)
        ymid = float((lo + d / 2).mid())
        k1, x1, w1, S, rho, rS, tau = split(xs, ws, y, ymid, a)
        maxS = max(maxS, len(S))
        if not (tau < 1):
            tau_fail += 1
        m = 1 - tau
        lower = arb(m.mid() - m.rad())
        if worst is None or lower < worst:
            worst, worst_y = lower, ymid
        # the one-partner split of q1834 on the same box, for P3
        order = sorted(
            range(len(xs)),
            key=lambda k: -(float(ws[k].mid().log().mid()) - (ymid - float(xs[k].mid())) ** 2 / 2),
        )
        old = sum((ws[k] / w1 * rho[k] for k in order[2:]), arb(0))
        mo = 1 - old
        lo_old = arb(mo.mid() - mo.rad())
        if worst_old is None or lo_old < worst_old:
            worst_old = lo_old
        if lower < lo_old - arb("1e-30"):
            p3 = False
    return dict(
        boxes=nbox, tau_fail=tau_fail, worst=worst, worst_y=worst_y, worst_old=worst_old, maxS=maxS, p3=p3
    )


def main():
    ctx.prec = 128
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(STATE)
    A = arb(A_str)
    xs = [x for x in u] + [-x for x in u] + [A, -A] + ([arb(0)] if wc is not None else [])
    ws = [t for t in w] + [t for t in w] + [we, we] + ([wc] if wc is not None else [])
    print("state", STATE, "A", A_str, "atoms", len(xs))
    r = run("0.5", A, xs, ws)
    print(
        "a = 0.5: boxes %d, tau failures %d, smallest margin 1 - tau %s at y = %.2f (one-partner split: %s), largest set S %d"
        % (
            r["boxes"],
            r["tau_fail"],
            r["worst"].str(5, radius=False),
            r["worst_y"],
            r["worst_old"].str(5, radius=False),
            r["maxS"],
        )
    )
    p1 = r["tau_fail"] == 0
    p2 = r["worst"] >= arb("0.3")
    p3 = r["p3"]
    print("P1 %s; P2 %s; P3 %s" % tuple("held" if p else "did not hold" for p in (p1, p2, p3)))
    print("READING (no verdict): zero-free strip with the phase set S; the script writes nothing.")


if __name__ == "__main__":
    main()
