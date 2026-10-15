r"""q1849_edge_bound_ball_certificate.py (13 Oct 2026, task PR2 step 4, claim B3): ball-arithmetic certificate of the edge functional of the
configuration gated in q1848, so that the lower bound of the strip constant rests on rigorous arithmetic and not on floating quadrature.

Object. g = 4.361045 and the edge atoms (s, nu) = (0, 3.49413), (-2.68983, 2.25076), (-4.36104, 0.62400) (exact decimals); G(t) = Phi(-g - t) +
sum nu_k phi(t - s_k), E = int_R G log G dt, B = 2(sum nu_k - g - E). By reading 294, liminf_{A->oo} (e^(C+h_1) - 2A) >= B.
Method. E on [-T, T], T = 40, by the certified integrator acb.integral of python-flint at 160 bits (log with analytic=True, so the branch cut is
never touched), split at -g and at the atoms. Tails, bounded by hand and evaluated in balls:
  t >= T: s_k <= 0, so phi(t - s_k) <= phi(t), and Phi(-g - t) <= phi(t)/t <= phi(t); G <= (1 + |nu|) phi(t) <= e^-2, where |G log G| <= sqrt(G); hence
          int_T^oo |G log G| <= sqrt(1 + |nu|) (2 pi)^(-1/4) (2/T) e^(-T^2/4);
  t <= -T: |G - 1| <= Phibar(T - g) + |nu| phi(T - g + s_min)... all bounded by (1 + |nu|) phi(T - g) for T - g >= 1 (s_k in [-g, 0]); with |G - 1| <= 1/2,
          |G log G| <= 3 |G - 1|, hence int_{-oo}^{-T} |G log G| <= 3 (1 + |nu|) Phibar(T - g) <= 3 (1 + |nu|) phi(T - g).
Criteria, fixed before the first run:
 K1 positive control: the same code with no atom and g = 0 gives a ball for 2 kappa whose centre is within 1e-19 of the value 1.8063945711372507073
    of q1845/q1848 (that value is printed to 20 significant digits). Repair before the first real run: the selftest first required the ball to CONTAIN
    the 20-digit decimal; the ball [1.806394571137250707265 +- 2.8e-25] is far narrower than that rounding, so the control could not pass whatever
    the code did (an instrument fault of the control, audited; K2 to K4 are unchanged);
 K2 the ball of B has radius below 1e-20;
 K3 the ball of B lies entirely above 3.9796 (the value stated in the paper) and entirely below sqrt(2 pi e);
 K4 the tail bounds are below 1e-100.
Verdict (last line): VERDICT q1849: PASS if K1 to K4 hold; FAIL if K3 fails; VOID if K1, K2 or K4 fails or the integrator does not converge;
INCOMPLETE is not used (no external input).
Usage: py q1849_edge_bound_ball_certificate.py [--registration-run] [--selftest]   (from temporal_within/)"""

import sys

from flint import acb, arb, ctx

ctx.prec = 160
G_STR = "4.361045"
ATOMS = [("0", "3.49413"), ("-2.68983", "2.25076"), ("-4.36104", "0.62400")]
T = 40
SQ2 = arb(2).sqrt()
TWO_PI = 2 * arb.pi()


def Phi(z):
    return (-z / acb(SQ2)).erfc() / 2


def phi(z):
    return (-(z * z) / 2).exp() / acb(TWO_PI.sqrt())


def B_ball(g_str, atoms):
    g = arb(g_str)
    s = [arb(a) for a, _ in atoms]
    nu = [arb(b) for _, b in atoms]
    tot = sum(nu, arb(0))

    def f(z, analytic):
        G = Phi(-acb(g) - z) + sum((acb(n) * phi(z - acb(x)) for x, n in zip(s, nu)), acb(0))
        return G * G.log(analytic=analytic)

    pts = sorted(set([-T, T, float(-g.mid())] + [float(x.mid()) for x in s]))
    E = acb(0)
    for a, b in zip(pts[:-1], pts[1:]):
        if b > a:
            E += acb.integral(f, a, b)
    Er = E.real
    if not E.imag.contains(0) or not Er.is_finite():
        raise RuntimeError("integrator did not converge")
    Tb = arb(T)
    right = (1 + tot).sqrt() * TWO_PI ** arb("-0.25") * (2 / Tb) * (-(Tb * Tb) / 4).exp()
    tg = Tb - g
    left = 3 * (1 + tot) * (-(tg * tg) / 2).exp() / TWO_PI.sqrt()
    tail = right + left
    Eb = Er + arb(0, float((tail).mid() + tail.rad()))
    return 2 * (tot - g - Eb), tail


def run(g_str, atoms):
    lines = []
    b0, _ = B_ball("0", [])
    k1 = abs(b0 - arb("1.8063945711372507073")) < arb("1e-19")
    lines.append("K1 control: B(empty) = %s, within 1e-19 of 1.8063945711372507073: %s" % (b0.str(25), k1))
    b, tail = B_ball(g_str, atoms)
    k2 = b.rad() < arb("1e-20")
    k3 = (b > arb("3.9796")) and (b < (TWO_PI * arb(1).exp()).sqrt())
    k4 = tail < arb("1e-100")
    lines.append(
        "B = %s (radius %s); K2 %s; K3 %s; tail bound %s (K4 %s)"
        % (b.str(25), b.rad().str(3), k2, k3, tail.str(3), k4)
    )
    if not (k1 and k2 and k4):
        return lines, "VOID"
    return lines, "PASS" if k3 else "FAIL"


def selftest():
    ok = True
    for name, g, at, want in [
        ("the configuration of q1848", G_STR, ATOMS, "PASS"),
        ("one atom only (B about 3.93, below 3.9796)", "1.580807", [("0", "3.41694")], "FAIL"),
    ]:
        _, v = run(g, at)
        print("selftest %-45s -> %s (want %s)" % (name, v, want))
        ok = ok and v == want
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--registration-run" in sys.argv:
        print(__doc__.split("Usage:")[0][-1400:])
        print("registration run: criteria printed, nothing evaluated")
        sys.exit(0)
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    try:
        lines, v = run(G_STR, ATOMS)
    except Exception as e:
        print("crash:", repr(e))
        print("VERDICT q1849: VOID")
        sys.exit(2)
    for l in lines:
        print(l)
    print("VERDICT q1849: %s" % v)
