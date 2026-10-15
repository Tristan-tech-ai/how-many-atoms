r"""q1876_pr2b_kkt_edge_certificate.py (14 Oct 2026, task PR2b, claim B3): ball-arithmetic certificate that the edge KKT function
K(x) = int phi(t - x) log G(t) dt is >= 0 for every x <= 0, for the test profile
    G(t) = Phi(-g - t) + sum_k nu_k phi(t - s_k),  g = 4.361045,  (s_k, nu_k) = (0, 3.49413), (-2.68983, 2.25076), (-4.361045, 0.62400).
This is the explicit input printed in Section IV-A with the innermost atom moved from -4.36104 to -g = -4.361045 (by 5e-6), so that the atom sits at
the end of the Lebesgue part; the tail lemma T below needs s_3 <= -g. With q = G_A / M_A, G_A(y) = G(y - A) G(-y - A), M_A = int G_A, the duality
C <= sup_{|x| <= A} D(N(x,1) || q) gives C + h_1 <= log M_A - inf_{|x| <= A} [K(x - A) + K(-x - A)], so K >= 0 on (-inf, 0] gives e^(C + h_1) <= M_A
for every A > 0, and limsup (e^(C + h_1) - 2A) <= 2 int (G - 1{t < 0}) = 2(sum nu - g) = 4.01569.

Method (python-flint 0.9, 160-bit balls).
 Q  On [-X0, 0], X0 = 14: K, K' = int (t - x) phi(t - x) log G and K'' = int ((t - x)^2 - 1) phi(t - x) log G by acb.integral on [c - 16, c + 16]
    (log with the analytic flag), plus a closed-form tail bound from |log G(t)| <= a + t^2/2 for all real t, a = log(1 + sum nu / sqrt(2 pi)), valid
    because G <= 1 + sum nu / sqrt(2 pi) and G >= nu_1 phi(t) with nu_1 / sqrt(2 pi) >= 1 (asserted). A cell [l, r] is certified when
    max over e in {l, r} of min(K(e), K(e) + K'(e)(o - e) - M (r - l)^2 / 2) > 0, o the other end, M an upper bound of |K''| on the cell (K'' evaluated
    with x the ball of the cell). Cells start at width 0.25 and are bisected down to width 2^-24; a point where the ball of K lies below 0 is a
    certified negative value.
 T  For x <= -X0: G(t) - 1 >= phi(w)(nu_3 - 1/w) for w = -(t + g) > 0 (Mills ratio, s_3 <= -g), so log G >= 0 for t <= T = -g - 1/nu_3, and
    log G >= (G - 1)/Gmax >= nu_3 phi(w) / (2 Gmax) for t <= T' = -g - 2/nu_3. Hence K(x) >= P(x) - N(x) with
    P(x) = nu_3 / (2 Gmax) e^{-(x+g)^2/4} / (2 sqrt(pi)) Phi(sqrt2 (T' - (x - g)/2)) and N(x) = phi(D)((a + T^2/2)/D + 1/D^3), D = T - x.
    For x <= 2T + g every factor of N/P decreases as x decreases, so P(-X0) > N(-X0) with -X0 <= 2T + g proves K > 0 on (-inf, -X0].
Registered before the first run (14 Oct 2026, before any output of this script was read):
 C1 the integrator returns finite balls everywhere (else VOID);
 C2 every cell of [-14, 0] is certified (no certified negative value, no cell left at the minimum width);
 C3 the premises of T hold (s_3 <= -g, nu_1 >= sqrt(2 pi), -X0 <= 2T + g) and P(-X0) > N(-X0);
 C4 the printed dual value 2(sum nu - g) rounds to 4.01569.
Verdict (last line): VERDICT q1876: PASS if C1 to C4 hold; FAIL if a certified negative value of K is found on [-14, 0]; INCOMPLETE if C2 or C3
fails without a certified negative value (a cell at the minimum width, or a premise of T false); VOID if C1 fails, C4 fails or the script crashes.
Selftest (corrupted profiles, run before the real run): nu_1 lowered by 0.05 must give FAIL (K(0) about -0.01); s_3 = -g + 0.01 must give INCOMPLETE
(premise of T false). Repair of the selftest before the first real run (14 Oct 2026): the case s_3 = -g + 0.01 gave FAIL with a certified negative
K at x = -4.5, which is the true answer for that profile (the registered expectation was wrong); it is kept as a FAIL case, and the INCOMPLETE branch
is tested by s_3 = -g + 1e-6 (premise of T false while K stays positive on [-14, 0]). The criteria C1 to C4 of the real run are unchanged.
Repair after run 1 (14 Oct 2026, kept in q1876_out_run1_incomplete.txt, verdict INCOMPLETE): the premise s_3 <= -g was tested as a comparison of two
arb balls built from the decimals -4.361045 and 4.361045; neither is exact in binary, so arb could not certify equality and returned False. The
premise is now tested exactly on the rational values of the same decimal strings. Criteria, cells and bounds are unchanged.
Usage: py small_gap_q206/q1876_pr2b_kkt_edge_certificate.py [--registration-run] [--selftest]   (from temporal_within/)
"""

import sys
import time
from fractions import Fraction
from flint import acb, arb, ctx

ctx.prec = 160
G_STR = "4.361045"
ATOMS = [("0", "3.49413"), ("-2.68983", "2.25076"), ("-4.361045", "0.62400")]
X0 = 14
L = 16
W0 = arb(1) / 4
WMIN = arb(2) ** -24
SQ2 = arb(2).sqrt()
SQ2PI = (2 * arb.pi()).sqrt()


def make(gs, atoms):
    g = arb(gs)
    at = [(arb(s), arb(n)) for s, n in atoms]

    def Gc(z):
        v = (acb(g) + z) / acb(SQ2)
        r = v.erfc() / 2
        for s, n in at:
            u = z - acb(s)
            r += acb(n) * (-(u * u) / 2).exp() / acb(SQ2PI)
        return r

    return g, at, Gc


def kint(Gc, x, j, c):
    """int over [c - L, c + L] of w_j(t - x) phi(t - x) log G(t); x an arb (point or cell ball)"""
    xa = acb(x)

    def f(t, analytic):
        lg = Gc(t).log(analytic=analytic)
        u = t - xa
        w = acb(1) if j == 0 else (u if j == 1 else u * u - 1)
        return w * (-(u * u) / 2).exp() / acb(SQ2PI) * lg

    v = acb.integral(f, c - L, c + L, abs_tol=arb(2) ** -150, eval_limit=200000)
    return v.real


def tails(a, X, R):
    """bounds on the integrals outside |t - x| <= R of |w_j| phi |log G|, using |log G| <= a + u^2 + x^2 with |x| <= X"""
    phR = (-(R * R) / 2).exp() / SQ2PI
    M0 = (R / SQ2).erfc() / 2
    M1 = phR
    M2 = R * phR + M0
    M3 = (R * R + 2) * phR
    M4 = R**3 * phR + 3 * M2
    b = a + X * X
    return [2 * (b * M0 + M2), 2 * (b * M1 + M3), 2 * (b * (M2 + M0) + M4 + M2)]


def run(gs, atoms, x0=X0):
    g, at, Gc = make(gs, atoms)
    lines = []
    nusum = sum((n for _, n in at), arb(0))
    a = (1 + nusum / SQ2PI).log()
    nu1 = [n for s, n in at if float(s) == 0.0][0]
    s3, nu3 = min(at, key=lambda z: float(z[0]))
    prem_a = bool(nu1 >= SQ2PI)
    tb = tails(a, arb(x0) + 1, arb(L) - W0)
    cache = {}

    def K01(x):
        if x not in cache:
            xa = arb(x)
            k0 = kint(Gc, xa, 0, x)
            k1 = kint(Gc, xa, 1, x)
            cache[x] = (k0 + arb(0, tb[0].upper()), k1 + arb(0, tb[1].upper()))
        return cache[x]

    void = False
    neg = None
    stuck = []
    ncell = 0
    minlb = None
    stack = []
    n0 = int(x0 * 4)
    for i in reversed(range(n0)):
        stack.append((-(i + 1) / 4.0, -i / 4.0))
    t0 = time.time()
    while stack:
        l, r = stack.pop()
        kl, dl = K01(l)
        kr, dr = K01(r)
        for k in (kl, kr, dl, dr):
            if not k.is_finite():
                void = True
        if void:
            break
        if kl < 0 or kr < 0:
            neg = l if kl < 0 else r
            break
        cell = arb((l + r) / 2, (r - l) / 2)
        k2 = kint(Gc, cell, 2, (l + r) / 2) + arb(0, tb[2].upper())
        if not k2.is_finite():
            void = True
            break
        M = abs(k2).upper()
        h = arb(r - l)
        lbl = min(kl.lower(), (kl + dl * h - M * h * h / 2).lower())
        lbr = min(kr.lower(), (kr - dr * h - M * h * h / 2).lower())
        lb = max(lbl, lbr)
        ncell += 1
        if lb > 0:
            minlb = lb if minlb is None or lb < minlb else minlb
        elif h / 2 >= WMIN:
            m = (l + r) / 2
            stack.append((l, m))
            stack.append((m, r))
        else:
            stuck.append((l, r))
    lines.append(
        "cells certified or examined: %d; endpoint evaluations: %d; seconds %.0f"
        % (ncell, len(cache), time.time() - t0)
    )
    if minlb is not None:
        lines.append("smallest certified lower bound on a cell: %s" % minlb.str(5, radius=False))
    for x in (0.0, -0.25, -4.0, -14.0):
        if x in cache:
            lines.append("K(%s) in %s; K'(%s) in %s" % (x, cache[x][0].str(8), x, cache[x][1].str(8)))
    # tail lemma T
    T = -g - 1 / nu3
    Tp = -g - 2 / nu3
    Gmax = 1 + nusum / SQ2PI
    x = -arb(x0)
    D = T - x
    P = nu3 / (2 * Gmax) * (-((x + g) ** 2) / 4).exp() / (2 * arb.pi().sqrt()) * Phi(SQ2 * (Tp - (x - g) / 2))
    N = (-(D * D) / 2).exp() / SQ2PI * ((a + T * T / 2) / D + 1 / D**3)
    prem_s3 = min(Fraction(sa) for sa, _ in atoms) <= -Fraction(
        gs
    )  # exact rational comparison of the printed decimals (run 1 compared two inexact balls)
    prem_x = bool(x <= 2 * T + g)
    c3 = prem_a and prem_s3 and prem_x and bool(P > N)
    lines.append(
        "tail lemma: s_3 <= -g %s; nu_1 >= sqrt(2 pi) %s; -X0 <= 2T + g = %s %s; P(-X0) = %s, N(-X0) = %s"
        % (prem_s3, prem_a, (2 * T + g).str(6), prem_x, P.str(5), N.str(5))
    )
    dual = 2 * (nusum - g)
    c4 = abs(dual - arb("4.01569")) < arb("5e-6")
    lines.append("dual value 2(sum nu - g) = %s" % dual.str(10))
    if void:
        return lines, "VOID"
    if neg is not None:
        lines.append("certified negative value of K at x = %s: %s" % (neg, cache[neg][0].str(6)))
        return lines, "FAIL"
    if not c4:
        return lines, "VOID"
    if stuck or not c3:
        lines.append("cells left at minimum width: %d" % len(stuck))
        return lines, "INCOMPLETE"
    return lines, "PASS"


def Phi(z):
    return (-z / SQ2).erfc() / 2


def selftest():
    ok = True
    bad1 = [("0", "3.44413")] + ATOMS[1:]
    bad2 = ATOMS[:2] + [("-4.351045", "0.62400")]
    bad3 = ATOMS[:2] + [("-4.361044", "0.62400")]
    for name, at, want in [
        ("nu_1 lowered by 0.05", bad1, "FAIL"),
        ("s_3 = -g + 0.01", bad2, "FAIL"),
        ("s_3 = -g + 1e-6", bad3, "INCOMPLETE"),
    ]:
        lines, v = run(G_STR, at)
        print("selftest %-25s -> %s (want %s); %s" % (name, v, want, lines[-1]))
        ok = ok and v == want
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--registration-run" in sys.argv:
        print(__doc__.split("Usage:")[0][-2600:])
        print("registration run: criteria printed, nothing evaluated")
        sys.exit(0)
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    try:
        lines, v = run(G_STR, ATOMS)
    except Exception as e:
        print("crash:", repr(e))
        print("VERDICT q1876: VOID")
        sys.exit(2)
    for l in lines:
        print(l)
    print("VERDICT q1876: %s" % v)
