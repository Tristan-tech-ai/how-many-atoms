r"""q1891_centre_mass_flow_exact.py (15 Oct 2026, task RAISE2): statement (S1) of Section V-A at the exact odd-K states of Proposition 1.
A gate with a VERDICT line.

Claim audit of 15 Oct (paper_unified/referee/CLAIM_AUDIT_2026-10-15.md, item B2): (S2) is certified at the 91 exact states of Proposition 1
through the outward motion, while (S1), the growth of the center mass for odd K, is only numerical. The Krawczyk box of q1882 contains the
exact solution of the Kuhn--Tucker system, and the Jacobian J and the A-column b of that system are enclosed over the box with the
quadrature errors of q1882 (rows widened by 2 eps_F, entries and b by 2 eps_J + eps_F). Along the branch of stationary states through the
exact state (implicit function theorem, J nonsingular on the box), dz/dA = -J^{-1} b. This script encloses that vector over the box in
ball arithmetic (600 bits) and reads the sign of the center-mass component.

The bounds, the strip test and the Krawczyk step are those of q1882 (the code is copied as text, not imported, so the frozen script stays
unchanged); the certificate of D <= C on [-A, A] is not repeated, because q1882 passed it at every one of these states (Proposition 1).

Registered before the first run on a target state (15 Oct 2026), for every odd-K state of small_gap_q206/q1843_pr1b_states.txt (39 states,
all OPTIMAL):
 E1 the strip test of q1882 holds (tau < 1 on every box of [0, Y], tau_A(Y) < 1);
 E2 the Krawczyk inclusion of q1882 holds for X = v0 + [-1e-110, 1e-110]^n and every mass of X is positive;
 P  positive control: every enclosed du_i/dA (i = 1..m) is certainly positive, as Proposition 1 states at these states;
 S1 the enclosed dw_0/dA is certainly positive (lower end of the ball above 0).
Negative control at every state: the same solve with b replaced by -b must give dw_0/dA certainly negative.
Verdict per state (last line): VERDICT q1891: PASS if E1, E2, P, S1 and the negative control hold; FAIL if E1, E2, P hold and dw_0/dA is
certainly negative; INCOMPLETE if E1, E2 hold and an enclosure of P or S1 contains 0; VOID if E1 or E2 fails, if P gives a certainly
negative du_i/dA (a contradiction with Proposition 1, so an instrument fault), if the negative control fails, if the polish does not
converge or the script crashes.
Readout over the 39 states (q1891_readout.py): PASS if every state gives PASS; FAIL if one gives FAIL; otherwise INCOMPLETE or VOID.
Selftest (--selftest): on the control state K = 61, A = 41.2100 the solve with -b in place of b must be read as FAIL by the same code path.
Usage: py small_gap_q206/q1891_centre_mass_flow_exact.py STATE_JSON [--registration-run] [--selftest]   (from temporal_within/)
"""

import math
import os
import sys
import time

sys.dont_write_bytecode = True
TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TW)
os.chdir(TW)
if os.environ.get("TIGHT_SOFT_EXIT"):
    print("VERDICT q1891: VOID (TIGHT_SOFT_EXIT is set; a verdict script refuses it)")
    os._exit(2)
from flint import arb, arb_mat, ctx  # noqa: E402
import q920_cap_flint as F  # noqa: E402
import q921_cap_tools as T  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1855_strip_general_reading as SG  # noqa: E402

R_BOX = arb("1e-110")
t0 = time.time()


def say(*x):
    print("[%5.0f s]" % (time.time() - t0), *x, flush=True)


def up(x):
    return x.mid() + x.rad()


def lo(x):
    return x.mid() - x.rad()


def U(x):
    return arb(0, up(x) * (1 + 1e-30))


def tail_g(xs, ws, A, we, a, L2P):
    """tau_A(Y), c0 and g(Y) of the tail lemma (copied from q1882)"""
    Y = A + 40
    pi = arb.pi()
    tau = arb(0)
    for x, w in zip(xs, ws):
        if (x - A).contains(0) and (x - A).rad() < 1e-50:
            continue
        if up(A - x) <= float(pi.mid()) - 1e-9 and up(A - x) >= 0:
            continue
        tau += (w / we) * ((x - A) * (Y - (x + A) / 2)).exp()
    c0 = abs(we.log()) + L2P + a * a / 2 + arb(max(up(abs((1 - tau).log())), up((2 + 1 / we).log()))) + pi / 2
    phi40 = (-(arb(40) ** 2) / 2 - L2P).exp()
    g = (a * a / 2).exp() * phi40 * 720 * (2 * Y) ** 6 * (c0 + L2P + 3 + 3 * Y * Y + 2 * Y / (we * (1 - tau)))
    return tau, c0, g


def judge(flow, m, odd):
    """per-state reading of an enclosed flow vector: returns (P_ok, P_contradiction, s1) with s1 in {+1, -1, 0}"""
    du = [flow[i] for i in range(m)]
    p_ok = all(lo(t) > 0 for t in du)
    p_bad = any(up(t) < 0 for t in du)
    if not odd:
        return p_ok, p_bad, None
    dw0 = flow[2 * m + 1]
    s1 = 1 if lo(dw0) > 0 else (-1 if up(dw0) < 0 else 0)
    return p_ok, p_bad, s1


def main(STATE, flip=False):
    ctx.prec = 600
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(STATE)
    m = len(u)
    odd = wc is not None
    if not odd:
        print("VERDICT q1891: VOID (not an odd-K state)")
        return 2
    F.HD = F.HD / 2
    F.G.H = F.G.H / 2
    cap = F.Cap(A_str, m, odd, grid_A=repr(float(A_str) + 31))
    say("state", STATE, "K", 2 * m + 3, "A", A_str, "grid", cap.N)
    v = [t.mid() for t in (u + w + [we] + [wc])]
    v0, res, ok = T.newton_full(cap, v, lambda *x: None, tol=arb("1e-135"), maxit=14)
    say("polished: ok", ok, "residual", F.maxabs(res).str(3, radius=False))
    if not ok:
        print("VERDICT q1891: VOID (polish did not converge)")
        return 2
    n = cap.nn
    Xb = [x + arb(0, R_BOX.mid()) for x in v0]

    # ---- bounds over the box X (192 bits), as q1882 ----
    ctx.prec = 192
    L2P = (arb.pi() * 2).log() / 2
    F.L2P = L2P
    ub = Xb[:m]
    wb = Xb[m : 2 * m]
    web = Xb[2 * m]
    wcb = Xb[2 * m + 1]
    A = arb(A_str)
    a = arb("0.5")
    ea = (a * a / 2).exp()
    h = F.HD
    delta = arb("0.01")
    phi = lambda s: (-(s * s) / 2 - L2P).exp()
    xs = list(ub) + [-t for t in ub] + [A, -A] + [arb(0)]
    ws = list(wb) * 2 + [web] * 2 + [wcb]
    Y = A + 40
    nbox = int(float(Y.mid()) / 0.01) + 1
    MF = arb(0)
    MJ = arb(0)
    worst_tau = arb(0)
    e1 = True
    for i in range(nbox):
        lo_ = delta * i
        y = (lo_ + delta / 2).union(lo_).union(lo_ + delta)
        ym = float((lo_ + delta / 2).mid())
        k1, x1, w1, S, rho, rS, tau = SG.split(xs, ws, y, ym, a)
        if not up(tau) < 1:
            e1 = False
            say("strip: tau not below 1 at y =", ym)
            break
        if up(tau) > worst_tau:
            worst_tau = arb(up(tau))
        logbr = max(up(abs((1 - tau).log())), up((1 + rS + tau).log()))
        logQ = arb(up(abs(w1.log()))) + ((abs(y - x1) + a) ** 2 + a * a) / 2 + L2P + arb(logbr) + arb.pi() / 2
        R = arb(max(up(ea * phi(y - xs[j]) * (1 + abs(y - xs[j]) + a) ** 2) for j in range(len(xs))))
        Glog = arb(max(up(((abs(y - xs[j]) + a) ** 2 + a * a) / 2) for j in range(len(xs)))) + L2P + logQ + 2
        Cc = arb(
            max(up(ws[k] * rho[k] * (1 + abs(y - xs[k]) + a) / (w1 * (1 - tau))) for k in range(len(xs)))
        )
        MF += delta * arb(R) * Glog
        MJ += delta * arb(R) * arb(Cc) * 2
    tauA, c0, g = tail_g(xs, ws, A, web, a, L2P)
    e1 = e1 and up(tauA) < 1
    say(
        "strip: boxes %d, largest tau %s; tau_A(Y) %s; E1 %s"
        % (nbox, worst_tau.str(5, radius=False), tauA.str(5), e1)
    )
    if not e1:
        print("VERDICT q1891: VOID (strip)")
        return 2
    fac = 2 / ((2 * arb.pi() * a / h).exp() - 1)
    Mtail = 2 * g / 39
    trunc = 2 * g * (1 / arb(39) + h)
    epsF = fac * (2 * MF + Mtail) + trunc
    epsJ = fac * (2 * MJ + Mtail) + trunc
    E_row = arb(up(2 * epsF))
    E_ent = arb(up(2 * epsJ + epsF))
    say("eps_F %s, eps_J %s" % (arb(up(epsF)).str(5, radius=False), arb(up(epsJ)).str(5, radius=False)))

    # ---- Krawczyk over the box, as q1882 ----
    ctx.prec = 600
    F.L2P = (arb.pi() * 2).log() / 2
    r0, J0, _ = cap.res_jac(v0, True, False)
    Jm = arb_mat(n, n, [J0[i, j].mid() for i in range(n) for j in range(n)])
    Ym = Jm.inv()
    Ym = arb_mat(n, n, [Ym[i, j].mid() for i in range(n) for j in range(n)])
    rX, JX, bX = cap.res_jac(Xb, True, True)
    Fv0 = arb_mat(n, 1, [t + U(E_row) for t in r0])
    JXe = arb_mat(n, n, [JX[i, j] + U(E_ent) for i in range(n) for j in range(n)])
    I = arb_mat(n, n, [arb(1) if i == j else arb(0) for i in range(n) for j in range(n)])
    Dv = arb_mat(n, 1, [arb(0, R_BOX.mid()) for _ in range(n)])
    Kd = -(Ym * Fv0) + (I - Ym * JXe) * Dv
    incl = all(up(abs(Kd[i, 0])) < R_BOX.mid() for i in range(n))
    massmin = min(float(x.mid()) - float(x.rad()) for x in Xb[m:])
    e2 = incl and massmin > 0
    say("Krawczyk inclusion %s; smallest mass in X %.6e; E2 %s" % (incl, massmin, e2))
    if not e2:
        print("VERDICT q1891: VOID (Krawczyk)")
        return 2

    # ---- the flow dz/dA = -J^{-1} b over the box ----
    sgn = -1 if flip else 1
    bXe = arb_mat(n, 1, [sgn * (t + U(E_ent)) for t in bX])
    flow = -(JXe.solve(bXe))
    flow = [flow[i, 0] for i in range(n)]
    p_ok, p_bad, s1 = judge(flow, m, True)
    bneg = (
        arb_mat(n, 1, [-(t) for t in bXe.entries()])
        if hasattr(bXe, "entries")
        else arb_mat(n, 1, [-bXe[i, 0] for i in range(n)])
    )
    flow_n = -(JXe.solve(bneg))
    _, _, s1n = judge([flow_n[i, 0] for i in range(n)], m, True)
    neg_ok = (s1n == -s1) and s1 != 0
    dw0 = flow[2 * m + 1]
    du1 = flow[0]
    say(
        "flow over the box: dw_0/dA = %s (relative radius %.2e); du_1/dA = %s; all du_i/dA certainly > 0: %s; any certainly < 0: %s"
        % (
            dw0.str(12, radius=True),
            float(dw0.rad()) / max(abs(float(dw0.mid())), 1e-300),
            du1.str(12, radius=True),
            p_ok,
            p_bad,
        )
    )
    say("relative growth (dw_0/dA)/w_0 = %s" % ((dw0 / wcb).str(8, radius=False)))
    say("negative control (b -> -b): sign of dw_0/dA %d against %d; control %s" % (s1n, s1, neg_ok))
    verdict, why = decide_state(p_ok, p_bad, s1, neg_ok)
    print("VERDICT q1891: %s%s" % (verdict, (" (%s)" % why) if why else ""))
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 1}.get(verdict, 2)


def decide_state(p_ok, p_bad, s1, neg_ok):
    """the per-state rule of 2255, unchanged; separated in 2256 so that the selftest can exercise every branch on synthetic input"""
    if p_bad:
        return "VOID", "a du_i/dA is certainly negative, against Proposition 1"
    if not neg_ok and s1 != 0:
        return "VOID", "negative control"
    if s1 == -1 and p_ok:
        return "FAIL", ""
    if s1 == 1 and p_ok:
        return "PASS", ""
    return "INCOMPLETE", ""


def selftest_synthetic():
    """2256: synthetic flow vectors (m = 3 positive atoms, odd K, so the vector has 2m + 2 = 8 entries, dw_0 at index 7), read by judge()
    and decide_state(); the negative control is the same vector with every entry negated, as the solve with -b gives by linearity
    """

    def ball(x, r=1e-20):
        return arb(x, r)

    def run(vec):
        f = [ball(x) for x in vec]
        p_ok, p_bad, s1 = judge(f, 3, True)
        _, _, s1n = judge([-t for t in f], 3, True)
        if not p_bad:
            s1n_ok = (s1n == -s1) and s1 != 0
        else:
            s1n_ok = True
        return decide_state(p_ok, p_bad, s1, s1n_ok)[0]

    cases = [
        ([1, 2, 3, 0.1, 0.1, 0.1, 0.1, 0.5], "PASS"),  # all outward, center mass grows
        ([1, 2, 3, 0.1, 0.1, 0.1, 0.1, -0.5], "FAIL"),  # all outward, center mass falls
        ([1, 2, 3, 0.1, 0.1, 0.1, 0.1, 0.0], "INCOMPLETE"),  # center-mass enclosure contains 0
        ([0.0, 2, 3, 0.1, 0.1, 0.1, 0.1, 0.5], "INCOMPLETE"),  # a du_i/dA enclosure contains 0
        ([-1, 2, 3, 0.1, 0.1, 0.1, 0.1, 0.5], "VOID"),
    ]  # a du_i/dA certainly negative
    ok = True
    for vec, want in cases:
        got = run(vec)
        print("selftest case %s -> %s (want %s)" % (vec, got, want))
        ok = ok and got == want
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--registration-run" in a:
        print(__doc__.split("Usage:")[0])
        print("registration run: criteria printed, nothing evaluated")
        sys.exit(0)
    if "--selftest" in a:
        # 2256: the selftest of 2255 flipped the whole forcing b; by linearity that flips every du_i/dA as well, so the positive
        # control answered VOID before the FAIL branch could be reached. The repaired selftest exercises every branch on synthetic
        # flow vectors; the in-run negative control (the solve with -b) is unchanged.
        try:
            ok = selftest_synthetic()
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("SELFTEST: VOID (crash %r)" % (e,))
            sys.exit(2)
        print("SELFTEST:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    st = [x for x in a if not x.startswith("--")]
    try:
        sys.exit(main(st[0]))
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("VERDICT q1891: VOID (crash: %r)" % (e,))
        sys.exit(2)
