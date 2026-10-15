r"""q1882_exact_state_closed_chain.py (14 Oct 2026, task F8, claim B2 at the exact states): the exact-state chain of Section VIII in one process,
with three gaps of the earlier chain (q1855, q1856, q1836, q1857/q1881) closed. A gate with a VERDICT line.

Gaps found on 14 Oct 2026 while the proof text was drafted (reading 342):
 G1 Error per row. A residual row D(u_j) - D(A) is the difference of two trapezoid sums, so its quadrature error is at most 2 eps_F; a Jacobian
    entry (and an entry of the A-column b) is at most two Gram sums plus one sum of the D' or D'' kind, so its error is at most 2 eps_J + eps_F.
    q1836 widened rows by eps_F and entries by max(eps_F, eps_J), and it read eps_F and eps_J as 5-digit decimals rounded to nearest.
 G2 Tails. The strip bound M of the trapezoid theorem must cover the whole line, and the computed sums stop at |y| = Y = A + 40; q1856 and q1857
    bounded M on [-Y, Y] only, and q1857's window tail covered |y - x| <= 54 only.
 G3 Box. q1857 evaluated the certificate on the polished state widened by 1e-120, while the Krawczyk step only places the exact zero in the box of
    radius 1e-110 (its K(X) reached 1.5e-116 at K = 64). Here every bound and the certificate are computed over the same box X = v0 + [-r, r], r = 1e-110.
The arithmetic is that of the earlier scripts (python-flint balls; 600 bits for the polish and Krawczyk, 1200 for minors, 192 for the bounds and the
certificate); the constants pi/2 and log sqrt(2 pi) are balls here (q1857 used the double pi/2, 6e-17 below pi/2), and every widening uses the upper
end of its ball.

Tail lemma (written out here and in the paper). Let Y = A + 40 and |Im z| <= a = 1/2. Take the wall atom A as reference and let S be the atoms x_k
with A - x_k <= pi (phase b (x_k - A) within pi/2). For y >= Y, phi(z - x_k)/phi(z - A) = exp((x_k - A)(z - (x_k + A)/2)) has modulus
rho_k(y) = exp((x_k - A)(y - (x_k + A)/2)), which decreases in y, so tau_A(y) = sum_{k not in S} (w_k / w_e) rho_k(y) <= tau_A(Y). If tau_A(Y) < 1,
then |Q(z)| >= w_e |phi(z - A)| (1 - tau_A(Y)), and with r_S <= 1/w_e,
  |log Q(z)| <= c0 + (y - A)^2 / 2,   c0 = |log w_e| + log sqrt(2 pi) + a^2/2 + max(|log(1 - tau_A(Y))|, log(2 + 1/w_e)) + pi/2.
Every integrand of the scheme (rows phi(z - x_j) times a polynomial of degree <= 2 times 1 or a log bracket with constant at most 2; columns
w_k phi(z -+ x_k)(1 or z -+ x_k)/Q(z); and He_k(z - x) phi(z - x) log Q(z) with k <= 6 and |x| <= A) is bounded for |y| >= Y by
  g(y) = e^{a^2/2} phi(|y| - A) 720 (2|y|)^6 (c0 + log sqrt(2 pi) + 3 + 3 y^2 + 2|y| / (w_e (1 - tau_A(Y)))),
using phi(y - x) <= phi(|y| - A), 1 + |z - x| <= 2|y|, |He_k(s)| <= k! (1 + |s|)^k and w_k <= 1. The polynomial factor has degree 8 and |y| >= 40,
so g(y) <= g(Y) e^{-39(|y| - Y)}. Hence int_{|y|>Y} g <= 2 g(Y)/39 (added to M) and h sum over grid points with |y| > Y <= 2 g(Y)(1/39 + h)
(added as the truncation of the grid). For the windowed sums of the certificate, grid points with 54 <= |y - x| and |y| <= Y add at most
2 * 720 (1 + 54 + a)^6 phi(54) max|log Q| (1/53 + h).

Registered before the first run (14 Oct 2026), for every state of small_gap_q206/q1843_pr1b_states.txt (94 states: 91 OPTIMAL, 3 not):
 C1 tau < 1 on every box of [0, Y] (the strip) and tau_A(Y) < 1;
 C2 the Krawczyk inclusion holds for X = v0 + [-1e-110, 1e-110]^n with rows widened by 2 eps_F and entries by 2 eps_J + eps_F, and every mass of X
    is positive;
 C3 all m(m + 1) initial minors of X = [T | v] are certainly positive over the box, and X1 holds;
 C4 (OPTIMAL states) every atom zone passes with z >= 1e-14, the wall zone passes, every gap box passes, at most 400000 boxes, with eps_G including
    the tail terms;
 C5 the added tail terms (M tails, grid truncation, far window) are each below 1e-300.
Verdict (last line): VERDICT q1882: PASS if C1 to C5 hold at the state (C4 only for an OPTIMAL state); FAIL if C2 or C3 fails with C1 holding, or if
C4 fails with a certified positive upper bound of D - C on a gap box; INCOMPLETE if C4 fails otherwise (box cap, zone floor) or C5 fails; VOID if C1
fails, the polish does not converge or the script crashes.
Control registered before the target runs: the state K = 40, A = 29.85 (q1836, q1857, q1881 passed there).
Selftest (--selftest): the stored state with its first position moved by 1e-30 and no polish (not a zero within 1e-110) must give FAIL.
Usage: py small_gap_q206/q1882_exact_state_closed_chain.py STATE_JSON [--optimal 0|1] [--registration-run] [--selftest]   (from temporal_within/)
"""

import math
import os
import sys
import time

sys.dont_write_bytecode = True
TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TW)
os.chdir(TW)
from flint import arb, arb_mat, ctx  # noqa: E402
import q920_cap_flint as F  # noqa: E402
import q921_cap_tools as T  # noqa: E402
import q973_extended_tp_test_hi as Q  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1855_strip_general_reading as SG  # noqa: E402

R_BOX = arb("1e-110")
ZMIN, DEPTH, WIN, KMAX, CAP = 1e-14, 45, 14.0, 6, 400000
t0 = time.time()


def say(*x):
    print("[%5.0f s]" % (time.time() - t0), *x, flush=True)


def up(x):
    return x.mid() + x.rad()


def U(x):
    """an exact ball [0 +- upper(x)], a little above, for widening"""
    return arb(0, up(x) * (1 + 1e-30))


def tail_g(xs, ws, A, we, a, L2P):
    """tau_A(Y), c0 and g(Y) of the tail lemma"""
    Y = A + 40
    pi = arb.pi()
    tau = arb(0)
    for x, w in zip(xs, ws):
        if (x - A).contains(0) and (x - A).rad() < 1e-50:
            continue  # the reference atom itself
        if up(A - x) <= float(pi.mid()) - 1e-9 and up(A - x) >= 0:
            continue  # in S: phase within pi/2
        tau += (w / we) * ((x - A) * (Y - (x + A) / 2)).exp()
    c0 = abs(we.log()) + L2P + a * a / 2 + arb(max(up(abs((1 - tau).log())), up((2 + 1 / we).log()))) + pi / 2
    phi40 = (-(arb(40) ** 2) / 2 - L2P).exp()
    g = (a * a / 2).exp() * phi40 * 720 * (2 * Y) ** 6 * (c0 + L2P + 3 + 3 * Y * Y + 2 * Y / (we * (1 - tau)))
    return tau, c0, g


def main(STATE, optimal, corrupt=False):
    ctx.prec = 600
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(STATE)
    m = len(u)
    odd = wc is not None
    F.HD = F.HD / 2
    F.G.H = F.G.H / 2
    cap = F.Cap(A_str, m, odd, grid_A=repr(float(A_str) + 31))
    say(
        "state",
        STATE,
        "K",
        2 * m + 2 + (1 if odd else 0),
        "A",
        A_str,
        "grid",
        cap.N,
        "points from",
        cap.y[0].str(8, radius=False),
        "to",
        cap.y[-1].str(8, radius=False),
    )
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    if corrupt:
        v[0] = v[0] + arb("1e-30")
        say("selftest: first position moved by 1e-30 from the stored state; no polish")
        v0, ok = v, True
    else:
        v0, res, ok = T.newton_full(cap, v, lambda *x: None, tol=arb("1e-135"), maxit=14)
        say("polished: ok", ok, "residual", F.maxabs(res).str(3, radius=False))
    if not ok:
        print("VERDICT q1882: VOID (polish did not converge)")
        return 2
    n = cap.nn
    Xb = [x + arb(0, R_BOX.mid()) for x in v0]

    # ---- bounds over the box X (192 bits) ----
    ctx.prec = 192
    L2P = (arb.pi() * 2).log() / 2
    F.L2P = L2P
    ub = Xb[:m]
    wb = Xb[m : 2 * m]
    web = Xb[2 * m]
    wcb = Xb[2 * m + 1] if odd else None
    A = arb(A_str)
    Af = float(A.mid())
    a = arb("0.5")
    ea = (a * a / 2).exp()
    h = F.HD
    delta = arb("0.01")
    phi = lambda s: (-(s * s) / 2 - L2P).exp()
    xs = list(ub) + [-t for t in ub] + [A, -A] + ([arb(0)] if odd else [])
    ws = list(wb) * 2 + [web] * 2 + ([wcb] if odd else [])
    kf = math.factorial(KMAX)
    sgrid = [arb(k) / 100 for k in range(0, 4001)]
    f1 = [up(phi(s.union(s + arb(1) / 100)) * (1 + s + arb(1) / 100 + a) ** KMAX * kf) for s in sgrid]
    suf = list(f1)
    for k in range(len(sgrid) - 2, -1, -1):
        suf[k] = max(suf[k], suf[k + 1])

    def Hs(dist):
        k = min(int(max(float(dist.mid()) - float(dist.rad()), 0.0) * 100), len(sgrid) - 1)
        return arb(suf[k])

    Y = A + 40
    nbox = int(float(Y.mid()) / 0.01) + 1
    MF = arb(0)
    MJ = arb(0)
    MG = arb(0)
    worst_tau = arb(0)
    c1ok = True
    for i in range(nbox):
        lo = delta * i
        y = (lo + delta / 2).union(lo).union(lo + delta)
        ym = float((lo + delta / 2).mid())
        k1, x1, w1, S, rho, rS, tau = SG.split(xs, ws, y, ym, a)
        if not up(tau) < 1:
            c1ok = False
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
        dpos = y - A if ym > Af else arb(0)
        MG += delta * ea * (Hs(dpos) + Hs(y)) * logQ
    tauA, c0, g = tail_g(xs, ws, A, web, a, L2P)
    c1ok = c1ok and up(tauA) < 1
    say(
        "strip: boxes %d of [0, Y], largest tau %s; tau_A(Y) %s; C1 %s"
        % (nbox, worst_tau.str(5, radius=False), tauA.str(5), c1ok)
    )
    if not c1ok:
        print("VERDICT q1882: VOID (strip)")
        return 2
    fac = 2 / ((2 * arb.pi() * a / h).exp() - 1)
    Mtail = 2 * g / 39
    trunc = 2 * g * (1 / arb(39) + h)
    epsF = fac * (2 * MF + Mtail) + trunc
    epsJ = fac * (2 * MJ + Mtail) + trunc
    E_row = arb(up(2 * epsF))
    E_ent = arb(up(2 * epsJ + epsF))
    say(
        "g(Y) %s; M tail %s; grid truncation %s"
        % (
            arb(up(g)).str(4, radius=False),
            arb(up(Mtail)).str(4, radius=False),
            arb(up(trunc)).str(4, radius=False),
        )
    )
    say(
        "eps_F %s, eps_J %s; rows widened by %s, entries by %s"
        % (
            arb(up(epsF)).str(5, radius=False),
            arb(up(epsJ)).str(5, radius=False),
            E_row.str(5, radius=False),
            E_ent.str(5, radius=False),
        )
    )

    # ---- Krawczyk and minors over the box (as q1836, with the widths of G1) ----
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
    mags = [abs(Kd[i, 0]) for i in range(n)]
    incl = all(up(t) < R_BOX.mid() for t in mags)
    worstK = max(up(t) for t in mags)
    massmin = min(float(x.mid()) - float(x.rad()) for x in Xb[m:])
    c2 = incl and massmin > 0
    say(
        "Krawczyk: largest |K(X) - v0| %.4e against r = 1e-110; inclusion %s; smallest mass in X %.6e; C2 %s"
        % (float(worstK), incl, massmin, c2)
    )
    J = [[JX[i, j] + U(E_ent) for j in range(n)] for i in range(n)]
    b = [t + U(E_ent) for t in bX]
    Rr = list(range(m)) + list(range(2 * m, n))
    Mc = list(range(m, n))
    r_ = len(Rr)
    As = arb_mat(m, m, [J[m + p][q] for p in range(m) for q in range(m)])
    Bs = arb_mat(m, r_, [J[m + p][c] for p in range(m) for c in Mc])
    Cs = arb_mat(r_, m, [J[rr][q] for rr in Rr for q in range(m)])
    Dm = arb_mat(r_, r_, [J[rr][c] for rr in Rr for c in Mc])
    S_ = As - Bs * Dm.solve(Cs)
    br = arb_mat(r_, 1, [b[rr] for rr in Rr])
    bu = arb_mat(m, 1, [b[m + p] for p in range(m)])
    beff = bu - Bs * Dm.solve(br)
    ctx.prec = Q.MINPREC
    sg = 1 if m % 2 == 1 else -1
    Tm = [[S_[p, q] * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    vv = [sg * beff[p, 0] * (1 if p % 2 == 0 else -1) for p in range(m)]
    gt = Q.gates(Tm, vv, m)
    c3 = gt["X1"] and gt["X2"] and gt["count"] == m * (m + 1)
    say(
        "minors over the box: X1 %s, X2 %s (%d initial minors, %d not certainly positive); C3 %s"
        % (gt["X1"], gt["X2"], gt["count"], gt["bad"], c3)
    )

    # ---- certificate of D <= C over the same box (as q1881, with the bounds of G2 and the box of G3) ----
    c4 = None
    worst = None
    failed_pos = False
    incomplete = False
    ctx.prec = 192
    L2P = (arb.pi() * 2).log() / 2
    F.L2P = L2P
    _, _, _, _, Qg, lQ = cap.tables(ub, wb, web, wcb)
    ys = [float(yy.mid()) for yy in cap.y]
    lqmax = arb(max(up(abs(t)) for t in lQ))
    tailw = arb(0)
    for k in range(0, 4000):
        sk = arb(WIN) + h * k
        tailw += 2 * h * kf * (1 + sk + a) ** KMAX * phi(sk) * lqmax
    far = 2 * 720 * (1 + arb(54) + a) ** 6 * phi(arb(54)) * lqmax * (1 / arb(53) + h)
    epsG = fac * (2 * MG + Mtail) + tailw + far + trunc
    c5 = all(up(t) < 1e-300 for t in (Mtail * fac, trunc, far))
    say(
        "eps_G %s (window tail %s, far window %s); C5 %s"
        % (
            arb(up(epsG)).str(5, radius=False),
            arb(up(tailw)).str(3, radius=False),
            arb(up(far)).str(3, radius=False),
            c5,
        )
    )
    wid = U(epsG)

    def Gk(X, kmax=KMAX):
        xm = float(X.mid())
        xr = float(X.rad())
        lo_i = max(0, int((xm - WIN - xr - ys[0]) / float(h.mid())) - 2)
        hi_i = min(len(ys), int((xm + WIN + xr - ys[0]) / float(h.mid())) + 3)
        out = [arb(0)] * (kmax + 1)
        for i in range(lo_i, hi_i):
            s = cap.y[i] - X
            base = (-(s * s) / 2 - L2P).exp() * lQ[i]
            hk_prev, hk = arb(0), arb(1)
            for k in range(kmax + 1):
                out[k] += hk * base
                hk_prev, hk = hk, s * hk - k * hk_prev
        return [h * t + wid for t in out]

    if optimal:
        GA = Gk(A, 3)
        passed = failed = boxes = 0
        zones = []
        for t in sorted(list(ub) + ([arb(0)] if odd else []), key=lambda z: float(z.mid())):
            gg = Gk(t, 3)
            z = 0.05
            okz = False
            while z >= ZMIN:
                Zb = arb(float(t.mid()), z + float(t.rad()))
                g4 = Gk(Zb, 4)[4]
                if up(-gg[2] + abs(gg[3]) * z + abs(g4) * z * z / 2) < 0:
                    okz = True
                    break
                z /= 2
            if okz:
                passed += 1
                zones.append((float(t.mid()) - z, float(t.mid()) + z))
            else:
                failed += 1
                say("atom zone failed at u = %.6f" % float(t.mid()))
                zones.append((float(t.mid()) - 1e-6, float(t.mid()) + 1e-6))
        z = 0.05
        okw = False
        while z >= ZMIN:
            Zb = arb(Af - z / 2, z / 2)
            if up(GA[1] + abs(GA[2]) * z + abs(Gk(Zb, 3)[3]) * z * z / 2) < 0:
                okw = True
                break
            z /= 2
        zones.append((Af - (z if okw else 0.0), Af))
        zones.sort()
        work = []
        pos = 0.0
        for z0, z1 in zones:
            if z0 > pos:
                nn = max(1, int((z0 - pos) / 0.05) + 1)
                for k in range(nn):
                    work.append((pos + (z0 - pos) * k / nn, pos + (z0 - pos) * (k + 1) / nn, 0))
            pos = max(pos, z1)
        gfail = 0
        while work and boxes < CAP:
            x0, x1, depth = work.pop()
            boxes += 1
            c = arb((x0 + x1) / 2)
            wd = (x1 - x0) / 2
            gc = Gk(c, KMAX - 1)
            g6 = Gk(arb((x0 + x1) / 2, wd), KMAX)[KMAX]
            t_ = arb(0, wd)
            poly = g6 / math.factorial(KMAX)
            for k in range(KMAX - 1, -1, -1):
                poly = poly * t_ + gc[k] / math.factorial(k)
            enc = GA[0] - poly
            if up(enc) < 0:
                if worst is None or up(enc) > worst:
                    worst = up(enc)
            elif depth < DEPTH:
                mid = (x0 + x1) / 2
                work.append((x0, mid, depth + 1))
                work.append((mid, x1, depth + 1))
            else:
                gfail += 1
                if float(enc.mid()) - float(enc.rad()) > 0:
                    failed_pos = True
        left = len(work)
        c4 = failed == 0 and okw and gfail == 0 and left == 0
        incomplete = not c4 and not failed_pos
        say(
            "certificate: atom zones %d passed, %d failed; wall %s; boxes %d; gap failures %d (certified positive %s); left %d; largest upper bound of D - C %s; C4 %s"
            % (
                passed,
                failed,
                okw,
                boxes,
                gfail,
                failed_pos,
                left,
                "%.4e" % float(worst) if worst is not None else "none",
                c4,
            )
        )
    else:
        say("certificate not run: the state is not marked OPTIMAL")
    if not (c2 and c3):
        verdict = "FAIL"
    elif optimal and not c4:
        verdict = "FAIL" if failed_pos else "INCOMPLETE"
    elif not c5:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    print("VERDICT q1882: %s" % verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--registration-run" in a:
        print(__doc__.split("Usage:")[0])
        print("registration run: criteria printed, nothing evaluated")
        sys.exit(0)
    if "--selftest" in a:
        rc = main("results_q206_gauss_K40_A29.85.json", True, corrupt=True)
        print("SELFTEST:", "PASS" if rc == 1 else "FAIL (the corrupted state did not give FAIL)")
        sys.exit(0 if rc == 1 else 1)
    st = [x for x in a if not x.startswith("--") and x not in ("0", "1")]
    opt = a[a.index("--optimal") + 1] == "1" if "--optimal" in a else True
    try:
        sys.exit(main(st[0] if st else "results_q206_gauss_K40_A29.85.json", opt))
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("VERDICT q1882: VOID (crash: %r)" % (e,))
        sys.exit(2)
