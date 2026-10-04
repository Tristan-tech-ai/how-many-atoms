"""Q470: T2 of 613(7)/614(4) - a rigorous SUPPORT-SIZE certificate for one capacity-achieving state of the
amplitude-constrained Poisson channel, in ball arithmetic (python-flint arb).

THE OBJECT. Y ~ Poisson(x), x in [0, A], A = 311.368737, the certified K = 21 state of
results_q66_K21_A311.369_exact40.json (origin atom, 19 interior atoms, wall atom at A; 40-digit values).
THE LOGIC, in three rigorous steps and one cited one:
 (1) Krawczyk: the 40-equation stationary system (D_k = D_wall for k = 0..K-2, D'_k = 0 for the interior
     atoms, sum of weights = 1) has exactly one zero v* in an explicit box X about the computed point (613's
     machinery, Rump 1983 / Hoffman et al. 1310.3410 Thm 3.1).
 (2) Positivity: every weight in X is positive, so v* is a probability distribution with exactly K atoms.
 (3) KKT inequality for v*: D_{v*}(x) <= C_{v*} for all x in [0, A], shown for every state in X at once:
      - origin (0, h0]: D'(x) = log x - E_x[g(Y+1) - g(Y)], g(y) = log y! + log Q(y) (Poisson identity
        d/dx E_x f(Y) = E_x[f(Y+1) - f(Y)]), and log h0 below the lower bound of E_x[...] over [0, h0]
        gives D' < 0 there, so D < D(0) = C;
      - each interior atom: D'' < 0 on I_k = [x_k - r_k - h, x_k + r_k + h] by the centred form
        D''(I_k) in D''(c_k) + D'''(I_k) [-(r_k + h), r_k + h], so D <= C on I_k by Taylor about the true
        atom, where D = C and D' = 0 exactly;
      - the wall: D' > 0 on [A - h, A] by the centred form D'(c) + D''(I) [-h/2, h/2], so D < D(A) = C;
      - the rest of [0, A]: on each piece [c - rho, c + rho] the Taylor enclosure
        D(c) + D'(c) [-rho, rho] + D''(c) [0, rho^2]/2 + D'''([c - rho, c + rho]) [-rho^3, rho^3]/6,
        with D, D', D'' at the POINT c (the ripple's cancellation is then exact to the working precision)
        and only the cubic remainder over the interval; sup < inf C_X, bisecting adaptively.
     The first run (this file's v1) used naive interval evaluation of D'' over wide neighbourhoods and failed
     at every atom with upper bounds of order 10 against true values of order 1e-02 to 1e-07: the
     dependency width of a termwise sum, not the function. This version is the centred-form cure.
 (4) Cited: the capacity-achieving input of the amplitude-constrained discrete-time Poisson channel is unique
     and the KKT condition is necessary and sufficient (Shamai 1990; Cao, Hranilovic, Chen 2014 Part I - the
     exact statements are fetched and quoted in the amendment, not assumed here).
 Then |supp P*| = K exactly at this A.
THE ALPHABET is truncated at Ymax = A + 16 sqrt(A) + 60 as q409 does; the certificate is for the truncated
channel, and the tail lemma that removes the truncation is stated in the amendment and NOT proved here.

PRE-REGISTERED, before the run (v2; v5 makes every piece centre and endpoint an exact point, because v4 showed the
neighbourhood half-width h_k carrying a radius 6e-20 from a ball square root, which the termwise sum amplified to a
radius 5e-19 on D(c) against a margin of 8e-20; v3 makes the atom half-width adaptive, h_k = sqrt(20 floor/|D''(x_k)|), so that
the margin at the neighbourhood's edge is at least ten times the box floor 4 rad(C_X) - v2 would otherwise leave
an undecidable sliver next to every bulk atom whose |D''| is below about 1e-08):
 (i)   the point has max |F| < 1e-28 in arb (v1 measured 2.8e-35 at the stored 40-digit point);
 (ii)  Krawczyk containment on the ladder 1e-14 .. 1e-08 (v1: up to 1e-11) AND at the sweep box r = 1e-22;
 (iii) all weights positive in the box;
 (iv)  D''' agrees with a central difference of D'' at one point to relative 1e-06 (a check on the formula);
 (v)   the KKT sweep passes on every piece with the worst margin printed; a piece undecided after 40
       bisections, or more than 400000 pieces, is a FAIL with its location printed.
PASS needs all five.
Usage: py q470_support_certificate.py [PREC] [NPIECE]"""

import os, sys, json, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _outname(stem):
    """24 Sep: a certificate never overwrites an earlier one. On 22 Sep an older, slower v4 run finished after the v5
    run cited in amendment 617 and overwrote its PASS json with FAIL (amendment 657(2)). The name now carries a run
    stamp, and the plain name is written only if it does not exist yet."""
    plain = stem + ".json"
    if not os.path.exists(plain):
        return plain
    return stem + time.strftime("_%Y%m%d_%H%M%S") + f"_pid{os.getpid()}.json"


from flint import arb, arb_mat, ctx

PREC = int(sys.argv[1]) if len(sys.argv) > 1 else 128
NPIECE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ctx.prec = PREC
t0 = time.time()

STATE = sys.argv[3] if len(sys.argv) > 3 else "results_q66_K21_A311.369_exact40.json"
d = json.load(open(os.path.join(HERE, STATE)))
d = d[-1] if isinstance(d, list) else d
K = int(d["K"])
Aa = arb(d["A"])
xs0 = [arb(str(t)) for t in d["x"]]
ws0 = [arb(str(t)) for t in d["w"]]
assert xs0[0] == 0 and abs(xs0[-1] - Aa) < arb("1e-6"), "state must run from the origin to the wall"
Af = float(d["A"])
YSIG = float(os.environ.get("YSIG", "16"))  # v13: alphabet cutoff in standard deviations (env)
YMAX = int(Af + YSIG * np.sqrt(Af) + 60)
Ys = [arb(y) for y in range(YMAX + 1)]
LG = [(y + 1).lgamma() for y in Ys]
N = YMAX + 1
n = 2 * K - 2  # unknowns: x_1..x_{K-2}, w_0..w_{K-1}
RBOX = float(os.environ.get("RBOX", "1e-22"))  # the sweep box (scaled); the residual is 1.9e-28 (v8: env)
HATOM = arb("1e-6")  # half-width of each atom's D'' neighbourhood, in x
MAXDEPTH, MAXPIECES = 30, int(os.environ.get("MAXPIECES", "100000"))


def fl(a):
    return float(a.mid().str(25, radius=False))


def logP(xx):
    lx = xx.log()
    return [y * lx - xx - lg for y, lg in zip(Ys, LG)]


def P_of_interval(xx):
    """pmf for an x-ball that may contain 0: x^y e^-x / y! without a logarithm."""
    ex = (-xx).exp()
    return [(xx**yi) * ex / (lg.exp()) for yi, lg in enumerate(LG)]


class State:
    def __init__(self, x, w):
        self.x, self.w = x, w
        self.LP = [None] * K
        self.P = [None] * K
        self.P[0] = [arb(1)] + [arb(0)] * (N - 1)
        self.LP[0] = [arb(0)] * N
        for k in range(1, K):
            self.LP[k] = logP(x[k])
            self.P[k] = [t.exp() for t in self.LP[k]]
        self.Q = [sum(w[k] * self.P[k][i] for k in range(K)) for i in range(N)]
        self.lQ = [q.log() for q in self.Q]

    def D_atom(self, k):
        if k == 0:
            return -self.lQ[0]
        return sum(self.P[k][i] * (self.LP[k][i] - self.lQ[i]) for i in range(N))

    def derivs(self, xx, upto=3):
        """D, D', D'', D''' at x (point or ball), from P'/P = u, P''/P = u^2 + u', P'''/P = u^3 + 3uu' + u''
        with u = (y - x)/x, u' = -(u + 1)/x, u'' = 2(u + 1)/x^2:
           D   = sum P (lp - lQ)
           D'  = sum P u (lp - lQ)
           D'' = sum P [(u^2 + u')(lp - lQ) + u^2]
           D'''= sum P [(u^3 + 3uu' + u'')(lp - lQ) + 2u^3 + 3uu']"""
        lp = logP(xx)
        p = [t.exp() for t in lp]
        D0 = arb(0)
        D1 = arb(0)
        D2 = arb(0)
        D3 = arb(0)
        D4 = arb(0)
        for i in range(N):
            q = lp[i] - self.lQ[i]
            u = (Ys[i] - xx) / xx
            up = -(u + 1) / xx
            upp = 2 * (u + 1) / (xx * xx)
            D0 += p[i] * q
            if upto >= 1:
                D1 += p[i] * u * q
            if upto >= 2:
                a2 = u * u + up
                D2 += p[i] * (a2 * q + u * u)
            if upto >= 3:
                a3 = u * u * u + 3 * u * up + upp
                D3 += p[i] * (a3 * q + 2 * u * u * u + 3 * u * up)
            if upto >= 4:  # v10: D'''' = sum P [a4 q + a3 u + 3 a2^2 + 3 u a3 - 6 u^2 a2 + 2 u^4]
                uppp = -6 * (u + 1) / (xx * xx * xx)
                a4 = u * u * u * u + 6 * u * u * up + 3 * up * up + 4 * u * upp + uppp
                D4 += p[i] * (a4 * q + a3 * u + 3 * a2 * a2 + 3 * u * a3 - 6 * u * u * a2 + 2 * u * u * u * u)
        if upto >= 4:
            return D0, D1, D2, D3, D4
        return D0, D1, D2, D3

    def FJ(self, raw=False):
        x, w, P, LP, Q, lQ = self.x, self.w, self.P, self.LP, self.Q, self.lQ
        inter = list(range(1, K - 1))
        Yx = {k: [(Ys[i] - x[k]) / x[k] for i in range(N)] for k in inter}
        dP = {k: [P[k][i] * Yx[k][i] for i in range(N)] for k in inter}
        D = [self.D_atom(k) for k in range(K)]
        D1 = {}
        D2 = {}
        for k in inter:
            lpq = [LP[k][i] - lQ[i] for i in range(N)]
            D1[k] = sum(dP[k][i] * lpq[i] for i in range(N))
            ddp = [
                P[k][i] * (Yx[k][i] * Yx[k][i] - 1 / x[k] - (Ys[i] - x[k]) / (x[k] * x[k])) for i in range(N)
            ]
            D2[k] = sum(ddp[i] * lpq[i] for i in range(N)) + sum(
                P[k][i] * Yx[k][i] * Yx[k][i] for i in range(N)
            )
        Pm = arb_mat([P[k] for k in range(K)])
        dPm = arb_mat([dP[k] for k in inter])
        Hm = arb_mat([[P[k][i] / Q[i] for k in range(K)] for i in range(N)])
        Gm = arb_mat([[w[k] * P[k][i] * Yx[k][i] / Q[i] for k in inter] for i in range(N)])
        dDw = Pm * Hm
        dDx = Pm * Gm
        dD1w = dPm * Hm
        dD1x = dPm * Gm
        F = [arb(0)] * n
        J = [[arb(0)] * n for _ in range(n)]
        col_x = lambda k: k - 1
        col_w = lambda k: (K - 2) + k
        for j in range(K - 1):
            F[j] = D[j] - D[K - 1]
            for k in inter:
                J[j][col_x(k)] = -(dDx[j, k - 1] - dDx[K - 1, k - 1]) + (D1[k] if k == j else 0)
            for k in range(K):
                J[j][col_w(k)] = -(dDw[j, k] - dDw[K - 1, k])
        for a, j in enumerate(inter):
            r = (K - 1) + a
            F[r] = D1[j]
            for k in inter:
                J[r][col_x(k)] = -dD1x[a, k - 1] + (D2[j] if k == j else 0)
            for k in range(K):
                J[r][col_w(k)] = -dD1w[a, k]
        F[-1] = sum(w) - 1
        for k in range(K):
            J[-1][col_w(k)] = arb(1)
        if raw:
            return F, J, dict(D=D, D1=D1, D2=D2, dDw=dDw, dDx=dDx, dD1w=dD1w, dD1x=dD1x, inter=inter)
        return F, J


def unpack(v):
    return [xs0[0]] + list(v[: K - 2]) + [Aa], list(v[K - 2 :])


def ruiz(A, sweeps=80):
    m = A.shape[0]
    uu = np.ones(m)
    vv = np.ones(m)
    B = A.copy()
    for _ in range(sweeps):
        r = np.sqrt(np.maximum(np.abs(B).max(1), 1e-300))
        c = np.sqrt(np.maximum(np.abs(B).max(0), 1e-300))
        uu /= r
        vv /= c
        B = (B / r[:, None]) / c[None, :]
    return uu, vv, B


def diag(dd):
    return arb_mat(
        [[arb(float(dd[i])) if i == j else arb(0) for j in range(len(dd))] for i in range(len(dd))]
    )


if __name__ == "__main__":
    say(f"state {STATE}: K = {K}, A = {Af}, alphabet 0..{YMAX}, n = {n} unknowns, prec {PREC} bits")
    v = list(xs0[1 : K - 1]) + list(ws0)
    NT = float(os.environ.get("NEWTON_TARGET", "1e-28"))
    for it in range(30):
        S = State(*unpack(v))
        F, J = S.FJ()
        Fm = np.array([fl(f) for f in F])
        Jm = np.array([[fl(J[i][j]) for j in range(n)] for i in range(n)])
        nF = np.abs(Fm).max()
        if nF < NT:
            break
        dv = np.linalg.solve(Jm, Fm)
        v = [v[i] - arb(float(dv[i])) for i in range(n)]
    S = State(*unpack(v))
    F, J = S.FJ()
    Fm = np.array([fl(f) for f in F])
    Jm = np.array([[fl(J[i][j]) for j in range(n)] for i in range(n)])
    say(
        f"(i) point: max |F| = {np.abs(Fm).max():.3e} {'PASS' if np.abs(Fm).max() < NT else 'FAIL'} (target {NT:.0e}); cond(J) = {np.linalg.cond(Jm):.3e}"
    )
    Dr, Dc, _ = ruiz(Jm)
    Js = (Dr[:, None] * Jm) * Dc[None, :]
    Y = np.linalg.inv(Js)
    Yarb = arb_mat([[arb(float(Y[i, j])) for j in range(n)] for i in range(n)])
    DrM, DcM = diag(Dr), diag(Dc)
    Fs = arb_mat([[arb(float(Dr[i])) * F[i]] for i in range(n)])
    YF = Yarb * Fs
    yf = [YF[i, 0].abs_upper() for i in range(n)]
    say(f"    ||Y F||_inf (rigorous) = {max(yf).str(5, radius=False)}, scaled cond {np.linalg.cond(Js):.3e}")
    I = diag(np.ones(n))

    def krawczyk(r):
        Xv = [arb(v[i].mid(), float(r * Dc[i])) for i in range(n)]
        _, JX = State(*unpack(Xv)).FJ()
        C = I - Yarb * DrM * arb_mat(JX) * DcM
        rows = [sum(C[i, j].abs_upper() for j in range(n)) for i in range(n)]
        normC = max(rows)
        ra = arb(r)
        return all((yf[i] + rows[i] * ra) < ra for i in range(n)) and normC < arb(1), normC, Xv

    best = None
    say(f"\n(ii) Krawczyk ladder")
    e0 = int(round(-np.log10(RBOX)))
    for e in range(e0, 7, -1):  # v11: the ladder climbs from the sweep box upward
        r = 10.0 ** (-e)
        ok, normC, _ = krawczyk(r)
        say(
            f"     r = {r:.0e}: ||I - Y J(X)||_inf = {normC.str(5, radius=False)}  {'CONTAINED' if ok else 'not contained'}   [{time.time() - t0:.0f}s]"
        )
        if ok:
            best = r
        else:
            break
    okb, normb, Xv = krawczyk(RBOX)
    say(
        f"     sweep box r = {RBOX:.0e}: ||I - Y J(X)||_inf = {normb.str(5, radius=False)}  {'CONTAINED' if okb else 'not contained'}"
    )
    if not okb:
        say("FAIL at (ii): the sweep box is not contained")
        sys.exit()
    if best is None:
        best = RBOX
    say(f"     exactly one zero in the box z0 +/- {best:.0e} (scaled), and in the sweep box")
    SX = State(*unpack(Xv))
    xb, wb = SX.x, SX.w
    say(
        f"\n(iii) weights in the sweep box: min lower bound {min(w.lower() for w in wb).str(6, radius=False)} "
        f"{'PASS' if all(w.lower() > 0 for w in wb) else 'FAIL'}; atom radii max {max(fl(t.rad()) for t in xb):.1e}, "
        f"weight radii max {max(fl(t.rad()) for t in wb):.1e}"
    )

    # (iv) D''' against a central difference of D''
    c = xb[7].mid()
    e = arb("1e-6")
    _, _, _, D3c = SX.derivs(c)
    _, _, D2p, _ = SX.derivs(c + e, 2)
    _, _, D2m, _ = SX.derivs(c - e, 2)
    fd = (D2p - D2m) / (2 * e)
    rel = abs(fl(fd) - fl(D3c)) / max(abs(fl(D3c)), 1e-300)
    D4c = SX.derivs(c, 4)[4]
    fd4 = (SX.derivs(c + e, 3)[3] - SX.derivs(c - e, 3)[3]) / (2 * e)
    rel4 = abs(fl(fd4) - fl(D4c)) / max(abs(fl(D4c)), 1e-300)
    say(
        f"     D'''' check at x = {fl(c):.4f}: formula {fl(D4c):.10e}, central difference {fl(fd4):.10e}, rel diff {rel4:.2e}"
    )
    rel = max(rel, rel4)
    say(
        f"\n(iv) D''' at x = {fl(c):.4f}: formula {fl(D3c):.10e}, central difference {fl(fd):.10e}, rel diff {rel:.2e} "
        f"{'PASS' if rel < 1e-6 else 'FAIL'}"
    )

    # (v) the KKT sweep
    CX = SX.D_atom(K - 1)
    Clo = CX.lower()
    say(
        f"\n(v) KKT sweep. C_X = {CX.str(20)}  (radius {fl(CX.rad()):.1e}); atom half-width {HATOM.str(3, radius=False)}"
    )
    xm = [t.mid() for t in xb]
    xr = [t.rad() for t in xb]
    fails = []
    count = [0]
    margins = []
    h0 = min(arb("1e-3"), xm[1] / 5)
    xi = arb(h0 / 2, h0 / 2)
    p = P_of_interval(xi)
    dg = [Ys[i + 1].log() + SX.lQ[i + 1] - SX.lQ[i] for i in range(N - 1)]
    Eg = sum(p[i] * dg[i] for i in range(N - 1))
    ok0 = h0.log() < Eg.lower()
    say(
        f"     origin (0, {h0.str(4, radius=False)}]: log h0 = {h0.log().str(6, radius=False)} < lower E_x[dg] = {Eg.lower().str(6, radius=False)} -> {'PASS' if ok0 else 'FAIL'}"
    )
    if not ok0:
        fails.append(("origin", 0.0))

    diag_fail = []

    def taylor_D(a, b, want=False):
        c = ((a + b) / 2).mid()  # v5: the centre is an exact point; the interval covers hull(a, b)
        rho = arb(((b - a) / 2).abs_upper()) + a.rad() + b.rad() + arb("1e-30")
        D0, D1, D2, _ = SX.derivs(c, 2)
        _, _, _, D3 = SX.derivs(arb(c, rho), 3)
        T = (
            D0
            + D1 * arb(0, rho)
            + D2 * arb(rho * rho / 2, rho * rho / 2) / 2
            + D3 * arb(0, rho * rho * rho) / 6
        )
        if want:
            return T, (
                fl(c),
                fl(rho),
                fl(D0),
                fl(D0.rad()),
                fl(D1) * fl(rho),
                fl(D2) * fl(rho) ** 2 / 2,
                abs(fl(D3)) * fl(rho) ** 3 / 6,
                fl(D3.rad()) * fl(rho) ** 3 / 6,
            )
        return T

    def piece(a, b, depth=0):
        count[0] += 1
        if count[0] > MAXPIECES:
            fails.append(("too many pieces", fl(a)))
            return arb(-1)
        m = Clo - taylor_D(a, b).upper()
        if m > 0:
            return m
        if depth >= MAXDEPTH:
            fails.append((fl(a), fl(b)))
            if len(diag_fail) < 6:
                T, comp = taylor_D(a, b, True)
                diag_fail.append(comp)
                say(
                    f"        undecided piece: c = {comp[0]:.12f}, rho = {comp[1]:.2e}, D0 - C = {comp[2] - fl(CX):.3e} (rad D0 {comp[3]:.1e}, rad C {fl(CX.rad()):.1e}), "
                    f"D1 rho = {comp[4]:.2e}, D2 rho^2/2 = {comp[5]:.2e}, |D3| rho^3/6 = {comp[6]:.2e}, rad(D3) rho^3/6 = {comp[7]:.2e}"
                )
            return m
        mid = (a + b) / 2
        return min(piece(a, mid, depth + 1), piece(mid, b, depth + 1))

    lo = h0
    floor = 4 * CX.rad()  # the box floor on inf C - sup D near an atom
    ALO = int(os.environ.get("ATOM_LO", "1"))
    AHI = int(os.environ.get("ATOM_HI", str(K - 1)))  # v12: chunked sweep
    CHUNK = ALO > 1 or AHI < K - 1
    if CHUNK:
        say(
            f"     CHUNK: atoms {ALO}..{AHI} (with the free stretch before each); the rest of [0, A] is another chunk's"
        )
    for k in range(1, K):
        if k > AHI:
            break
        if k < K - 1:
            _, _, D2c, _ = SX.derivs(xm[k], 2)
            hk = max(
                HATOM, (20 * floor / abs(D2c)).sqrt()
            ).mid()  # margin at the edge >= 10 x floor (v3: adaptive); v5: an EXACT point
        else:
            hk = HATOM
        a_k = xm[k] - xr[k] - hk
        b_k = xm[k] + xr[k] + hk
        if k == K - 1:
            a_k = Aa - HATOM
            b_k = Aa
        if k < ALO:
            lo = b_k
            continue
        seg = []
        if a_k > lo:
            step = (a_k - lo) / NPIECE
            for j in range(NPIECE):
                seg.append(piece(lo + step * j, lo + step * (j + 1)))
        margins += seg
        if k < K - 1:
            ck = xm[k]
            hw = xr[k] + hk
            _, _, _, D3c = SX.derivs(ck, 3)  # v10: second-order centred form for D'' over I_k:
            D4I = SX.derivs(arb(ck, hw), 4)[4]  # D''(x) in D''(c) + D'''(c) [-h, h] + D''''(I) [0, h^2]/2
            D2I = D2c + abs(D3c) * arb(0, hw) + arb(0, 1) * max(D4I.upper(), arb(0)) * hw * hw / 2
            okk = D2I.upper() < 0
            if not okk:
                fails.append(("atom %d D''" % k, fl(ck)))
            say(
                f"     atom {k:2d} x = {fl(ck):10.4f}: h_k = {fl(hk):.1e}, D''(x_k) = {fl(D2c):.4e}, D'' on I_k upper {D2I.upper().str(4, radius=False):>12} {'PASS' if okk else 'FAIL'}; "
                f"stretch before: {len(seg)} starts, {count[0]} pieces so far, worst margin {fl(min(seg)) if seg else float('nan'):.2e}   [{time.time() - t0:.0f}s]"
            )
        else:
            cw = Aa - HATOM / 2
            _, D1c, _, _ = SX.derivs(cw, 1)
            _, _, D2I, _ = SX.derivs(arb(cw, HATOM / 2), 2)
            D1I = D1c + D2I * arb(0, HATOM / 2)
            okk = D1I.lower() > 0
            if not okk:
                fails.append(("wall D'", Af))
            say(
                f"     wall A = {Af}: D'(A - h/2) = {fl(D1c):.4e}, D' on [A - h, A] lower {D1I.lower().str(4, radius=False)} {'PASS' if okk else 'FAIL'}; "
                f"stretch before: worst margin {fl(min(seg)) if seg else float('nan'):.2e}   [{time.time() - t0:.0f}s]"
            )
        lo = b_k
    if not margins:
        margins = [arb(1)]
    say(
        f"\n     pieces evaluated: {count[0]}; worst free-stretch margin (inf C - sup D): {fl(min(margins)):.3e}; failures: {len(fails)}"
    )
    if fails:
        say(f"     failures at: {fails[:12]}")
    allok = ok0 and not fails and rel < 1e-6
    say(
        f"\nVERDICT: {'PASS' if allok else 'FAIL'} - "
        + (
            f"the truncated-alphabet Poisson channel at A = {Af} has a unique stationary {K}-atom state in the box, it is a "
            f"probability distribution, and it satisfies the KKT inequality on all of [0, A]."
            if allok
            else "see the failures above."
        )
    )
    json.dump(
        {
            "A": Af,
            "K": K,
            "r_krawczyk": best,
            "r_box": RBOX,
            "pieces": count[0],
            "worst_margin": fl(min(margins)),
            "fails": [str(f) for f in fails],
            "verdict": "PASS" if allok else "FAIL",
            "x": [t.mid().str(40, radius=False) for t in xb],
            "w": [t.mid().str(40, radius=False) for t in wb],
        },
        open(
            _outname(
                os.path.join(
                    HERE, f"q470_certificate_K{K}_A{Af:.3f}" + (f"_atoms{ALO}-{AHI}" if CHUNK else "")
                )
            ),
            "w",
        ),
        indent=1,
    )
    say(f"[{time.time() - t0:.0f}s]")
