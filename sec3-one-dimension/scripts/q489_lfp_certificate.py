"""Q489: a rigorous SUPPORT-SIZE certificate for the least-favourable prior (LFP) of the bounded normal mean under
squared error, in ball arithmetic (python-flint arb/acb). It mirrors q470_support_certificate.py (Poisson).

THE OBJECT. X ~ N(theta, 1), theta in [-m, m]. The stored symmetric state of results_q486_lfp_sym.json (row with m
closest to the argument): atoms +-a_j, j = 1..J, a_J = m pinned, plus a centre atom 0 when centre is true; v = [a_1..a_{J-1},
w_1..w_J, (c)], w_j the weight of EACH of +-a_j, c the centre weight. Bayes rule delta = N/Z, Z = sum w_k phi(x - theta_k),
N = sum w_k theta_k phi(x - theta_k); risk R(t) = int (delta(x) - t)^2 phi(x - t) dx; Bayes risk r = sum w_k R(theta_k).

THE LOGIC, three rigorous steps and one cited one:
 (1) Krawczyk: the stationary system F(v) = 0,
        R(a_j) - R(m) = 0 and R'(a_j) = 0 (j < J),  R(0) - R(m) = 0 (centre only),  2 sum w + c - 1 = 0,
     has exactly one zero v* in an explicit box X about the polished point z (F and the interval Jacobian J(X) in ball
     arithmetic, Ruiz scaling, preconditioner Y = inverse of the scaled midpoint Jacobian, as in q470).
 (2) Positivity: every weight in X is positive and the atoms satisfy 0 < a_1 < ... < a_{J-1} < m in X, so v* is a
     probability measure with exactly K distinct atoms; at v*, r = sum w_k R(theta_k) = R(m) (all R(theta_k) = R(m)).
 (3) KKT inequality R(t) <= r = R(m) on [0, m] for every prior in X (hence for v*). The prior is symmetric by
     construction, so R is even and [0, m] gives all of [-m, m]. The cover of [0, m]:
      - each interior atom: R'' < 0 on I_k = [c - h', c + h'] (c the exact midpoint of the atom ball, h' >= its radius
        + h_k) by the second-order centred form R''(c) + |R'''(c)| h' + max(sup R''''(I_k), 0) h'^2/2; with R(a_k*) = r
        and R'(a_k*) = 0 exactly at v*, Taylor about the true atom gives R <= r on I_k;
      - the centre atom (if present): R'' < 0 on [-h0, h0] by the same form, with R(0) = r and R'(0) = 0 (evenness);
      - the wall [m - hw, m]: R' > 0 by R'(c) - |R''(c)| hw/2 - sup|R'''(I)| hw^2/8 > 0, c = m - hw/2, so R < R(m) = r;
      - the free stretches between these: the Taylor enclosure R(c) + R'(c)[-rho, rho] + R''(c)[0, rho^2]/2 +
        R'''(I)[-rho^3, rho^3]/6 on [c - rho, c + rho], R, R', R'' at the exact point c, R''' over the ball I; sup < inf r_X,
        bisecting adaptively.
      COVERAGE: every endpoint is an exact (radius 0) dyadic arb number and every bisection point is the exact midpoint
      ball's .mid(); all accepted closed intervals are collected in order and checked in exact arithmetic to start at 0,
      end at m and share endpoints (hi_i == lo_{i+1}), so their union is exactly [0, m] with no gap.
 (4) Cited, quoted in the log from papers/johnstone_GE_2011.txt: Prop. 4.18 (a unique least favourable distribution
     exists, it is symmetric; conversely supp(pi) in M(pi) makes the Bayes rule minimax) and Remark 1 after Thm 4.11
     (a saddlepoint makes the prior least favourable). R <= r on [-m, m] is the left saddlepoint inequality.
 Then the LFP at this m is v* and has exactly K support points.

INTEGRALS. Every integral is flint's acb.integral in u = x - t over [-L, L], L = 16 (the window is centred on t, where
the Gaussian weight phi(x - t) lives; for t in [0, m] this replaces the design's fixed window [-m - L, m + L] and the tail
bound below covers what is outside), with the analytic flag honoured (nan when Z contains 0 on the ball), PLUS an explicit
analytic bound for the tail |u| > L added as a ball radius. On the real line delta^(j) is the (j+1)-th posterior
cumulant of theta in [-m, m], so |e| = |delta - t| <= m + |t|, |delta' - 1| <= m^2, |delta''| <= 2m^3, |delta'''| <= 4m^4,
|delta''''| <= 28m^5, and each integrand is at most (its bound) phi(u) or (A0 + A1|u|) phi(u) (parameter derivatives), with
int_{|u|>L} phi = erfc(L/sqrt2), int_{|u|>L} |u| phi = 2 phi(L).
 delta derivatives: Tweedie, delta = x + l', l = log Z, so delta^(j) = l^(j+1) (+ x, + 1 for j = 0, 1) = posterior cumulant
 kappa_{j+1}; evaluated in the centred-cumulant form and CHECKED against the literal form from the Hermite derivatives
 Z^(j) = sum w (-1)^j He_j(x - theta) phi(x - theta) and against central differences.
 R derivatives (x = t + u, e = delta(t + u) - t, d_j = delta^(j)):
   R' = int 2 e (d1 - 1) phi,  R'' = int 2[(d1 - 1)^2 + e d2] phi,  R''' = int 2[3 (d1 - 1) d2 + e d3] phi,
   R'''' = int 2[3 d2^2 + 4 (d1 - 1) d3 + e d4] phi;  each CHECKED against central differences of the lower one.
 Jacobian: d_p R(t) = int 2 e delta_p phi, d_p R'(t) = int 2[(d1 - 1) delta_p + e delta_p'] phi, with (posterior pi_k,
   g_k = theta_k - delta) d delta/d omega_k = pi_k g_k/omega_k, d delta/d theta_k = pi_k [1 + (x - theta_k) g_k]; CHECKED
   against central differences of F.

PRE-REGISTERED, before the runs:
 (C1) the stored state has K = 7 atoms at m = 5 and K = 14 at m = 10;
 (C2) the rigorous R at the stored (unpolished) point agrees with q484's double trapezoid rule (Prob.rule, Prob.risk)
      to 1e-12 at every atom a_j and at 0;
 (i)   the polished point has max |F| < 1e-30 (rigorous upper bound, arb);
 (ii)  derivative formulas: delta^(j) (j = 1..4) against the Hermite-Tweedie form and central differences, R^(k)
       (k = 1..4) against central differences of R^(k-1), the Jacobian against central differences of F: relative
       error < 1e-10 each;
 (iii) Krawczyk containment at the sweep box (the smallest ladder radius); the ladder is reported upward until it fails;
 (iv)  all weights positive and the atoms strictly ordered in (0, m) in the sweep box;
 (v)   the KKT sweep passes on every piece with the worst margin printed; a piece undecided after 30 bisections or
       more than 20000 pieces is a FAIL with its location printed; the exact coverage check passes.
 PASS needs (C1), (C2) and (i)-(v).
AMENDMENT v2 (after the first runs; no criterion changed): v1 stopped the Newton polish as soon as max |F| < 1e-30. At
m = 10 that left max |F| = 1.2e-35 with scaled cond 3.2e+04, so ||Y Dr F(z)|| = 1.57e-34 exceeded the smallest ladder radius
1e-34, and (iii) FAILED there (||I - Y J(X)|| = 6.3e-12; log q489_m10_v1.log; m = 5 had passed). v2 continues the polish
past the target until the residual no longer falls by a factor 10 (the integration floor, at most 10 steps) and keeps
the best point; the ladder, the sweep box rule and every threshold are as pre-registered.
AMENDMENT v3 (23 Sep 07:40, before the runs at m = 15 and 20): (C1) extended with K = 21 at m = 15 (q486: K = 21 on
(14.3875, 15.0179)) and K = 30 at m = 20 (K = 30 on (19.8584, 20.4419)); nothing else changed.
AMENDMENT v4 (23 Sep, after the m = 20 run failed at 08:00): at m = 20 the polish reached the integration floor max |F| = 5.77e-39
(TOL_F = 1e-40) and the scaled cond 1.85e8 gave ||Y Dr F(z)|| = 3.8e-32 > 1e-34, so (iii) FAILED at the smallest ladder
radius (||I - Y J(X)|| = 3.0e-7; log q489_m20_v3_fail.log). v4 reads the integration tolerance of F from the env TOLF (default
1e-40, unchanged for m = 5, 10, 15); the m = 20 rerun uses TOLF = 1e-48. The ladder, the sweep box rule and every
threshold are as pre-registered.
Usage: py q489_lfp_certificate.py M [PREC]"""

import os, sys, json, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import arb, acb, arb_mat, ctx

MT = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
PREC = int(sys.argv[2]) if len(sys.argv) > 2 else 256
ctx.prec = PREC
T0 = time.time()
L = 16
TOL_F = arb(os.environ.get("TOLF", "1e-40"))  # F at a point, Newton, derivative checks (v4: TOLF env)
TOL_J = arb("1e-20")  # Jacobian entries (point and box)
TOL_S = arb("1e-26")  # sweep point values
TOL_B = arb("1e-8")  # enclosures over a t-ball (dominated by the ball's own width)
NT = arb("1e-30")  # (i)
DCHK = 1e-10  # (ii)
CTRL = 1e-12  # (C2)
LADDER = [1e-34, 1e-30, 1e-26, 1e-22, 1e-18, 1e-14, 1e-10, 1e-8, 1e-6]
MAXDEPTH, MAXPIECES = 30, 20000
PIECE0 = arb(1) / 16  # free stretches are first cut (without evaluation) into pieces no wider than this
HLIST = [0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 1e-6, 1e-7, 1e-8]
EXPECT_K = {
    5.0: 7,
    10.0: 14,
    15.0: 21,
    20.0: 30,
}  # 15 and 20 added 23 Sep 07:40 from q486's continuation, before those runs
I1 = acb(0, 1)
S2P = 1 / (2 * arb.pi()).sqrt()
ERFC_L = (arb(L) / arb(2).sqrt()).erfc()  # int_{|u|>L} phi(u) du
PHI_L2 = 2 * (-arb(L) * L / 2).exp() * S2P  # int_{|u|>L} |u| phi(u) du = 2 phi(L)
NAN = acb(arb("nan"))
NEV = [0]


def fl(a):
    return float(a.mid())


def pm(x):
    """the ball [-x, x] for an arb x >= 0"""
    return arb(0, x)


class Prior:
    """symmetric prior: atoms +-a_j (a list of J arbs, a[-1] = m), weights w_j (each of +-a_j), centre weight c or None"""

    def __init__(self, m, a, w, c):
        self.m, self.a, self.w, self.c = m, list(a), list(w), c
        J = len(a)
        self.J = J
        th = [-t for t in self.a[::-1]] + ([arb(0)] if c is not None else []) + self.a
        om = self.w[::-1] + ([c] if c is not None else []) + self.w
        self.th, self.om, self.K = th, om, len(th)
        self.thc = [acb(t) for t in th]
        self.omc = [acb(o) for o in om]
        self.lnw = [
            acb(om[k].log() - th[k] * th[k] / 2) for k in range(self.K)
        ]  # omega_k phi(x - theta_k) = e^{-x^2/2} e^{lnw_k + theta_k x}/sqrt(2pi)
        off = J + (1 if c is not None else 0)
        self.plus = [off + j for j in range(J)]
        self.minus = [J - 1 - j for j in range(J)]
        self.i0 = J if c is not None else None
        # tail constants: |delta' - 1| <= m^2, |delta''| <= 2m^3, |delta'''| <= 4m^4, |delta''''| <= 28m^5
        self.D1, self.D2, self.D3, self.D4 = m**2, 2 * m**3, 4 * m**4, 28 * m**5

    # ---------------- posterior quantities (Tweedie = posterior cumulants) ----------------
    def post(self, x, order, analytic=False):
        E = [(l + t * x).exp() for l, t in zip(self.lnw, self.thc)]
        Z = E[0]
        for e_ in E[1:]:
            Z = Z + e_
        if analytic and Z.contains(0):
            return None
        iZ = 1 / Z
        p = [e_ * iZ for e_ in E]
        mu = sum(pk * tk for pk, tk in zip(p, self.thc))
        D = [mu]
        g = None
        if order >= 1:
            g = [tk - mu for tk in self.thc]
            q = [pk * gk * gk for pk, gk in zip(p, g)]
            m2 = sum(q)
            D.append(m2)
            if order >= 2:
                q = [qk * gk for qk, gk in zip(q, g)]
                m3 = sum(q)
                D.append(m3)
            if order >= 3:
                q = [qk * gk for qk, gk in zip(q, g)]
                m4 = sum(q)
                D.append(m4 - 3 * m2 * m2)
            if order >= 4:
                q = [qk * gk for qk, gk in zip(q, g)]
                m5 = sum(q)
                D.append(m5 - 10 * m3 * m2)
        return p, g, D

    def tweedie_hermite(self, x, nd=4):
        """literal Tweedie: Z^(j) = sum omega (-1)^j He_j(x - theta) phi(x - theta), l = log Z, delta = x + l', delta' = 1 + l'',
        delta^(j) = l^(j+1) for j >= 2; the log-derivatives from the ratios Z^(j)/Z by the moment-cumulant recursion
        """
        n = nd + 1
        Zj = [acb(0)] * (n + 1)
        for k in range(self.K):
            y = x - self.thc[k]
            ph = self.omc[k] * (-y * y / 2).exp()
            He = [acb(1), y]
            for i in range(1, n):
                He.append(y * He[i] - i * He[i - 1])
            for j in range(n + 1):
                Zj[j] = Zj[j] + (He[j] if j % 2 == 0 else -He[j]) * ph
        mu = [Zj[j] / Zj[0] for j in range(n + 1)]
        from math import comb

        kap = [None] * (n + 1)
        for s in range(1, n + 1):
            kap[s] = mu[s] - sum(comb(s - 1, i - 1) * kap[i] * mu[s - i] for i in range(1, s))
        return [x + kap[1], 1 + kap[2]] + [kap[j + 1] for j in range(2, nd + 1)]

    # ---------------- R and its t-derivatives ----------------
    def rfun(self, t, ks):
        order = max(ks)

        def f(u, analytic):
            NEV[0] += 1
            x = t + u
            P = self.post(x, order, analytic)
            if P is None:
                return NAN
            D = P[2]
            e = D[0] - t
            out = []
            for k in ks:
                if k == 0:
                    out.append(e * e)
                elif k == 1:
                    out.append(2 * e * (D[1] - 1))
                elif k == 2:
                    d1 = D[1] - 1
                    out.append(2 * (d1 * d1 + e * D[2]))
                elif k == 3:
                    d1 = D[1] - 1
                    out.append(2 * (3 * d1 * D[2] + e * D[3]))
                elif k == 4:
                    d1 = D[1] - 1
                    out.append(2 * (3 * D[2] * D[2] + 4 * d1 * D[3] + e * D[4]))
            val = out[0] if len(out) == 1 else out[0] + I1 * out[1]
            return val * (-u * u / 2).exp() * S2P

        return f

    def tailR(self, k, T):
        E = self.m + T
        D1, D2, D3, D4 = self.D1, self.D2, self.D3, self.D4
        G = [
            E * E,
            2 * E * D1,
            2 * (D1 * D1 + E * D2),
            2 * (3 * D1 * D2 + E * D3),
            2 * (3 * D2 * D2 + 4 * D1 * D3 + E * D4),
        ][k]
        return G * ERFC_L

    def R(self, t, ks, tol):
        """[R^(k1)(t), R^(k2)(t)] for a real point or real ball t; two orders packed as real and imaginary parts"""
        t = acb(t)
        I = acb.integral(self.rfun(t, ks), -L, L, abs_tol=tol, rel_tol=tol)
        T = abs(t.real).upper()
        out = [I.real + pm(self.tailR(ks[0], T).upper())]
        if len(ks) > 1:
            out.append(I.imag + pm(self.tailR(ks[1], T).upper()))
        return out

    # ---------------- parameter derivatives (Jacobian) ----------------
    def dparam(self, P, x, kind, j, needx):
        p, g, D = P
        m2 = D[1]
        dp = acb(0)
        dpx = acb(0)
        if kind in ("w", "c"):
            ks = [self.plus[j], self.minus[j]] if kind == "w" else [self.i0]
            for k in ks:
                dp = dp + p[k] * g[k] / self.omc[k]
                if needx:
                    dpx = dpx + p[k] * (g[k] * g[k] - m2) / self.omc[k]
        else:
            for k, s in ((self.plus[j], 1), (self.minus[j], -1)):
                y = x - self.thc[k]
                q = 1 + y * g[k]
                term = p[k] * q
                dp = dp + term if s > 0 else dp - term
                if needx:
                    tx = p[k] * (g[k] * q + g[k] - y * m2)
                    dpx = dpx + tx if s > 0 else dpx - tx
        return dp, dpx

    def pfun(self, t, specs):
        def f(u, analytic):
            NEV[0] += 1
            x = t + u
            P = self.post(x, 1, analytic)
            if P is None:
                return NAN
            D = P[2]
            e = D[0] - t
            d1 = D[1] - 1
            out = []
            for kind, j, what in specs:
                dp, dpx = self.dparam(P, x, kind, j, what == "B")
                out.append(2 * e * dp if what == "A" else 2 * (d1 * dp + e * dpx))
            val = out[0] if len(out) == 1 else out[0] + I1 * out[1]
            return val * (-u * u / 2).exp() * S2P

        return f

    def tailP(self, kind, j, what, T):
        m = self.m
        E = m + T
        if kind in ("w", "c"):
            wl = (self.w[j] if kind == "w" else self.c).lower()
            A0, A1, B0, B1 = 2 * m / wl, arb(0), 5 * m * m / wl, arb(0)
        else:
            A0, A1, B0, B1 = 1 + 2 * m * (T + m), 2 * m, 4 * m + 5 * m * m * (T + m), 5 * m * m
        dA = A0 * ERFC_L + A1 * PHI_L2
        dB = B0 * ERFC_L + B1 * PHI_L2
        return 2 * E * dA if what == "A" else 2 * (self.D1 * dA + E * dB)

    def pR(self, t, specs, tol):
        t = acb(t)
        I = acb.integral(self.pfun(t, specs), -L, L, abs_tol=tol, rel_tol=tol)
        T = abs(t.real).upper()
        out = [I.real + pm(self.tailP(*specs[0], T).upper())]
        if len(specs) > 1:
            out.append(I.imag + pm(self.tailP(*specs[1], T).upper()))
        return out


# ------------------------------------------------------------------------------------------------------------------
d = json.load(open(os.path.join(HERE, "results_q486_lfp_sym.json")))
ROW = min(d["rows"], key=lambda rr: abs(rr["m"] - MT))
J = int(ROW["J"])
CENTRE = bool(ROW["centre"])
MF = float(ROW["m"])
M = arb(MF)
K = 2 * J + (1 if CENTRE else 0)
NV = 2 * J - 1 + (1 if CENTRE else 0)
PRM = [("a", j) for j in range(J - 1)] + [("w", j) for j in range(J)] + ([("c", 0)] if CENTRE else [])


def make_prior(v):
    a = list(v[: J - 1]) + [M]
    w = list(v[J - 1 : 2 * J - 1])
    c = v[2 * J - 1] if CENTRE else None
    return Prior(M, a, w, c)


def F_of(P, tol):
    Rm = P.R(M, (0,), tol)[0]
    F = []
    G = []
    for j in range(J - 1):
        r0, r1 = P.R(P.a[j], (0, 1), tol)
        F.append(r0 - Rm)
        G.append(r1)
    F += G
    if CENTRE:
        F.append(P.R(arb(0), (0,), tol)[0] - Rm)
    F.append(2 * sum(P.w) + (P.c if CENTRE else 0) - 1)
    return F, Rm


def J_of(P, tol):
    n = NV
    Jm = [[arb(0)] * n for _ in range(n)]
    Am = {}
    A0 = {}
    for i in range(0, n, 2):
        pr = PRM[i : i + 2]
        vals = P.pR(M, [(k, j, "A") for k, j in pr], tol)
        for p_, vv in zip(pr, vals):
            Am[p_] = vv
        if CENTRE:
            vals = P.pR(arb(0), [(k, j, "A") for k, j in pr], tol)
            for p_, vv in zip(pr, vals):
                A0[p_] = vv
    for jr in range(J - 1):
        t = P.a[jr]
        R1, R2 = P.R(t, (1, 2), tol)
        for ci, (k, j) in enumerate(PRM):
            A, B = P.pR(t, [(k, j, "A"), (k, j, "B")], tol)
            diag_ = k == "a" and j == jr
            Jm[jr][ci] = A - Am[(k, j)] + (R1 if diag_ else 0)
            Jm[J - 1 + jr][ci] = B + (R2 if diag_ else 0)
    row = 2 * (J - 1)
    if CENTRE:
        for ci, p_ in enumerate(PRM):
            Jm[row][ci] = A0[p_] - Am[p_]
        row += 1
    for ci, (k, j) in enumerate(PRM):
        Jm[row][ci] = arb(2) if k == "w" else (arb(1) if k == "c" else arb(0))
    return Jm


def ruiz(A, sweeps=80):
    m_ = A.shape[0]
    uu = np.ones(m_)
    vv = np.ones(m_)
    B = A.copy()
    for _ in range(sweeps):
        r = np.sqrt(np.maximum(np.abs(B).max(1), 1e-300))
        c = np.sqrt(np.maximum(np.abs(B).max(0), 1e-300))
        uu /= r
        vv /= c
        B = (B / r[:, None]) / c[None, :]
    return uu, vv, B


def diagm(dd):
    return arb_mat(
        [[arb(float(dd[i])) if i == j else arb(0) for j in range(len(dd))] for i in range(len(dd))]
    )


def el():
    return f"[{time.time() - T0:.0f}s]"


if __name__ == "__main__":
    say(
        f"Q489 LFP support certificate: m = {MF} (row closest to {MT}), J = {J}, centre = {CENTRE}, K = {K}, "
        f"{NV} unknowns, prec {PREC} bits, u-window [-{L}, {L}]"
    )
    ok_all = True

    # ---------------- the cited statements ----------------
    say(
        "\nCITED (quoted verbatim from papers/johnstone_GE_2011.txt; the text extraction dropped the Greek letters and some"
        " symbols, so a bracketed reading follows each quote):"
    )
    try:
        lines = (
            open(os.path.join(HERE, "papers", "johnstone_GE_2011.txt"), encoding="utf-8", errors="replace")
            .read()
            .split("\n")
        )
        for (lo_, hi_), reading in (
            (
                (7309, 7312),
                "Proposition 4.18 For the non-linear minimax risk rho_N(tau, eps) given by (4.26) [inf over estimators of"
                " sup over |theta| <= tau of E_theta(hat-delta - theta)^2], a unique least favorable distribution pi_tau exists and"
                " (hat-delta_tau, pi_tau) is a saddlepoint. The distribution pi_tau is symmetric, supp(pi_tau) subset M(pi_tau) and"
                " M(pi_tau) is a finite set. Conversely, if a prior pi satisfies supp(pi) subset M(pi) then hat-delta_pi is minimax.",
            ),
            (
                (7302, 7307),
                "M(pi) = {theta in [-tau, tau] : r(hat-delta_pi, theta) = rbar(hat-delta_pi)}, rbar(hat-delta) = max_{|theta| <= tau} r(hat-delta, theta).",
            ),
            (
                (6875, 6880),
                "Remark 1 [after Theorem 4.11]: a pair (hat-delta*, pi*) is a saddlepoint if for all hat-delta and all pi in P,"
                " B(hat-delta*, pi) <= B(hat-delta*, pi*) <= B(hat-delta, pi*). If a saddlepoint exists, then hat-delta* is a Bayes"
                " rule for pi* (from the right side), and pi* is a least favorable distribution (since the left side implies"
                " B(pi) <= B(pi*) for all pi).",
            ),
            (
                (6910, 6913),
                "Example 4.10 continued: Theta = [-tau, tau] and rho_N(tau, 1) = sup{B(pi) : supp pi subset [-tau, tau]} (4.20).",
            ),
        ):
            say(f"  lines {lo_}-{hi_}:")
            for ln in lines[lo_ - 1 : hi_]:
                if ln.strip():
                    say("    | " + ln.rstrip())
            say(f"    reading: {reading}")
        say(
            "  USE: the certificate shows R(t) <= r = B(pi*) for all t in [-m, m]. Then for every prior pi on [-m, m],"
            " B(delta*, pi) = int R dpi <= r = B(delta*, pi*) (left inequality), and delta* is the Bayes rule for pi*"
            " (right inequality): a saddlepoint, so pi* is least favourable (Remark 1), supp(pi*) subset M(pi*) so delta* is"
            " minimax (Prop. 4.18, converse), and by the uniqueness in Prop. 4.18 pi* is THE least favourable prior."
        )
    except Exception as ex:
        say(
            f"  could not read the source file ({ex}); the cited facts are NOT quoted and the final step is not supported here"
        )
        ok_all = False

    # ---------------- (C1) ----------------
    expK = EXPECT_K.get(round(MF, 6))
    okC1 = expK is not None and K == expK and int(ROW["K"]) == K
    say(
        f"\n(C1) stored row m = {MF}: K = {ROW['K']} (J = {J}, centre {CENTRE} -> 2J + c = {K}); expected {expK}: "
        f"{'PASS' if okC1 else 'FAIL'}; stored residual {ROW['res']:.1e}, stored off-support max(R - r) {ROW['off']:+.2e}"
    )
    ok_all &= okC1

    # ---------------- tails ----------------
    Pst = make_prior([arb(x) for x in ROW["v"]])
    Tm = M
    tl = [Pst.tailR(k, Tm).upper() for k in range(5)]
    tp = [Pst.tailP(k_, j_, wh, Tm).upper() for (k_, j_) in PRM for wh in ("A", "B")]
    tmax = max(tl + tp)
    say(
        f"\nTAILS: erfc(L/sqrt2) = {ERFC_L.str(4, radius=False)}, 2 phi(L) = {PHI_L2.str(4, radius=False)}; tail bounds added as radii:"
        f" R..R'''' {', '.join(x.str(3, radius=False) for x in tl)}; Jacobian integrands max {max(tp).str(3, radius=False)};"
        f" all < 1e-40: {'PASS' if tmax < arb('1e-40') else 'FAIL'}"
    )
    ok_all &= bool(tmax < arb("1e-40"))

    # ---------------- (C2) control against q484's trapezoid ----------------
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    import q484_lfp_bnm as Q

    sys.argv = _argv
    v0 = np.array(ROW["v"])
    a0 = np.concatenate([v0[: J - 1], [MF]])
    w0 = v0[J - 1 : 2 * J - 1]
    th0 = np.concatenate([-a0[::-1], [0.0], a0]) if CENTRE else np.concatenate([-a0[::-1], a0])
    ww0 = np.concatenate([w0[::-1], [v0[-1]], w0]) if CENTRE else np.concatenate([w0[::-1], w0])
    PQ = Q.Prob(MF)
    dQ = PQ.rule(th0, ww0)
    pts = list(a0) + [0.0]
    RQ, _ = PQ.risk(np.array(pts), dQ)
    worst = 0.0
    for tq, rq in zip(pts, RQ):
        ra = Pst.R(arb(float(tq)), (0,), TOL_S)[0]
        diff = abs(fl(ra) - rq)
        worst = max(worst, diff)
        say(f"     t = {tq:.10f}: rigorous R = {ra.str(22)}, q484 trapezoid {rq:.16f}, |diff| {diff:.2e}")
    okC2 = worst < CTRL
    say(
        f"(C2) worst |rigorous - trapezoid| = {worst:.2e} < {CTRL:.0e}: {'PASS' if okC2 else 'FAIL'}   {el()}"
    )
    ok_all &= okC2

    # ---------------- (ii-a) delta derivatives ----------------
    say("\n(ii) derivative-formula checks (relative errors)")
    rels = []
    for xs in (acb("0.77"), acb("3.3"), acb("0.77", "0.4"), acb(MF + 2.5)):
        _, _, Dc_ = Pst.post(xs, 4)
        Dt = Pst.tweedie_hermite(xs, 4)
        r_ = [float(abs(Dc_[j] - Dt[j]).mid()) / max(float(abs(Dc_[j]).mid()), 1e-300) for j in range(5)]
        rels += r_[1:]
        say(
            f"     x = {xs.str(3)}: cumulant form vs Hermite-Tweedie form, delta^(0..4): "
            + ", ".join(f"{q:.1e}" for q in r_)
        )
    for xr in (0.77, 3.3):
        h = arb(2.0**-60)
        x_ = arb(xr)
        Dp = Pst.post(acb(x_ + h), 4)[2]
        Dm = Pst.post(acb(x_ - h), 4)[2]
        D0 = Pst.post(acb(x_), 4)[2]
        r_ = []
        for j in range(1, 5):
            fd = (Dp[j - 1] - Dm[j - 1]) / (2 * h)
            r_.append(float(abs(fd - D0[j]).mid()) / max(float(abs(D0[j]).mid()), 1e-300))
        rels += r_
        say(
            f"     x = {xr}: delta^(j) vs central difference of delta^(j-1), j = 1..4: "
            + ", ".join(f"{q:.1e}" for q in r_)
            + "   (values "
            + ", ".join(f"{float(D0[j].real.mid()):+.4e}" for j in range(5))
            + ")"
        )
    # R derivatives
    tc = float(0.5 * (v0[0] + (v0[1] if J > 2 else MF)))
    hR = arb(2.0**-30)
    Rp = Pst.R(arb(tc) + hR, (0, 1), TOL_F) + Pst.R(arb(tc) + hR, (2, 3), TOL_F)
    Rm_ = Pst.R(arb(tc) - hR, (0, 1), TOL_F) + Pst.R(arb(tc) - hR, (2, 3), TOL_F)
    Rc = Pst.R(arb(tc), (0, 1), TOL_F) + Pst.R(arb(tc), (2, 3), TOL_F) + Pst.R(arb(tc), (4,), TOL_F)
    r_ = []
    for k in range(1, 5):
        fd = (Rp[k - 1] - Rm_[k - 1]) / (2 * hR)
        r_.append(float(abs(fd - Rc[k]).mid()) / max(abs(fl(Rc[k])), 1e-300))  # difference taken in arb
    rels += r_
    say(
        f"     t = {tc:.6f}: R^(k) vs central difference of R^(k-1), k = 1..4: "
        + ", ".join(f"{q:.1e}" for q in r_)
        + "   (R..R'''' = "
        + ", ".join(f"{fl(Rc[k]):+.6e}" for k in range(5))
        + ")   "
        + el()
    )
    okD = max(rels) < DCHK
    say(f"     delta and R formulas: worst {max(rels):.1e} < {DCHK:.0e}: {'PASS' if okD else 'FAIL'}")
    ok_all &= okD

    # ---------------- (i) Newton polish ----------------
    say(
        f"\n(i) Newton polish in arb (F by acb.integral at abs tol {TOL_F.str(1, radius=False)}; the step uses the double midpoint of the rigorous Jacobian)"
    )
    v = [arb(x) for x in ROW["v"]]
    Jd = None
    hist = []
    for it in range(10):
        Pz = make_prior(v)
        F, _ = F_of(Pz, TOL_F)
        nF = max(abs(f).upper() for f in F)
        say(f"     it {it}: max |F| <= {nF.str(3, radius=False)}   {el()}")
        hist.append((fl(nF), v))
        if len(hist) >= 2 and hist[-2][0] < float(NT.mid()) and not (hist[-1][0] < hist[-2][0] / 10):
            say(
                "     (v2) the residual no longer falls by 10x: integration floor reached; the best point is kept"
            )
            break
        if Jd is None:
            Jd = np.array([[fl(x) for x in rr] for rr in J_of(Pz, TOL_J)])
        dv = np.linalg.solve(Jd, np.array([fl(f) for f in F]))
        v = [(v[i] - arb(float(dv[i]))).mid() for i in range(NV)]
    z = min(hist, key=lambda hh: hh[0])[1]
    Pz = make_prior(z)
    Fz, Rmz = F_of(Pz, TOL_F)
    nF = max(abs(f).upper() for f in Fz)
    ok1 = bool(nF < NT)
    say(
        f"(i) polished point: max |F| <= {nF.str(3, radius=False)} {'PASS' if ok1 else 'FAIL'} (target 1e-30); "
        f"r = R(m) = {Rmz.str(30)}"
    )
    say("     a = " + ", ".join(x.str(25, radius=False) for x in z[: J - 1]) + f", {MF}")
    say(
        "     w = "
        + ", ".join(x.str(25, radius=False) for x in z[J - 1 : 2 * J - 1])
        + (f";  c = {z[-1].str(25, radius=False)}" if CENTRE else "")
    )
    ok_all &= ok1

    # ---------------- (ii-b) Jacobian against central differences of F ----------------
    Jz = J_of(Pz, TOL_J)
    Jmid = np.array([[fl(x) for x in rr] for rr in Jz])
    hJ = arb(2.0**-40)
    Jdiff = np.zeros((NV, NV))
    for i in range(NV):
        vp = list(z)
        vp[i] = z[i] + hJ
        vm = list(z)
        vm[i] = z[i] - hJ
        Fp, _ = F_of(make_prior(vp), TOL_F)
        Fm, _ = F_of(make_prior(vm), TOL_F)
        for r_i in range(NV):
            Jdiff[r_i, i] = float(
                abs((Fp[r_i] - Fm[r_i]) / (2 * hJ) - Jz[r_i][i]).mid()
            )  # difference taken in arb
    errJ = Jdiff.max() / np.abs(Jmid).max()
    okJ = errJ < DCHK
    say(
        f"(ii) Jacobian vs central differences of F (h = 2^-40): max |diff| / max |J| = {errJ:.1e} < {DCHK:.0e}: "
        f"{'PASS' if okJ else 'FAIL'}; max entry radius of J(z) {max(fl(x.rad()) for rr in Jz for x in rr):.1e}; cond(J) = {np.linalg.cond(Jmid):.3e}   {el()}"
    )
    ok_all &= okJ

    # ---------------- (iii) Krawczyk ----------------
    Dr, Dc, _ = ruiz(Jmid)
    Js = (Dr[:, None] * Jmid) * Dc[None, :]
    Y = np.linalg.inv(Js)
    Yarb = arb_mat([[arb(float(Y[i, j])) for j in range(NV)] for i in range(NV)])
    DrM, DcM = diagm(Dr), diagm(Dc)
    Fs = arb_mat([[arb(float(Dr[i])) * Fz[i]] for i in range(NV)])
    YF = Yarb * Fs
    yf = [YF[i, 0].abs_upper() for i in range(NV)]
    say(
        f"\n(iii) Krawczyk. scaled cond {np.linalg.cond(Js):.3e}; ||Y Dr F(z)||_inf (rigorous) = {max(yf).str(4, radius=False)}; "
        f"column scales Dc in [{Dc.min():.2e}, {Dc.max():.2e}]"
    )
    Imat = diagm(np.ones(NV))

    def in_domain(Xv):
        a_ = Xv[: J - 1]
        w_ = Xv[J - 1 : 2 * J - 1]
        ok = all(t.lower() > 0 for t in w_) and (not CENTRE or Xv[-1].lower() > 0)
        if J > 1:
            ok = (
                ok
                and a_[0].lower() > 0
                and a_[-1].upper() < M
                and all(a_[i].upper() < a_[i + 1].lower() for i in range(J - 2))
            )
        return bool(ok)

    def krawczyk(r):
        Xv = [arb(z[i].mid(), float(r * Dc[i])) for i in range(NV)]
        if not in_domain(Xv):
            return False, None, None, Xv, "box leaves the domain (weight <= 0 or atoms not ordered in (0, m))"
        JX = J_of(make_prior(Xv), TOL_J)
        C = Imat - Yarb * DrM * arb_mat(JX) * DcM
        rows = [
            sum(C[i, j].abs_upper() for j in range(NV)).upper() for i in range(NV)
        ]  # exact upper bounds: max is decidable
        normC = max(rows)
        ra = arb(r)
        lhs = arb(max(fl((yf[i] + rows[i] * ra) / ra) for i in range(NV)))  # display only
        ok = all((yf[i] + rows[i] * ra) < ra for i in range(NV)) and normC < arb(1)
        return bool(ok), normC, lhs, Xv, ""

    contained = []
    Xsweep = None
    rsweep = None
    for r in LADDER:
        ok, normC, lhs, Xv, why = krawczyk(r)
        if normC is None:
            say(f"     r = {r:.0e}: {why}  not contained   {el()}")
        else:
            say(
                f"     r = {r:.0e}: ||I - Y J(X)||_inf = {normC.str(5, radius=False)}, max_i (|Y F|_i + row_i r)/r = "
                f"{lhs.str(4, radius=False)}  {'CONTAINED' if ok else 'not contained'}; unscaled radii <= {r*Dc.max():.1e}   {el()}"
            )
        if ok:
            contained.append(r)
            if Xsweep is None:
                Xsweep, rsweep = Xv, r
        else:
            break
    ok3 = bool(contained) and contained[0] == LADDER[0]
    if ok3:
        say(
            f"(iii) PASS: exactly one zero v* of F in the box z +/- r Dc for every contained r (largest {max(contained):.0e});"
            f" the sweep uses the smallest, r = {rsweep:.0e}"
        )
    else:
        say(f"(iii) FAIL: the smallest ladder box r = {LADDER[0]:.0e} is not contained")
        say(
            f"\nVERDICT: FAIL - m = {MF}, K = {K}: Krawczyk containment failed at r = {LADDER[0]:.0e}; no sweep.  {el()}"
        )
        sys.exit()
    ok_all &= ok3

    # ---------------- (iv) weights and atoms in the box ----------------
    SX = make_prior(Xsweep)
    ok4 = in_domain(Xsweep)
    wl = [x.lower() for x in SX.w] + ([SX.c.lower()] if CENTRE else [])
    say(
        f"\n(iv) sweep box: min weight lower bound {min(wl).str(8, radius=False)}, atoms ordered in (0, m): "
        f"{'PASS' if ok4 else 'FAIL'}; atom radii max {max(fl(t.rad()) for t in SX.a[:J - 1]) if J > 1 else 0:.1e}, weight radii max "
        f"{max(fl(t.rad()) for t in (SX.w + ([SX.c] if CENTRE else []))):.1e}"
    )
    ok_all &= ok4

    # ---------------- (v) the KKT sweep ----------------
    rX = SX.R(M, (0,), TOL_S)[0]
    rlo = rX.lower()
    say(
        f"\n(v) KKT sweep of R(t) <= r on [0, m] (R even). r_X = R_X(m) = {rX.str(30)} (radius {fl(rX.rad()):.1e})"
    )
    ivs = []  # accepted closed intervals in order: (lo, hi, kind)
    fails = []
    count = [0]
    free_margins = []
    diag_fail = []

    def taylor(a_, b_):
        c = ((a_ + b_) / 2).mid()
        rho = max((c - a_).upper(), (b_ - c).upper())
        R0, R1 = SX.R(c, (0, 1), TOL_S)
        R2 = SX.R(c, (2,), TOL_S)[0]
        R3I = SX.R(arb(c, rho), (3,), TOL_B)[0]
        T = R0 + R1 * pm(rho) + R2 * arb(rho * rho / 2, rho * rho / 2) / 2 + R3I * pm(rho * rho * rho) / 6
        return T, (
            fl(c),
            fl(rho),
            fl(R0) - fl(rX),
            fl(R1) * fl(rho),
            fl(R2) * fl(rho) ** 2 / 2,
            float(R3I.abs_upper()) * fl(rho) ** 3 / 6,
        )

    def piece(a_, b_, depth=0):
        if b_ - a_ > PIECE0:
            mid = ((a_ + b_) / 2).mid()
            return min(piece(a_, mid, depth), piece(mid, b_, depth))
        count[0] += 1
        if count[0] > MAXPIECES:
            fails.append(("too many pieces", fl(a_)))
            ivs.append((a_, b_, "FAILED"))
            return -1.0
        T, comp = taylor(a_, b_)
        mg = rlo - T.upper()
        if mg > 0:
            ivs.append((a_, b_, "free"))
            free_margins.append((fl(mg), comp[0]))
            return fl(mg)
        if depth >= MAXDEPTH:
            fails.append(("undecided piece", fl(a_), fl(b_)))
            ivs.append((a_, b_, "FAILED"))
            if len(diag_fail) < 6:
                diag_fail.append(comp)
                say(
                    f"        undecided piece: c = {comp[0]:.12f}, rho = {comp[1]:.2e}, R(c) - r = {comp[2]:.3e}, |R'| rho = {comp[3]:.2e}, "
                    f"R'' rho^2/2 = {comp[4]:.2e}, sup|R'''(I)| rho^3/6 = {comp[5]:.2e}"
                )
            return fl(mg)
        mid = ((a_ + b_) / 2).mid()
        return min(piece(a_, mid, depth + 1), piece(mid, b_, depth + 1))

    def stretch(lo, hi, label):
        n0 = count[0]
        nm0 = len(free_margins)
        if not (lo < hi):
            say(f"     {label}: EMPTY or inverted stretch [{fl(lo)}, {fl(hi)}] - FAIL")
            fails.append(("stretch", label))
            return
        worst_ = piece(lo, hi)
        say(
            f"     free stretch {label} [{fl(lo):.6f}, {fl(hi):.6f}]: {count[0] - n0} pieces evaluated, "
            f"{len(free_margins) - nm0} accepted, worst margin {worst_:.3e}   {el()}"
        )

    def rpp_negative(c, hfloat, ra, cap, label):
        """largest h in HLIST (<= cap) with R'' < 0 on [c - h', c + h'], h' >= ra + h; returns (lo, hi, h', bound) or None"""
        R2c, R3c = SX.R(c, (2, 3), TOL_S)
        for h in HLIST:
            if h > cap:
                continue
            hw = (ra + arb(h)).upper()
            lo = (c - hw).lower()
            hi = (c + hw).upper()
            if lo < 0:
                lo = arb(0)
            he = max((c - lo).upper(), (hi - c).upper())
            R4I = SX.R(arb(c, he), (4,), TOL_B)[0]
            U = R2c.upper() + R3c.abs_upper() * he + max(R4I.upper(), arb(0)) * he * he / 2
            if U < 0:
                say(
                    f"     {label} at {fl(c):.10f}: R''(c) = {fl(R2c):+.6e}, R'''(c) = {fl(R3c):+.3e}, h = {h:.0e}, "
                    f"sup R'''' on I = {R4I.upper().str(3, radius=False)}, R'' on I <= {U.upper().str(4, radius=False)} < 0  PASS   {el()}"
                )
                return lo, hi, he, U
        say(
            f"     {label} at {fl(c):.10f}: R''(c) = {fl(R2c):+.6e}, R'''(c) = {fl(R3c):+.3e}: no h in the list gives R'' < 0 - FAIL"
        )
        return None

    amid = [t.mid() for t in SX.a[: J - 1]]
    arad = [arb(t.rad()) for t in SX.a[: J - 1]]
    cur = arb(0)
    ok_nb = True
    if CENTRE:
        cap = 0.4 * fl(amid[0]) if J > 1 else 0.4 * MF
        got = rpp_negative(arb(0), None, arb(0), cap, "centre atom: R'' < 0 on [-h, h]")
        if got is None:
            ok_nb = False
            fails.append(("centre", 0.0))
            hi0 = arb(1) / 1024
            ivs.append((arb(0), hi0, "FAILED"))
        else:
            hi0 = got[1]
            ivs.append((arb(0), hi0, "centre"))
        cur = hi0
    for k in range(J - 1):
        left = fl(amid[k] - (amid[k - 1] if k > 0 else 0))
        right = fl((amid[k + 1] if k < J - 2 else M) - amid[k])
        cap = 0.4 * min(left, right)
        got = rpp_negative(amid[k], None, arb(SX.a[k].rad()), cap, f"atom {k + 1}: R'' < 0 on I_k")
        if got is None:
            ok_nb = False
            fails.append((f"atom {k + 1}", fl(amid[k])))
            lo_k = (amid[k] - arb(SX.a[k].rad()) - arb(1e-8)).lower()
            hi_k = (amid[k] + arb(SX.a[k].rad()) + arb(1e-8)).upper()
            stretch(cur, lo_k, f"before atom {k + 1}")
            ivs.append((lo_k, hi_k, "FAILED"))
            cur = hi_k
            continue
        lo_k, hi_k, he, U = got
        stretch(cur, lo_k, f"before atom {k + 1}")
        ivs.append((lo_k, hi_k, f"atom {k + 1}"))
        cur = hi_k
    # wall
    capw = 0.4 * fl(M - (amid[-1] if J > 1 else 0))
    wall_ok = False
    for h in HLIST:
        if h > capw:
            continue
        hw = arb(h)
        cw = (M - hw / 2).mid()
        low = M - hw
        if not (low.rad() == 0 and (cw - hw / 2) == low):
            continue
        R1c, R2c = SX.R(cw, (1, 2), TOL_S)
        R3I = SX.R(arb(cw, hw / 2), (3,), TOL_B)[0]
        LB = R1c.lower() - R2c.abs_upper() * hw / 2 - R3I.abs_upper() * hw * hw / 8
        if LB > 0:
            wall_ok = True
            stretch(cur, low, "before the wall")
            say(
                f"     wall [m - h, m], h = {h:.0e}: R'(m - h/2) = {fl(R1c):+.6e}, R''(m - h/2) = {fl(R2c):+.3e}, sup|R'''| on I = "
                f"{R3I.abs_upper().str(3, radius=False)}, R' on [m - h, m] >= {LB.str(4, radius=False)} > 0  PASS   {el()}"
            )
            ivs.append((low, M, "wall"))
            cur = M
            break
    if not wall_ok:
        say("     wall: no h in the list gives R' > 0 on [m - h, m] - FAIL")
        fails.append(("wall", MF))
        low = (M - arb(1e-8)).lower()
        stretch(cur, low, "before the wall")
        ivs.append((low, M, "FAILED"))
        cur = M

    # coverage, in exact arithmetic
    cov_ok = bool(ivs) and bool(ivs[0][0] == 0) and bool(ivs[-1][1] == M)
    nbad = 0
    for i, (lo, hi, kind) in enumerate(ivs):
        if not (lo.rad() == 0 and hi.rad() == 0 and lo < hi):
            cov_ok = False
            nbad += 1
        if i + 1 < len(ivs) and not bool(ivs[i + 1][0] == hi):
            cov_ok = False
            nbad += 1
    nfailed = sum(1 for iv in ivs if iv[2] == "FAILED")
    kinds = {}
    for iv in ivs:
        kk = iv[2].split()[0]
        kinds[kk] = kinds.get(kk, 0) + 1
    say(
        f"     COVERAGE: {len(ivs)} closed intervals with exact endpoints, first starts at 0: {bool(ivs[0][0] == 0)}, last ends at m: "
        f"{bool(ivs[-1][1] == M)}, consecutive endpoints equal: {nbad == 0}; kinds {kinds}; FAILED intervals {nfailed} -> "
        f"{'PASS' if cov_ok and nfailed == 0 else 'FAIL'}"
    )
    wm = min(free_margins) if free_margins else (float("nan"), float("nan"))
    say(
        f"     pieces evaluated: {count[0]}; worst free-stretch margin (inf r_X - sup R) = {wm[0]:.3e} at t ~ {wm[1]:.6f}; "
        f"failures: {len(fails)}"
    )
    if fails:
        say(f"     failures at: {fails[:12]}")
    ok5 = cov_ok and nfailed == 0 and not fails and ok_nb and wall_ok
    ok_all &= ok5
    unsc = max(rsweep * float(Dc[i]) for i in range(NV))
    say(
        f"\nVERDICT: {'PASS' if ok_all else 'FAIL'} - K = {K}, m = {MF}: Krawczyk box z +/- {rsweep:.0e} Dc (scaled; unscaled radius "
        f"<= {unsc:.1e}; contained up to {max(contained):.0e}), worst sweep margin {wm[0]:.3e}, {count[0]} pieces "
        f"({len(ivs)} intervals); "
        + (
            "the stationary state in the box is a K-atom probability measure satisfying R <= r on [-m, m]."
            if ok_all
            else "see the failures above."
        )
    )
    say(el())
