"""q626 = q607 copied as text; ONLY change: the seed counting function is the c33 NPMLE half-line (law + residual spline;
env HL_JSON, HL_STAGE), used with D0/NU0 = the half-line law numbers."""

"""q607 (01 Oct 00:18:26, date stamp): c19 (centre_scripts/c19_full_chain.py) copied as text for the NPMLE functional: the weight is
S = W/P (KKT c(y) = int phi(x - y) q/P dx) instead of W/P^2, so T = dS/dP = -S/P; the law guess uses NPMLE edge numbers (d0, nu0) =
(-0.2153, 0.1415) (tail-included fit of the q601 state at a = 22.7484). Control before any use: the 200-bit NPMLE transitions 29 -> 30
(24.0387045-24.0387052) and 30 -> 31 (24.6721005-24.6721012) must be reproduced inside their brackets. Output files q607_<tag>.json.
c19 = FULL CHAIN: every atom free (the real immune problem on [-a, a], edges included), C fixed at its initial value (the KKT
positions and the ratios g/C are invariant under the mass scaling w -> lam w, C -> lam^-2 C, so the normalisation sum w = 1 is not needed).
Built by text edits of c17 (NF = K//2, no frozen atoms, no C unknown). c17 = c16 with the FROZEN atoms placed on the sharp profile (law + deltaM(x/v)) instead of the bare law. c16: discrete immune chain with the outer atoms frozen on the O(rho^-4) law and the central atoms free; the transition is located by
stepping a (hence the central phase e = 2N(a) - K) until g(0) = C (K even, birth) or g''(0) = 0 (K odd, split). Run with `py` (flint).

Model (stated choice):
- Symmetric chain on [-a, a], uniform Q, immune KKT: g(y) = int phi(x - y)/P^2 dx, C level, phi(t) = exp(-t^2/2) (normalisation
  constants cancel in the conditions). The j-th atom from each edge sits at d_j = N^{-1}(j - 1/2), N the O(rho^-4) law with the immune
  edge numbers (d0, nu0) = (0.24568, 0.05644) (the tail-included fit of the 200-bit row a = 25.5581); every atom carries mass G/rho(d_j)
  with G = 1 (the measured immune mass density is constant to 2e-4 in the bulk, c16 docstring check in CENTRE_LAG_DISCRETE.md).
- The innermost NF atoms per side (and the centre atom when K is odd) are FREE: positions and masses. All other atoms are FROZEN on the law.
- Unknowns: free positions, free masses, (centre mass), and the level C. Equations: g(p_j) = C and g'(p_j) = 0 at every free atom,
  g(0) = C at a centre atom, and the weight condition g(q_1) = C at the first frozen atom (it fixes C against the frozen mass level;
  the global normalisation sum w = 1 of the full problem has no local analogue).
- Integrals over x in [0, XI] (folded), XI = min(a, p_NF + L1), L1 = sqrt(2 * 1.2 * prec * ln 2); atoms with |x| > XI + L1 are dropped.
  Gauss-Legendre panels of width 0.5 with NGL nodes. Newton with the analytic Jacobian (arb_mat products), midpoints only.
Transition: secant in a on s(a) = g(0)/C - 1 (even K) or g''(0)/C (odd K). Output: e_t = 2N(a_t) - K and c = 1/2 - e_t."""
import sys, json, time, math, os
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import mpmath as mp
from flint import arb, arb_mat, ctx

s0 = 1 / (16 * np.pi**2)
import os as _os

D0, NU0 = float(_os.environ.get("D0", "-0.2153")), float(
    _os.environ.get("NU0", "0.1415")
)  # q607: NPMLE edge numbers (initial guess only; env override v3)


def rho_of(d, d0=D0):
    return brentq(
        lambda r: r**3 - 3 * r / (8 * np.pi**2) - 1 / (32 * np.pi**2 * r) - 3 * s0 * (d + d0), 0.2, 80
    )


def N_of(d, d0=D0, nu0=NU0):
    r = rho_of(d, d0)
    return 4 * np.pi**2 * r**4 - r**2 + np.log(r) / 6 + nu0


_N_law = N_of
import json as _js
from scipy.interpolate import CubicSpline as _CS

_HL = _os.environ.get("HL_JSON", "c33_npm_D66.json")
_HLD = float(_os.environ.get("HL_STAGE", "0"))
_stl = _js.load(open(_HL))["stages"]
_st = [x for x in _stl if x["D"] == _HLD][0] if _HLD else _stl[-1]
_p = np.array(_st["p"][: _st["nfree"]])
_p = _p[_p <= _st["D"] - 20]
_cs = _CS(_p, np.arange(1, len(_p) + 1) - 0.5 - np.array([_N_law(d) for d in _p]))
_PL = float(_p[-1])


def N_of(d, d0=D0, nu0=NU0):  # q626: NPMLE half-line counting function (law + residual spline), seed only
    return _N_law(d, d0, nu0) + float(_cs(min(max(d, float(_p[0])), _PL)))


def d_of_N(n):
    return brentq(lambda d: N_of(d) - n, 1e-9, 1e5)


def dM(xi):
    xi = abs(xi)
    return 1 / 12 if xi == 0 else quad(lambda t: t * (1 / np.tanh(np.pi * t) - 1), xi, np.inf, limit=200)[0]


say = lambda *a: print(*a, flush=True)


class Chain:
    def __init__(self, K, NF, prec, ngl=24, panel=0.5):
        self.K, self.NF, self.centre = K, NF, (K % 2 == 1)
        ctx.prec = prec
        self.prec = prec
        self.L1 = math.sqrt(2 * 1.25 * prec * math.log(2)) + 1.0
        mp.mp.prec = prec + 20
        xg, wg = mp.gauss_quadrature(ngl, "legendre") if hasattr(mp, "gauss_quadrature") else self._gl(ngl)
        self.xg = [arb(mp.nstr(x, int(prec / 3.2) + 10)) for x in xg]
        self.wg = [arb(mp.nstr(w, int(prec / 3.2) + 10)) for w in wg]
        self.panel = panel
        self.npos = K // 2  # positive atoms (excluding a centre atom)
        self.dj = [d_of_N(j - 0.5) for j in range(1, self.npos + 1)]  # j = 1 outermost ... npos innermost

    def _gl(self, n):
        xs, ws = [], []
        for i in range(1, n + 1):
            x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            for _ in range(200):
                p0, p1 = mp.mpf(1), x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dp
                x -= dx
                if abs(dx) < mp.mpf(2) ** (-mp.mp.prec + 8):
                    break
            xs.append(x)
            ws.append(2 / ((1 - x * x) * dp * dp))
        return xs, ws

    def law_state(self, a):
        """law positions (innermost first) and masses of all positive atoms, and the sharp-profile guess for the free ones"""
        xs = [a - d for d in self.dj][::-1]  # innermost first
        ms = [1 / rho_of(d) for d in self.dj][::-1]
        v = 4 * np.pi * rho_of(a)
        XL = xs[self.NF - 1] + 2 * self.L1 + 2 if len(xs) >= self.NF else 1e9
        for k in range(self.NF, len(xs)):  # c17: frozen atoms on the sharp profile
            if xs[k] > XL or xs[k] / v > 3:
                continue
            j = self.npos - k
            try:
                xs[k] = brentq(
                    lambda x: N_of(a - x) - dM(x / v) - (j - 0.5), xs[k] - 0.5, min(xs[k] + 0.5, a - 1e-9)
                )
            except ValueError:
                pass
        guess = []
        for k in range(self.NF):
            j = self.npos - k  # index from the edge
            f = lambda x: N_of(a - x) - dM(x / v) - (j - 0.5)
            try:
                guess.append(brentq(f, max(1e-6, xs[k] - 1.0), xs[k] + 1.0))
            except ValueError:
                guess.append(xs[k])
        if os.environ.get(
            "EDGE_SEED"
        ):  # q607 v2 (00:25): the edge layer is a fixed half-line structure (873); take the
            import json as _j  # outermost NE atoms (depth from the edge and mass) from a converged state

            hh = [h for h in _j.load(open(os.environ["EDGE_SEED"]))["hist"] if "p" in h][-1]
            sa = hh["a"]
            o = np.argsort(hh["p"])[::-1]
            sp = np.array(hh["p"])[o]
            sm = np.array(hh["m"])[o]  # outermost first
            sd = sa - sp
            NE = int(os.environ.get("EDGE_N", "8"))
            scl = np.median([sm[jj] * rho_of(sd[jj]) for jj in range(4, min(10, len(sd)))])
            for jj in range(min(NE, len(sd))):
                k = self.npos - 1 - jj
                if 0 <= k < len(xs):
                    xs[k] = a - sd[jj]
                    ms[k] = sm[jj] / scl
                    if k < self.NF:
                        guess[k] = a - sd[jj]
        return xs, ms, guess

    def setup(self, a):
        xs, ms, guess = self.law_state(a)
        self.a = a
        NF = self.NF
        XI = a
        self.XI = XI
        npan = int(math.ceil(XI / self.panel))
        h = arb(XI) / npan
        self.x = []
        self.W = []
        for k in range(npan):
            c0 = h * k + h / 2
            for t, w in zip(self.xg, self.wg):
                self.x.append(c0 + h / 2 * t)
                self.W.append(h / 2 * w)
        froz = [(arb(x), arb(m)) for x, m in zip(xs[NF:], ms[NF:]) if x < XI + self.L1]
        self.q1 = arb(xs[NF]) if len(xs) > NF else None
        self.nfroz = len(froz)
        self.Pf = []
        for xi in self.x:
            s = arb(0)
            for q, m in froz:
                s += m * (self.f(xi - q) + self.f(xi + q))
            self.Pf.append(s)
        return xs, ms, guess

    @staticmethod
    def f(t):
        return (-(t * t) / 2).exp()

    def evaluate(self, u, jac=True, extra=()):
        NF = self.NF
        p = u[:NF]
        m = u[NF : 2 * NF]
        C = self.Cfix
        m0 = u[2 * NF] if self.centre else None
        N = len(self.x)
        Bp = [[self.f(xi - pj) for pj in p] for xi in self.x]
        Bm = [[self.f(xi + pj) for pj in p] for xi in self.x]
        f0 = [self.f(xi) for xi in self.x] if self.centre or True else None
        P = []
        for i in range(N):
            s = self.Pf[i]
            for j in range(NF):
                s += m[j] * (Bp[i][j] + Bm[i][j])
            if self.centre:
                s += m0 * f0[i]
            P.append(s)
        S = [self.W[i] / P[i] for i in range(N)]
        T = [-S[i] / P[i] for i in range(N)]  # q607: NPMLE weight

        def kern(y, i):
            xi = self.x[i]
            em, ep = self.f(xi - y), self.f(xi + y)
            dm_, dp_ = xi - y, xi + y
            return em + ep, dm_ * em - dp_ * ep, (dm_ * dm_ - 1) * em + (dp_ * dp_ - 1) * ep

        # evaluation rows
        g0 = []
        g1 = []
        g2 = []
        K0 = []
        K1 = []
        for j in range(NF):
            k0 = [Bp[i][j] + Bm[i][j] for i in range(N)]
            k1 = [(self.x[i] - p[j]) * Bp[i][j] - (self.x[i] + p[j]) * Bm[i][j] for i in range(N)]
            k2 = [
                ((self.x[i] - p[j]) ** 2 - 1) * Bp[i][j] + ((self.x[i] + p[j]) ** 2 - 1) * Bm[i][j]
                for i in range(N)
            ]
            g0.append(sum(k0[i] * S[i] for i in range(N)))
            g1.append(sum(k1[i] * S[i] for i in range(N)))
            g2.append(sum(k2[i] * S[i] for i in range(N)))
            K0.append(k0)
            K1.append(k1)
        k00 = [kern(arb(0), i) for i in range(N)]
        gc0 = sum(k00[i][0] * S[i] for i in range(N))
        gc2 = sum(k00[i][2] * S[i] for i in range(N))
        E = [g0[j] - C for j in range(NF)] + [g1[j] for j in range(NF)] + ([gc0 - C] if self.centre else [])
        out = {"E": E, "g0c": gc0, "g2c": gc2, "C": C}
        for y in extra:
            ky = [kern(arb(y), i) for i in range(N)]
            out.setdefault("extra", []).append(sum(ky[i][0] * S[i] for i in range(N)))
        if not jac:
            return out
        rows = (
            [[K0[j][i] * T[i] for i in range(N)] for j in range(NF)]
            + [[K1[j][i] * T[i] for i in range(N)] for j in range(NF)]
            + ([[k00[i][0] * T[i] for i in range(N)]] if self.centre else [])
        )
        R = arb_mat(rows)
        B = arb_mat(
            [[Bp[i][j] + Bm[i][j] for j in range(NF)] + ([f0[i]] if self.centre else []) for i in range(N)]
        )
        Bd = arb_mat(
            [
                [(self.x[i] - p[j]) * Bp[i][j] - (self.x[i] + p[j]) * Bm[i][j] for j in range(NF)]
                for i in range(N)
            ]
        )
        RB = R * B
        RBd = R * Bd
        n = len(u)
        J = arb_mat(n, n)
        nr = len(rows)
        for r in range(nr):
            for j in range(NF):
                J[r, j] = (m[j] * RBd[r, j]).mid()
                J[r, NF + j] = RB[r, j].mid()
            if self.centre:
                J[r, 2 * NF] = RB[r, NF].mid()
        for j in range(NF):
            J[j, j] = (J[j, j] + g1[j]).mid()
            J[NF + j, j] = (J[NF + j, j] + g2[j]).mid()
        out["J"] = J
        return out

    def solve(self, u, tol_rel=1e-8, itmax=40):
        """Newton; convergence when max|E| < tol_abs = 1e-6 * |g0c - C| scale set by the bulk ripple (passed in via tol_abs)."""
        r = None
        for it in range(itmax):
            o = self.evaluate(u)
            E = o["E"]
            r = max(abs(e.mid()) for e in E)
            if r < self.tol_abs:
                return u, float(r), it, o
            rhs = arb_mat([[(-e).mid()] for e in E])
            st = o["J"].solve(rhs)
            lam = arb(1)
            ok = False
            for _ in range(25):
                un = [(u[i] + lam * st[i, 0]).mid() for i in range(len(u))]
                un = [arb(t) for t in un]
                rn = max(abs(e.mid()) for e in self.evaluate(un, jac=False)["E"])
                if rn < r:
                    u, ok = un, True
                    break
                lam /= 2
            if not ok:
                return u, float(r), it, o
        o = self.evaluate(u, jac=False)
        return u, float(max(abs(e.mid()) for e in o["E"])), itmax, o


def run(K, NF, prec, a_start, a_step, maxsteps=12, ngl=24, log=None):
    ch = Chain(K, NF, prec, ngl=ngl)
    rho_c = rho_of(a_start)
    ch.tol_abs = arb(math.exp(-4 * math.pi**2 * rho_c**2) * 1e-7)
    xs, ms, guess = ch.setup(a_start)
    u = (
        [arb(x) for x in guess]
        + [arb(mm) for mm in ms[:NF]]
        + ([arb(1 / rho_of(a_start))] if ch.centre else [])
    )
    ch.Cfix = arb(0)
    o = ch.evaluate(u, jac=False)
    ch.Cfix = arb(o["E"][0].mid())  # C fixed at g(p_1) of the guess
    hist = []
    a = a_start
    prev = None
    for step in range(maxsteps):
        if step > 0:
            xs_new, ms_new, _ = ch.setup(a)
        t0 = time.time()
        u, r, it, o = ch.solve(u)
        C = o["C"]
        s = float(((o["g2c"] if ch.centre else (o["g0c"] - C)) / C).mid())
        e = 2 * N_of(a) - K
        rec = dict(
            a=a,
            e=e,
            s=s,
            res=r,
            it=it,
            secs=round(time.time() - t0, 1),
            p=[float(t.mid()) for t in u[:NF]],
            m=[float(t.mid()) for t in u[NF : 2 * NF]],
        )
        hist.append(rec)
        say(f"   a = {a:.6f}  e = {e:+.6f}  s = {s:+.4e}  |E| = {r:.1e}  it {it}  [{rec['secs']}s]")
        if log:
            json.dump(hist, open(log, "w"), indent=1)
        if prev is not None and prev["s"] * s < 0:
            # secant refinement
            lo, hi = (prev, rec) if prev["a"] < rec["a"] else (rec, prev)
            u_hi = u
            for _ in range(6):
                at = lo["a"] - lo["s"] * (hi["a"] - lo["a"]) / (hi["s"] - lo["s"])
                ch.setup(at)
                u2, r2, it2, o2 = ch.solve(u_hi)
                C2 = o2["C"]
                s2 = float(((o2["g2c"] if ch.centre else (o2["g0c"] - C2)) / C2).mid())
                e2 = 2 * N_of(at) - K
                rr = dict(a=at, e=e2, s=s2, res=r2, it=it2)
                hist.append(rr)
                say(f"   secant a = {at:.7f}  e = {e2:+.6f}  s = {s2:+.4e}  |E| = {r2:.1e}")
                if log:
                    json.dump(hist, open(log, "w"), indent=1)
                if s2 < 0:
                    lo = rr
                else:
                    hi = rr
                if abs(hi["a"] - lo["a"]) < 2e-6 or abs(s2) < 1e-6 * abs(prev["s"]):
                    break
            at = lo["a"] - lo["s"] * (hi["a"] - lo["a"]) / (hi["s"] - lo["s"])
            et = 2 * N_of(at) - K
            return dict(K=K, NF=NF, a_t=at, e_t=et, c=0.5 - et, rho=rho_of(at), hist=hist)
        prev = rec
        a = a + a_step
    return dict(K=K, NF=NF, a_t=None, hist=hist)


if __name__ == "__main__":
    K = int(sys.argv[1])
    NF = K // 2
    prec = int(sys.argv[3])
    a0 = float(sys.argv[4])
    da = float(sys.argv[5])
    tag = sys.argv[6] if len(sys.argv) > 6 else f"K{K}_NF{NF}"
    say(
        f"q607 (NPMLE): K = {K}, NF = {NF}, prec = {prec}, a0 = {a0}, da = {da}, rho(a0) = {rho_of(a0):.4f}, e(a0) = {2*N_of(a0) - K:+.5f}"
    )
    res = run(K, NF, prec, a0, da, log=f"q607_{tag}.json")
    if res.get("a_t"):
        say(
            f"RESULT {tag}: a_t = {res['a_t']:.7f}, rho = {res['rho']:.4f}, e_t = {res['e_t']:.6f}, c = {res['c']:.5f}"
        )
    else:
        say(f"RESULT {tag}: no sign change in {len(res['hist'])} steps")
    json.dump(res, open(f"q607_{tag}.json", "w"), indent=1)
