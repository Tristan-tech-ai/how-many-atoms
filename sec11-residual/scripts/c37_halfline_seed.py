"""c37 = c34_halfline.py copied as text (8 Oct 2026, amendment 2077); ONLY change: env SEED_JSON seeds the first stage (see the line after the initialisation of solved_p). c34 text follows."""

"""c34 = c33c_halfline.py copied as text (8 Oct 2026, amendment 2046); ONLY change: FUNC may be a number k (the functional W P^-k, S = W/P^k, T = -k S/P, bulk constant C = (2 pi)^((1-k)/2)); for such a k the starting law numbers come from the environment LAW0 = "d0,nu0" (the stage keeps this law for its buffer, as c33c does). FUNC = immune or npmle runs the original code paths unchanged.
c33c text follows. c33c = c33_halfline.py copied as text; ONLY change: with env DERIVED_LAW=1 (and OFF1, OFF2) the law used for the buffer
(positions, frozen masses, first guesses) is the derived bulk law through rho^-10 integrated numerically."""
"""c33 = c32 with NO buffer refit: the semi-free and frozen atoms stay on the global law with G = 1 (c32's one-parameter refit
formed a positive feedback with the free atoms next to the interface and drifted). One solve per stage; the interface error is
measured by varying D, B1 and the growth step.
c32 = c30 with a ONE-parameter buffer refit (the phase nu0 only, d0 kept global, window [D - 10, D - 2]; c30's two-parameter
refit was ill-conditioned, d0 and nu0 being nearly collinear over a short deep window, and diverged at D = 38), at most 3 rounds per
stage, and no refit from an unconverged solve (the run stops instead).
c30: robust direct half-line solver (uniform immune or NPMLE), one edge at x = 0, uniform Q on [0, X_end]. Run with `py` (flint).

Construction (changes against q610, each aimed at one cause of its stall):
 (1) Three zones. FREE atoms (depth <= D): position and mass unknown; equations g = C and g' = 0.
     SEMI-FREE atoms (D < d <= D + B1): position fixed, mass unknown; equation g = C. They let the smooth mass level relax, so the
     free zone never meets a frozen block with a different mass scale (q610's suspected fault).
     FROZEN atoms (D + B1 < d <= D + B1 + 2L): position and mass fixed; they only complete P on the integration range.
 (2) Semi-free and frozen positions come from the O(rho^-4) law REFITTED to the solved free atoms with depth in [D - 16, D - 2]
     (local edge numbers), not the global law (about 1e-3 off). Frozen masses are G_loc/rho(d), with G_loc the solved w * rho there.
     Each stage alternates solve and refit until the refitted buffer moves by < 1e-9 (at most 4 rounds).
 (3) C is fixed at the bulk value for mass density G = 1: C = 1/sqrt(2 pi) (immune, S = W/P^2), C = 1 (NPMLE, S = W/P); phi = exp(-t^2/2).
 (4) Newton with the analytic Jacobian (arb_mat products), backtracking, and a Levenberg-Marquardt fallback. Growth in D by STEP from 20.
 Integration: Gauss-Legendre, panels 0.5, 24 nodes, over [0, D + B1 + L]; L = sqrt(2 * 0.69315 * PREC) + 1.
Usage: py c30_halfline.py immune|npmle DMAX PREC STEP B1 TAG. Output c30_<TAG>.json (per stage: depths, masses, residual, buffer law)."""
import sys, json, time, math
import numpy as np
from scipy.optimize import brentq, least_squares
import mpmath as mp
from flint import arb, arb_mat, ctx

say = lambda *a: print(*a, flush=True)
t0 = time.time()
FUNC = sys.argv[1]
DMAX = float(sys.argv[2])
PREC = int(sys.argv[3])
STEP = float(sys.argv[4])
B1 = float(sys.argv[5])
TAG = sys.argv[6]
D_START = float(sys.argv[7]) if len(sys.argv) > 7 else 20.0
ctx.prec = PREC
mp.mp.prec = PREC + 20
L = math.sqrt(2 * 0.69315 * PREC) + 1.0
s0 = 1 / (16 * np.pi**2)
import os

KEXP = {"immune": 2.0, "npmle": 1.0}.get(FUNC) or float(FUNC)
LAW0 = (
    (0.22573, 0.06806)
    if FUNC == "immune"
    else (-0.22676, 0.14770) if FUNC == "npmle" else tuple(float(x) for x in os.environ["LAW0"].split(","))
)
CFIX = (
    arb(1) / arb(2 * mp.pi).sqrt()
    if FUNC == "immune"
    else arb(1) if FUNC == "npmle" else arb(2 * mp.pi) ** arb((1 - KEXP) / 2)
)


def rho_of(d, d0):
    return brentq(
        lambda r: r**3 - 3 * r / (8 * np.pi**2) - 1 / (32 * np.pi**2 * r) - 3 * s0 * (d + d0), 0.05, 400
    )


def N_of(d, d0, nu0):
    r = rho_of(d, d0)
    return 4 * np.pi**2 * r**4 - r**2 + np.log(r) / 6 + nu0


import os as _o

if _o.environ.get("DERIVED_LAW"):  # c33c: buffer on the DERIVED bulk law through rho^-10 (Amendment 933)
    from scipy.integrate import cumulative_simpson as _cs_int
    from scipy.interpolate import CubicSpline as _CSp

    _pi = np.pi
    _k = 1 / (8 * _pi**2)
    _k4 = 1 / (64 * _pi**4) - 1 / (96 * _pi**2)
    _c6 = 1 / (512 * _pi**6) - 13 / (768 * _pi**4)
    _c8 = 1 / (4096 * _pi**8) - 17 / (1536 * _pi**6) - 1 / (3072 * _pi**4)
    _c10 = 1 / (32768 * _pi**10) - 29 / (6144 * _pi**8) - 109 / (368640 * _pi**6)
    _R = np.linspace(0.45, 2.6, 500001)
    _f = 1 + _k / _R**2 + _k4 / _R**4 + _c6 / _R**6 + _c8 / _R**8 + _c10 / _R**10
    _D = _cs_int(_R**2 / (s0 * _f), x=_R, initial=0)
    _N = _cs_int(_R**3 / (s0 * _f), x=_R, initial=0)
    _rinv = _CSp(_D, _R)
    _Nr = _CSp(_R, _N)
    _OFF1 = float(_o.environ["OFF1"])
    _OFF2 = float(_o.environ["OFF2"])

    def rho_of(d, d0=None):
        return float(_rinv(d + _OFF1))

    def N_of(d, d0=None, nu0=None):
        return float(_Nr(_rinv(d + _OFF1))) + _OFF2


def d_of(n, d0, nu0):
    return brentq(lambda d: N_of(d, d0, nu0) - n, 1e-9, 1e4)


def refit(ps, lo, hi, start):
    j = np.arange(1, len(ps) + 1) - 0.5
    ps = np.asarray(ps)
    m = (ps >= lo) & (ps <= hi)
    shift = float(np.mean(j[m] - np.array([N_of(d, start[0], start[1]) for d in ps[m]])))
    return (start[0], start[1] + shift)


# Gauss-Legendre nodes
xs_, ws_ = [], []
n = 24
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
    xs_.append(x)
    ws_.append(2 / ((1 - x * x) * dp * dp))
XG = [arb(mp.nstr(x, int(PREC / 3.2) + 10)) for x in xs_]
WG = [arb(mp.nstr(w, int(PREC / 3.2) + 10)) for w in ws_]
f = lambda t: (-(t * t) / 2).exp()


class Stage:
    def __init__(self, D, law, G, nfree_pos):
        self.D = D
        self.law = law
        # all atoms from the (refitted) law up to D + B1 + 2L; the first nfree_pos keep their solved positions (set later)
        dj = []
        j = 1
        while True:
            d = d_of(j - 0.5, *law)
            if d > D + B1 + 2 * L:
                break
            dj.append(d)
            j += 1
        self.dj = dj
        self.nact = sum(1 for d in dj if d <= D + B1)
        self.XEND = D + B1 + L
        npan = int(math.ceil(self.XEND / 0.5))
        h = arb(self.XEND) / npan
        self.X = []
        self.W = []
        for k in range(npan):
            c0 = h * k + h / 2
            for t, w in zip(XG, WG):
                self.X.append(c0 + h / 2 * t)
                self.W.append(h / 2 * w)
        froz = [(arb(d), arb(G / rho_of(d, law[0]))) for d in dj[self.nact :]]
        self.Pf = []
        for xi in self.X:
            s_ = arb(0)
            for q, m in froz:
                s_ += m * f(xi - q)
            self.Pf.append(s_)
        self.X2 = [xi * xi for xi in self.X]

    def evaluate(self, pos, mass, nfree, jac=True):
        NX = len(self.X)
        na = len(pos)
        Bm = [[f(self.X[i] - pos[k]) for k in range(na)] for i in range(NX)]
        Pv = arb_mat(Bm) * arb_mat([[mk] for mk in mass])
        P = [self.Pf[i] + Pv[i, 0] for i in range(NX)]
        if FUNC == "immune":
            S = [self.W[i] / (P[i] * P[i]) for i in range(NX)]
            T = [-2 * S[i] / P[i] for i in range(NX)]
        elif FUNC == "npmle":
            S = [self.W[i] / P[i] for i in range(NX)]
            T = [-S[i] / P[i] for i in range(NX)]
        else:
            S = [self.W[i] / ((P[i].log() * arb(KEXP)).exp()) for i in range(NX)]
            T = [-arb(KEXP) * S[i] / P[i] for i in range(NX)]
        BmT = arb_mat(Bm).transpose()
        V = BmT * arb_mat([[S[i], self.X[i] * S[i], self.X2[i] * S[i]] for i in range(NX)])
        g0 = [V[k, 0] for k in range(na)]
        g1 = [V[k, 1] - pos[k] * V[k, 0] for k in range(na)]
        g2 = [V[k, 2] - 2 * pos[k] * V[k, 1] + (pos[k] * pos[k] - 1) * V[k, 0] for k in range(na)]
        E = [g0[k] - CFIX for k in range(na)] + [g1[k] for k in range(nfree)]
        if not jac:
            return E
        M0 = BmT * arb_mat([[Bm[i][k] * T[i] for k in range(na)] for i in range(NX)])
        M1 = BmT * arb_mat([[Bm[i][k] * T[i] * self.X[i] for k in range(na)] for i in range(NX)])
        M2 = BmT * arb_mat([[Bm[i][k] * T[i] * self.X2[i] for k in range(na)] for i in range(NX)])
        nu = nfree + na
        J = arb_mat(nu, nu)  # unknowns: positions of free atoms (nfree), masses of all active atoms (na)
        for j in range(na):  # weight equations
            for k in range(nfree):
                J[j, k] = (mass[k] * (M1[j, k] - pos[k] * M0[j, k])).mid()
            for k in range(na):
                J[j, nfree + k] = M0[j, k].mid()
            if j < nfree:
                J[j, j] = (J[j, j] + g1[j]).mid()
        for jj in range(nfree):  # position equations
            r = na + jj
            y = pos[jj]
            for k in range(nfree):
                J[r, k] = (mass[k] * (M2[jj, k] - (y + pos[k]) * M1[jj, k] + y * pos[k] * M0[jj, k])).mid()
            for k in range(na):
                J[r, nfree + k] = (M1[jj, k] - y * M0[jj, k]).mid()
            J[r, jj] = (J[r, jj] + g2[jj]).mid()
        return E, J

    def solve(self, pos, mass, nfree, tol, itmax=40):
        na = len(pos)
        r = None
        for it in range(itmax):
            E, J = self.evaluate(pos, mass, nfree)
            r = max(abs(e.mid()) for e in E)
            if r < tol:
                return pos, mass, float(r), it
            rhs = arb_mat([[(-e).mid()] for e in E])
            st = J.solve(rhs)
            ok = False
            lam = arb(1)
            for _ in range(20):
                pn = [arb((pos[k] + lam * st[k, 0]).mid()) for k in range(nfree)] + pos[nfree:]
                mn = [arb((mass[k] + lam * st[nfree + k, 0]).mid()) for k in range(na)]
                rn = max(abs(e.mid()) for e in self.evaluate(pn, mn, nfree, jac=False))
                if rn < r:
                    pos, mass, ok = pn, mn, True
                    break
                lam /= 2
            if not ok:  # Levenberg-Marquardt fallback
                JT = J.transpose()
                A0 = JT * J
                gv = JT * arb_mat([[e.mid()] for e in E])
                mu = arb("1e-8")
                nu_ = A0.nrows()
                for _ in range(30):
                    A = arb_mat(nu_, nu_)
                    for i in range(nu_):
                        for k in range(nu_):
                            A[i, k] = A0[i, k] + (mu * A0[i, i] if i == k else 0)
                    dv = A.solve(gv)
                    pn = [arb((pos[k] - dv[k, 0]).mid()) for k in range(nfree)] + pos[nfree:]
                    mn = [arb((mass[k] - dv[nfree + k, 0]).mid()) for k in range(na)]
                    rn = max(abs(e.mid()) for e in self.evaluate(pn, mn, nfree, jac=False))
                    if rn < r:
                        pos, mass, ok = pn, mn, True
                        break
                    mu *= 10
            if not ok:
                return pos, mass, float(r), it
        E = self.evaluate(pos, mass, nfree, jac=False)
        return pos, mass, float(max(abs(e.mid()) for e in E)), itmax


out = {"func": FUNC, "PREC": PREC, "B1": B1, "L": L, "stages": []}
law = LAW0
G = 1.0
solved_p = None
solved_m = None
if _o.environ.get(
    "SEED_JSON"
):  # c37 (8 Oct 2026, amendment 2077): ONLY change: seed the first stage with the converged positions and masses of the stage D = D_START of another stage file (a run on the fitted law of the same functional)
    _sd = [x for x in json.load(open(_o.environ["SEED_JSON"]))["stages"] if x["conv"] and x["D"] == D_START][
        -1
    ]
    solved_p = list(_sd["p"][: _sd["nfree"]])
    solved_m = list(_sd["m"])
Ds = list(np.arange(D_START, DMAX + 1e-9, STEP))
if Ds[-1] < DMAX - 1e-9:
    Ds.append(DMAX)
for D in Ds:
    for rnd in range(3):
        st = Stage(D, law, G, 0)
        nfree = sum(1 for d in st.dj if d <= D)
        na = st.nact
        pos0 = [float(d) for d in st.dj[:na]]
        mass0 = [G / rho_of(d, law[0]) for d in st.dj[:na]]
        if solved_p is not None:
            k0 = min(len(solved_p), nfree)
            pos0[:k0] = solved_p[:k0]
            mass0[: min(len(solved_m), na)] = solved_m[: min(len(solved_m), na)]
        pos = [arb(x) for x in pos0]
        mass = [arb(x) for x in mass0]
        tol = math.exp(-4 * math.pi**2 * rho_of(D + B1, law[0]) ** 2) * 1e-9
        pos, mass, r, it = st.solve(pos, mass, nfree, tol)
        ps = [float(t.mid()) for t in pos]
        ms = [float(t.mid()) for t in mass]
        conv = r < tol * 1e3
        newlaw = law
        Gl = (
            float(
                np.median([ms[k] * rho_of(ps[k], law[0]) for k in range(nfree) if D - 10 <= ps[k] <= D - 2])
            )
            if D >= 22
            else 1.0
        )
        moved = (
            max(abs(d_of(j - 0.5, *newlaw) - d_of(j - 0.5, *law)) for j in range(nfree + 1, na + 1))
            if na > nfree
            else 0.0
        )
        say(
            f"D = {D:.1f} round {rnd}: {nfree} free + {na - nfree} semi-free, {it} steps, max|E| {r:.2e} (tol {tol:.1e}, conv {conv}); "
            f"law ({newlaw[0]:+.6f}, {newlaw[1]:+.6f}), G {Gl:.6f}, buffer moved {moved:.1e}; depths 1-4 {[round(x, 9) for x in ps[:4]]}   [{time.time() - t0:.0f}s]"
        )
        solved_p, solved_m = ps[:nfree], ms[:na]
        out["stages"].append(
            {
                "D": D,
                "round": rnd,
                "nfree": nfree,
                "na": na,
                "p": ps,
                "m": ms,
                "res": r,
                "tol": tol,
                "conv": bool(conv),
                "law": list(law),
                "G": G,
                "newlaw": list(newlaw),
                "Gl": Gl,
                "moved": moved,
            }
        )
        json.dump(out, open(f"c33_{TAG}.json", "w"), indent=1)
        if not conv:
            say("STOP: unconverged")
            raise SystemExit
        if moved < 1e-7 or D < 22:
            break
say(f"done [{time.time() - t0:.0f}s]")
