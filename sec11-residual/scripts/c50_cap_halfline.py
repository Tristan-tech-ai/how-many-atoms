"""c50: half-line solver for the amplitude-constrained Gaussian CAPACITY (c33's construction with the capacity KKT). Run with `py`.

Model: inputs x >= 0 (edge at 0), output density P(y) = sum_j w_j phin(y - x_j), phin(t) = exp(-t^2/2)/sqrt(2 pi) (NORMALISED, so a unit
mass density gives P = 1 in the bulk), KKT kernel g(x) = -int phin(y - x) log P(y) dy (the constant -h(phin) dropped), so the bulk level
for unit mass density is C = 0 exactly (up to O(ripple^2)). Outputs over y in [-LY, X_end] (the density has a tail below the edge).
Atoms: j = 1 is FIXED at the wall x = 0 (mass unknown, equation g = C only); free atoms 0 < d <= D (position and mass; g = C, g' = 0);
semi-free D < d <= D + B1 (law position, mass unknown; g = C); frozen beyond, to D + B1 + 2L (law position, mass 1/rho).
In c33 terms: S_i = -W_i log P_i, T_i = dS/dP = -W_i/P_i. Newton with the analytic Jacobian (arb_mat products), growth in D.
Buffer law: the O(rho^-4) law with capacity edge numbers; start (d0, nu0) = (0.0558, 0.5482) (plain fit of the K = 25 state, atoms 7 or
more from its centre, c4); after each stage the numbers are refitted ONCE on solved atoms with 4 <= d <= D - 12 (far from the interface,
unlike c30/c32 whose interface-adjacent refits fed back).
Usage: py c50_cap_halfline.py DMAX PREC STEP B1 TAG [REFIT=1]. Output c50_<TAG>.json."""

import sys, json, time, math
import numpy as np
from scipy.optimize import brentq, least_squares
import mpmath as mp
from flint import arb, arb_mat, ctx

say = lambda *a: print(*a, flush=True)
t0 = time.time()
DMAX = float(sys.argv[1])
PREC = int(sys.argv[2])
STEP = float(sys.argv[3])
B1 = float(sys.argv[4])
TAG = sys.argv[5]
REFIT = int(sys.argv[6]) if len(sys.argv) > 6 else 1
ctx.prec = PREC
mp.mp.prec = PREC + 20
L = math.sqrt(2 * 0.69315 * PREC) + 1.0
LY = L
s0 = 1 / (16 * np.pi**2)


def rho_of(d, d0):
    return brentq(
        lambda r: r**3 - 3 * r / (8 * np.pi**2) - 1 / (32 * np.pi**2 * r) - 3 * s0 * (d + d0), 0.02, 400
    )


def N_of(d, d0, nu0):
    r = rho_of(d, d0)
    return 4 * np.pi**2 * r**4 - r**2 + np.log(r) / 6 + nu0


def d_of(n, d0, nu0):
    return brentq(lambda d: N_of(d, d0, nu0) - n, 1e-6, 1e4)


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
NRM = 1 / arb(2 * mp.pi).sqrt()
f = lambda t: NRM * (-(t * t) / 2).exp()
CFIX = arb(0)


class Stage:
    def __init__(self, D, law):
        self.D = D
        self.law = law
        dj = [0.0]
        j = 2
        while True:
            d = d_of(j - 0.5, *law)
            if d > D + B1 + 2 * L:
                break
            dj.append(d)
            j += 1
        self.dj = dj
        self.nact = sum(1 for d in dj if d <= D + B1)
        self.nfree = sum(1 for d in dj[1:] if d <= D)
        self.XEND = D + B1 + L
        a0 = -LY
        span = self.XEND - a0
        npan = int(math.ceil(span / 0.5))
        h = arb(span) / npan
        self.X = []
        self.W = []
        for k in range(npan):
            c0 = arb(a0) + h * k + h / 2
            for t, w in zip(XG, WG):
                self.X.append(c0 + h / 2 * t)
                self.W.append(h / 2 * w)
        froz = [(arb(d), arb(1 / rho_of(d, law[0]))) for d in dj[self.nact :]]
        self.Pf = []
        for xi in self.X:
            s_ = arb(0)
            for q, m in froz:
                s_ += m * f(xi - q)
            self.Pf.append(s_)
        self.X2 = [xi * xi for xi in self.X]

    def evaluate(self, pos, mass, nfree, jac=True):
        # ordering: pos[:nfree] free (position unknown), pos[nfree] the wall atom (x = 0), then semi-free
        NX = len(self.X)
        na = len(pos)
        Bm = [[f(self.X[i] - pos[k]) for k in range(na)] for i in range(NX)]
        Pv = arb_mat(Bm) * arb_mat([[mk] for mk in mass])
        P = [self.Pf[i] + Pv[i, 0] for i in range(NX)]
        S = [-self.W[i] * P[i].log() for i in range(NX)]
        T = [-self.W[i] / P[i] for i in range(NX)]
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
        J = arb_mat(nu, nu)
        for j in range(na):
            for k in range(nfree):
                J[j, k] = (mass[k] * (M1[j, k] - pos[k] * M0[j, k])).mid()
            for k in range(na):
                J[j, nfree + k] = M0[j, k].mid()
            if j < nfree:
                J[j, j] = (J[j, j] + g1[j]).mid()
        for jj in range(nfree):
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
            st = J.solve(arb_mat([[(-e).mid()] for e in E]))
            ok = False
            lam = arb(1)
            for _ in range(25):
                pn = [arb((pos[k] + lam * st[k, 0]).mid()) for k in range(nfree)] + pos[nfree:]
                mn = [arb((mass[k] + lam * st[nfree + k, 0]).mid()) for k in range(na)]
                if min(float(x.mid()) for x in mn) <= 0:
                    lam /= 2
                    continue
                rn = max(abs(e.mid()) for e in self.evaluate(pn, mn, nfree, jac=False))
                if rn < r:
                    pos, mass, ok = pn, mn, True
                    break
                lam /= 2
            if not ok:
                return pos, mass, float(r), it
        E = self.evaluate(pos, mass, nfree, jac=False)
        return pos, mass, float(max(abs(e.mid()) for e in E)), itmax


def order(st, depths, masses):
    """reorder (wall, free..., semi...) -> (free..., wall, semi...)"""
    nf = st.nfree
    return (
        depths[1 : 1 + nf] + [depths[0]] + depths[1 + nf :],
        masses[1 : 1 + nf] + [masses[0]] + masses[1 + nf :],
    )


def unorder(nf, pos, mass):
    return [pos[nf]] + pos[:nf] + pos[nf + 1 :], [mass[nf]] + mass[:nf] + mass[nf + 1 :]


out = {"PREC": PREC, "B1": B1, "L": L, "stages": []}
law = (0.0558, 0.5482)
solved_d = None
solved_m = None
Ds = list(np.arange(20.0, DMAX + 1e-9, STEP))
if Ds[-1] < DMAX - 1e-9:
    Ds.append(DMAX)
for D in Ds:
    st = Stage(D, law)
    na = st.nact
    nf = st.nfree
    d0l = [float(x) for x in st.dj[:na]]
    m0l = [3.5] + [1 / rho_of(d, law[0]) for d in st.dj[1:na]]
    if solved_d is not None:
        k0 = min(len(solved_d), 1 + nf)
        d0l[:k0] = solved_d[:k0]
        k1 = min(len(solved_m), na)
        m0l[:k1] = solved_m[:k1]
    pos, mass = order(st, d0l, m0l)
    pos = [arb(x) for x in pos]
    mass = [arb(x) for x in mass]
    tol = math.exp(-4 * math.pi**2 * rho_of(D + B1, law[0]) ** 2) * 1e-9
    pos, mass, r, it = st.solve(pos, mass, nf, tol)
    ps, ms = unorder(nf, [float(t.mid()) for t in pos], [float(t.mid()) for t in mass])
    conv = r < tol * 1e3
    say(
        f"D = {D:.1f}: wall + {nf} free + {na - nf - 1} semi-free, {it} steps, max|E| {r:.2e} (tol {tol:.1e}, conv {conv}); law ({law[0]:+.5f}, {law[1]:+.5f}); "
        f"depths 1-5 {[round(x, 9) for x in ps[:5]]}; wall mass {ms[0]:.6f}   [{time.time() - t0:.0f}s]"
    )
    out["stages"].append(
        {
            "D": D,
            "nfree": nf,
            "na": na,
            "d": ps,
            "m": ms,
            "res": r,
            "tol": tol,
            "conv": bool(conv),
            "law": list(law),
        }
    )
    json.dump(out, open(f"c50_{TAG}.json", "w"), indent=1)
    if not conv:
        say("STOP: unconverged")
        break
    solved_d, solved_m = ps[: 1 + nf], ms[:na]
    if REFIT and D - 12 > 8:
        dd = np.array(ps[: 1 + nf])
        jj = np.arange(1, len(dd) + 1) - 0.5
        mm = (dd >= 4) & (dd <= D - 12)
        law = tuple(
            least_squares(lambda q: np.array([N_of(x, q[0], q[1]) for x in dd[mm]]) - jj[mm], list(law)).x
        )
say(f"done [{time.time() - t0:.0f}s]")
