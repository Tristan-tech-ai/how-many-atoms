"""G_core (02 Oct 2026): float64 capacity KKT on the ellipse CURVE x = (rp cos t, rm sin t), unit noise, D2-symmetric rings.
Library only: no computation and no file writes at import.

Quadrature. D(q) = -E log p(q + Z) - log(2 pi e), Z ~ N(0, I2), p the output density. E log p(q + Z) = int phi2(y - q) log p(y) dy is
taken on a FIXED grid: composite Gauss-Legendre panels (width H, n points) on [0, B] per axis, B = rp + 10. The output density of a
D2-symmetric ring is even in y1 and y2, so the plane integral folds to the quadrant with the factor phi(y - q) + phi(y + q) per axis.
p has complex zeros at |Im y| ~ pi/(2 rp) (two atoms at +-rp), so the panel half-width must stay well below that: default H 0.25,
n 16 (Bernstein parameter ~8 at rp 3, ~5 at rp 5; GL error far below float64). Checked by changing H and n (G_qcheck.py).

Orbits (per-atom masses): X = {0, pi} (2 atoms at (+-rp, 0)), Y = {pi/2, 3 pi/2} (2 atoms at (0, +-rm)), G(a) = {+-a, pi +- a}
(4 atoms). p(y) = sum_o m_o ex_o(y1) ey_o(y2), so P on the quadrant grid is EX^T diag(m) EY.
Unknowns v = [G angles] + [orbit masses (X, Y, G...)] + [C]; equations: D - C at one atom per orbit, dD/dt at each G atom, total mass
2 mX + 2 mY + 4 sum mG = 1 (the record's q719 system). Jacobian by complex step (exact to rounding)."""

import numpy as np

LOG2PIE = np.log(2 * np.pi * np.e)
SQ2PI = np.sqrt(2 * np.pi)


def phi(z):
    return np.exp(-z * z / 2) / SQ2PI


class Ring:
    """X, Y: bool; G: array of angles in (0, pi/2); m: per-atom masses [mX if X, mY if Y, mG...]; C."""

    def __init__(self, X, Y, G, m, C=0.0):
        self.X = bool(X)
        self.Y = bool(Y)
        self.G = np.array(G, dtype=float)
        self.m = np.array(m, dtype=float)
        self.C = float(C)

    @property
    def nO(self):
        return int(self.X) + int(self.Y) + len(self.G)

    @property
    def K(self):
        return 2 * int(self.X) + 2 * int(self.Y) + 4 * len(self.G)

    def mult(self):
        return np.array([2] * (int(self.X) + int(self.Y)) + [4] * len(self.G), dtype=float)

    def v(self):
        return np.concatenate([self.G, self.m, [self.C]])

    def with_v(self, v):
        nG = len(self.G)
        return Ring(self.X, self.Y, v[:nG], v[nG : nG + self.nO], v[-1])

    def reps(self):
        return np.array(([0.0] if self.X else []) + ([np.pi / 2] if self.Y else []) + list(self.G))

    def angles(self):
        """all atom angles in [0, 2 pi) with per-atom masses"""
        th, w = [], []
        k = 0
        if self.X:
            th += [0.0, np.pi]
            w += [self.m[k]] * 2
            k += 1
        if self.Y:
            th += [np.pi / 2, 3 * np.pi / 2]
            w += [self.m[k]] * 2
            k += 1
        for i, a in enumerate(self.G):
            th += [a, -a % (2 * np.pi), np.pi - a, np.pi + a]
            w += [self.m[k + i]] * 4
        o = np.argsort(th)
        return np.array(th)[o], np.array(w)[o]

    def label(self):
        return ("X" if self.X else "") + ("Y" if self.Y else "") + f"{len(self.G)}G"

    def to_json(self, rm, extra=None):
        d = {
            "rm": float(rm),
            "X": int(self.X),
            "Y": int(self.Y),
            "G": [float(a) for a in self.G],
            "m": [float(x) for x in self.m],
            "C": self.C,
            "K": self.K,
        }
        if extra:
            d.update(extra)
        return d

    @staticmethod
    def from_json(d):
        return Ring(d["X"], d["Y"], d["G"], d["m"], d.get("C", 0.0))


class Curve:
    def __init__(self, rp, B=None, H=0.25, n=16):
        self.rp = float(rp)
        B = B or self.rp + 10.0
        self.cost = 0.0  # cost: add cost*sin^2 t to D (G_family.md 21:36)
        x, w = np.polynomial.legendre.leggauss(n)
        a = np.arange(0.0, B - 1e-12, H)
        self.y = (a[:, None] + H / 2 * (1 + x[None, :])).ravel()
        self.v = np.tile(H / 2 * w, len(a))

    # ---- output density on the quadrant grid
    def factors(self, X, Y, G, rm):
        y = self.y
        rp = self.rp
        ex = []
        ey = []
        if X:
            ex.append(phi(y - rp) + phi(y + rp))
            ey.append(phi(y + 0 * rm))
        if Y:
            ex.append(phi(y + 0 * rm))
            ey.append(phi(y - rm) + phi(y + rm))
        for a in G:
            c = rp * np.cos(a)
            s = rm * np.sin(a)
            ex.append(phi(y - c) + phi(y + c))
            ey.append(phi(y - s) + phi(y + s))
        return np.array(ex), np.array(ey)

    def logP(self, X, Y, G, m, rm):
        ex, ey = self.factors(X, Y, G, rm)
        return np.log((ex.T * m) @ ey)

    # ---- KKT function at query points Q (nq, 2)
    def qf(self, q):
        y = self.y
        v = self.v
        a = y[None, :] - q[:, None]
        b = y[None, :] + q[:, None]
        pa = phi(a)
        pb = phi(b)
        return v * (pa + pb), v * (a * pa - b * pb), v * ((a * a - 1) * pa + (b * b - 1) * pb)

    def D(self, L, Q, order=0):
        Q = np.asarray(Q)
        F1, G1, H1 = self.qf(Q[:, 0])
        F2, G2, H2 = self.qf(Q[:, 1])
        F1L = F1 @ L
        D = -(F1L * F2).sum(1) - LOG2PIE
        if order == 0:
            return D
        G1L = G1 @ L
        g = np.stack([-(G1L * F2).sum(1), -(F1L * G2).sum(1)], 1)
        if order == 1:
            return D, g
        H1L = H1 @ L
        h = np.stack([-(H1L * F2).sum(1), -(G1L * G2).sum(1), -(F1L * H2).sum(1)], 1)  # d11, d12, d22
        return D, g, h

    def pos(self, t, rm):
        return np.stack([self.rp * np.cos(t), rm * np.sin(t)], -1)

    def tang(self, t, rm):
        return np.stack([-self.rp * np.sin(t), rm * np.cos(t)], -1)

    def Dt(self, L, t, rm, order=0):
        """D along the curve at angles t, with d/dt and d2/dt2 if order >= 1, 2"""
        t = np.atleast_1d(t)
        Q = self.pos(t, rm)
        c = self.cost
        if order == 0:
            return self.D(L, Q) + c * np.sin(t) ** 2
        D, g, h = self.D(L, Q, 2)
        tv = self.tang(t, rm)
        acc = -Q
        d1 = (g * tv).sum(1)
        d2 = (
            h[:, 0] * tv[:, 0] ** 2
            + 2 * h[:, 1] * tv[:, 0] * tv[:, 1]
            + h[:, 2] * tv[:, 1] ** 2
            + (g * acc).sum(1)
        )
        return D + c * np.sin(t) ** 2, d1 + c * np.sin(2 * t), d2 + 2 * c * np.cos(2 * t)

    # ---- KKT system
    def resid(self, ring, v, rm):
        nG = len(ring.G)
        nO = ring.nO
        G = v[:nG]
        m = v[nG : nG + nO]
        C = v[nG + nO]
        L = self.logP(ring.X, ring.Y, G, m, rm)
        reps = np.concatenate(
            [np.array(([0.0] if ring.X else []) + ([np.pi / 2] if ring.Y else []), dtype=v.dtype), G]
        )
        Q = np.stack([self.rp * np.cos(reps), rm * np.sin(reps)], 1)
        if nG:
            D, g = self.D(L, Q, 1)
            tv = np.stack([-self.rp * np.sin(G), rm * np.cos(G)], 1)
            tg = (g[nO - nG :] * tv).sum(1)
        else:
            D = self.D(L, Q)
            tg = np.zeros(0, dtype=v.dtype)
        mult = ring.mult()
        if self.cost:
            D = D + self.cost * np.sin(reps) ** 2
            tg = tg + self.cost * np.sin(2 * G)
        return np.concatenate([D - C, tg, [(mult * m).sum() - 1]])

    def jac(self, ring, v, rm):
        n = len(v)
        J = np.zeros((n, n))
        h = 1e-30
        for j in range(n - 1):
            vc = v.astype(complex)
            vc[j] += 1j * h
            J[:, j] = self.resid(ring, vc, rm).imag / h
        J[: ring.nO, n - 1] = -1.0
        return J

    def jac_rm(self, ring, v, rm):
        h = 1e-30
        return self.resid(ring, v.astype(complex), rm + 1j * h).imag / h

    def c_from_atoms(self, ring, rm):
        L = self.logP(ring.X, ring.Y, ring.G, ring.m, rm)
        D = self.Dt(L, ring.reps(), rm)
        return float((ring.mult() * ring.m * D).sum() / (ring.mult() * ring.m).sum())

    def newton(self, ring, rm, tol=1e-14, maxit=40, verbose=False):
        v = ring.v().copy()
        r = self.resid(ring, v, rm)
        nr = np.abs(r).max()
        for it in range(maxit):
            if nr < tol:
                break
            J = self.jac(ring, v, rm)
            dv = np.linalg.lstsq(J, -r, rcond=None)[0]
            lam = 1.0
            moved = False
            for _ in range(30):
                vn = v + lam * dv
                nG = len(ring.G)
                nO = ring.nO
                if np.all(vn[nG : nG + nO] > -1e-3) and np.all(
                    (vn[:nG] > -0.2) & (vn[:nG] < np.pi / 2 + 0.2)
                ):
                    rn = self.resid(ring, vn, rm)
                    nn = np.abs(rn).max()
                    if np.isfinite(nn) and (rn**2).sum() < (r**2).sum():
                        v, r, nr = vn, rn, nn
                        moved = True
                        break
                lam /= 2
            if verbose:
                print(f"   newton it {it}: residual {nr:.3e} lam {lam:.2e}", flush=True)
            if not moved:
                break
        return ring.with_v(v), nr

    # ---- KKT checks
    def scan(self, ring, rm, nscan=1800):
        """D - C on nscan + 1 points of [0, pi/2]. birth = the largest local maximum of D - C that is not at an atom (grid points
        within 2 spacings of an atom angle are skipped); margins midway between neighbouring atoms (and at a vertex without an atom);
        d2 = d2(D - C)/dt2 at each orbit representative (X, Y, G order)."""
        L = self.logP(ring.X, ring.Y, ring.G, ring.m, rm)
        ts = np.linspace(0, np.pi / 2, nscan + 1)
        h = ts[1]
        Dv = self.Dt(L, ts, rm) - ring.C
        th, _ = ring.angles()
        full = np.concatenate([th, [th[0] + 2 * np.pi]])
        dist = np.min(np.abs(((ts[:, None] - th[None, :]) + np.pi) % (2 * np.pi) - np.pi), 1)
        lm = np.ones(len(ts), bool)
        lm[1:] &= Dv[1:] >= Dv[:-1]
        lm[:-1] &= Dv[:-1] >= Dv[1:]
        cand = lm & (dist > 2.01 * h)
        if cand.any():
            k = np.argmax(np.where(cand, Dv, -np.inf))
            birth = (float(ts[k]), float(Dv[k]))
        else:
            birth = (float("nan"), -1.0)
        mids = [(a + b) / 2 for a, b in zip(full[:-1], full[1:]) if (a + b) / 2 < np.pi / 2 - 1e-9]
        mids = ([] if ring.X else [0.0]) + [x for x in mids if x > 1e-9] + ([] if ring.Y else [np.pi / 2])
        marg = list(zip(mids, (self.Dt(L, np.array(mids), rm) - ring.C).tolist()))
        Dr, d1, d2 = self.Dt(L, ring.reps(), rm, 2)
        return {
            "birth": birth,
            "margins": marg,
            "d2": d2.tolist(),
            "resD": (Dr - ring.C).tolist(),
            "d1": d1.tolist(),
            "scanmax": float(Dv.max()),
            "ts": ts,
            "Dv": Dv,
        }

    # ---- Blahut-Arimoto on the curve (orbit nodes in [0, pi/2])
    def ba(self, rm, N=400, iters=3000, H=0.5, n=10, tol=1e-12, verbose=False):
        cv = Curve(self.rp, H=H, n=n)
        t = np.linspace(0, np.pi / 2, N + 1)
        mult = np.where((t == 0) | (t == np.pi / 2), 2.0, 4.0)
        y = cv.y
        c = self.rp * np.cos(t)
        s = rm * np.sin(t)
        ex = phi(y[None, :] - c[:, None]) + phi(y[None, :] + c[:, None])
        ey = phi(y[None, :] - s[:, None]) + phi(y[None, :] + s[:, None])
        ex[0] = phi(y - self.rp) + phi(y + self.rp)
        ey[0] = phi(y)  # X node: 2 atoms
        ex[-1] = phi(y)
        ey[-1] = phi(y - rm) + phi(y + rm)  # Y node: 2 atoms
        F1, _, _ = cv.qf(c)
        F2, _, _ = cv.qf(s)
        Qm = np.full(N + 1, 1.0 / (N + 1))  # orbit total masses
        for it in range(iters):
            P = (ex.T * (Qm / mult)) @ ey
            L = np.log(P)
            D = -((F1 @ L) * F2).sum(1) - LOG2PIE
            I = (Qm * D).sum()
            gap = D.max() - I
            if gap < tol:
                break
            Qm = Qm * np.exp(D - D.max())
            Qm /= Qm.sum()
        if verbose:
            print(f"   BA rm {rm}: {it} iterations, I {I:.12f}, gap {gap:.2e}", flush=True)
        return t, Qm, I, gap

    def ring_from_ba(self, rm, t, Qm, thr=1e-6):
        """mass peaks -> orbits (X if a peak at t = 0, Y if at pi/2, G otherwise), masses per atom"""
        N = len(Qm)
        pk = [
            i
            for i in range(N)
            if Qm[i] > thr * Qm.max() and (i == 0 or Qm[i] >= Qm[i - 1]) and (i == N - 1 or Qm[i] > Qm[i + 1])
        ]
        lab = np.argmin(np.abs(t[:, None] - t[pk][None, :]), 1)
        X = False
        Y = False
        G = []
        mX = mY = 0.0
        mG = []
        for k, i in enumerate(pk):
            sel = lab == k
            M = Qm[sel].sum()
            tc = (t[sel] * Qm[sel]).sum() / M
            if i == 0 or tc < 0.02:
                X = True
                mX += M / 2
            elif i == N - 1 or tc > np.pi / 2 - 0.02:
                Y = True
                mY += M / 2
            else:
                G.append(tc)
                mG.append(M / 4)
        m = ([mX] if X else []) + ([mY] if Y else []) + mG
        r = Ring(X, Y, G, m)
        r.C = self.c_from_atoms(r, rm)
        return r
