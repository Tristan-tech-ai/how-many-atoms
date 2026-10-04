"""I_pair_ring (03 Oct 2026, test of 1419): the filled-ellipse optimum as a D2 boundary ring plus a PAIR of atoms at (+-a, 0) on the
major axis (float64, G_core quadrature). Text copy of I_centre_ring with the centre atom replaced by the pair (mass m0/2 each): output
density adds (m0/2)[phi(y1 - a) + phi(y1 + a)] phi(y2). Unknowns v = [G angles, orbit masses, m0, a, C]; equations: D - C = 0 at each
orbit representative, tangential derivative 0 at each G atom, D(a, 0) - C = 0, d/dx1 D(a, 0) = 0, total mass 1. Newton with
complex-step Jacobian; then the checks: curve scan of D - C, curvature at every atom, D - C on an interior grid of the filled ellipse
away from the pair, D(0) - C, and the Hessian of D at (a, 0).
Usage: py I_pair_ring.py RP SEED_STATE.json M0_START A_START "rm1,rm2,..."   (SEED: a G_core ring json; env OUTP names the output)
"""

import sys, json, os
import numpy as np
from G_core import Curve, Ring, phi


class CurveP(Curve):
    def logPp(self, X, Y, G, m, m0, a, rm):
        ex, ey = self.factors(X, Y, G, rm)
        y = self.y
        return np.log((ex.T * m) @ ey + (m0 / 2) * np.outer(phi(y - a) + phi(y + a), phi(y)))

    def residp(self, ring, v, rm):
        nG = len(ring.G)
        nO = ring.nO
        G = v[:nG]
        m = v[nG : nG + nO]
        m0 = v[nG + nO]
        a = v[nG + nO + 1]
        C = v[nG + nO + 2]
        L = self.logPp(ring.X, ring.Y, G, m, m0, a, rm)
        reps = np.concatenate(
            [np.array(([0.0] if ring.X else []) + ([np.pi / 2] if ring.Y else []), dtype=v.dtype), G]
        )
        Q = np.stack([self.rp * np.cos(reps), rm * np.sin(reps)], 1)
        D, g = self.D(L, Q, 1)
        tv = np.stack([-self.rp * np.sin(G), rm * np.cos(G)], 1)
        tg = (g[nO - nG :] * tv).sum(1)
        Qa = np.zeros((1, 2), dtype=v.dtype)
        Qa[0, 0] = a
        Da, ga = self.D(L, Qa, 1)
        return np.concatenate([D - C, tg, Da - C, [ga[0, 0]], [(ring.mult() * m).sum() + m0 - 1]])

    def solvep(self, ring, v, rm, tol=1e-13, maxit=60):
        r = self.residp(ring, v, rm)
        nr = np.abs(r).max()
        for it in range(maxit):
            if nr < tol:
                break
            n = len(v)
            J = np.zeros((n, n))
            h = 1e-30
            for j in range(n):
                vc = v.astype(complex)
                vc[j] += 1j * h
                J[:, j] = self.residp(ring, vc, rm).imag / h
            dv = np.linalg.lstsq(J, -r, rcond=None)[0]
            lam = 1.0
            moved = False
            for _ in range(30):
                vn = v + lam * dv
                rn = self.residp(ring, vn, rm)
                nn = np.abs(rn).max()
                if np.isfinite(nn) and (rn**2).sum() < (r**2).sum():
                    v, r, nr = vn, rn, nn
                    moved = True
                    break
                lam /= 2
            if not moved:
                break
        return v, nr


rp = float(sys.argv[1])
d = json.loads(open(sys.argv[2]).read().strip().splitlines()[0])
m00 = float(sys.argv[3])
a00 = float(sys.argv[4])
rms = [float(x) for x in sys.argv[5].split(",")]
cv = CurveP(rp)
ring = Ring.from_json(d)
nG = len(ring.G)
nO = ring.nO
v = np.concatenate([np.array(ring.G), np.array(ring.m) * (1 - m00), [m00], [a00], [d["C"]]])
for rm in rms:
    v, nr = cv.solvep(ring, v, rm)
    G = v[:nG]
    m = v[nG : nG + nO]
    m0 = v[nG + nO]
    a = v[nG + nO + 1]
    C = v[nG + nO + 2]
    L = cv.logPp(ring.X, ring.Y, G, m, m0, a, rm)
    tt = np.linspace(0, np.pi / 2, 3601)
    Dc = cv.D(L, cv.pos(tt, rm)) - C
    reps = np.concatenate([[0.0] if ring.X else [], [np.pi / 2] if ring.Y else [], G])
    away = np.ones_like(tt, dtype=bool)
    for b in reps:
        away &= np.abs(tt - b) > 0.004
    Dd, d1, d2 = cv.Dt(L, reps, rm, 2)
    s = np.linspace(0, 0.995, 40)
    th = np.linspace(0, np.pi / 2, 31)
    S, TH = np.meshgrid(s, th)
    Qi = np.stack([(S * rp * np.cos(TH)).ravel(), (S * rm * np.sin(TH)).ravel()], 1)
    Di = cv.D(L, Qi) - C
    offpair = np.hypot(Qi[:, 0] - a, Qi[:, 1]) > 0.05
    D0 = cv.D(L, np.zeros((1, 2)))[0] - C
    Qa = np.array([[a, 0.0]])
    _, _, ha = cv.D(L, Qa, 2)
    print(
        f"rm {rm}: residual {nr:.1e}; pair a {a:.6f}; m0 {m0:.10f}; C {C:.12f}; K {ring.K} + pair; curve max D-C off atoms {Dc[away].max():+.2e}; "
        f"atom d2 max {d2.max():+.2e}; interior max D-C (away from the pair) {Di[offpair].max():+.2e}; D(0)-C {D0:+.3e}; "
        f"Hessian at (a,0) {ha[0,0]:+.6f} {ha[0,2]:+.6f}",
        flush=True,
    )
    st = {
        "rm": rm,
        "X": int(ring.X),
        "Y": int(ring.Y),
        "G": G.tolist(),
        "m": m.tolist(),
        "m0": float(m0),
        "a": float(a),
        "C": float(C),
        "residual": float(nr),
    }
    open(f"G_work/{os.environ.get('OUTP', 'pair')}_rm{rm}.json", "w").write(json.dumps(st))
