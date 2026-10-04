"""I_centre_ring (03 Oct 2026, test of 1414): the filled-ellipse optimum as a D2 boundary ring plus one atom at the centre (float64,
G_core quadrature). Unknowns v = [G angles, orbit masses, m0, C]; equations: D - C = 0 at each orbit representative, tangential
derivative 0 at each G atom, D(0) - C = 0, total mass 1. Newton with complex-step Jacobian; then the checks: curve scan of D - C, the
curvature at every atom, and D - C on an interior grid of the filled ellipse (the centre is now an atom).
Usage: py I_centre_ring.py RP SEED_STATE.json M0_START "rm1,rm2,..."   (SEED: a G_core ring json, e.g. a line of c28f_states.jsonl)
"""

import sys, json
import numpy as np
from G_core import Curve, Ring, phi

LOG2PIE = np.log(2 * np.pi) + 1


class CurveC(Curve):
    def logPc(self, X, Y, G, m, m0, rm):
        ex, ey = self.factors(X, Y, G, rm)
        y = self.y
        return np.log((ex.T * m) @ ey + m0 * np.outer(phi(y), phi(y)))

    def residc(self, ring, v, rm):
        nG = len(ring.G)
        nO = ring.nO
        G = v[:nG]
        m = v[nG : nG + nO]
        m0 = v[nG + nO]
        C = v[nG + nO + 1]
        L = self.logPc(ring.X, ring.Y, G, m, m0, rm)
        reps = np.concatenate(
            [np.array(([0.0] if ring.X else []) + ([np.pi / 2] if ring.Y else []), dtype=v.dtype), G]
        )
        Q = np.stack([self.rp * np.cos(reps), rm * np.sin(reps)], 1)
        D, g = self.D(L, Q, 1)
        tv = np.stack([-self.rp * np.sin(G), rm * np.cos(G)], 1)
        tg = (g[nO - nG :] * tv).sum(1)
        D0 = self.D(L, np.zeros((1, 2), dtype=v.dtype))
        return np.concatenate([D - C, tg, D0 - C, [(ring.mult() * m).sum() + m0 - 1]])

    def solvec(self, ring, v, rm, tol=1e-13, maxit=60):
        r = self.residc(ring, v, rm)
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
                J[:, j] = self.residc(ring, vc, rm).imag / h
            dv = np.linalg.lstsq(J, -r, rcond=None)[0]
            lam = 1.0
            moved = False
            for _ in range(30):
                vn = v + lam * dv
                rn = self.residc(ring, vn, rm)
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
rms = [float(x) for x in sys.argv[4].split(",")]
cv = CurveC(rp)
ring = Ring.from_json(d)
nG = len(ring.G)
nO = ring.nO
v = np.concatenate([np.array(ring.G), np.array(ring.m) * (1 - m00), [m00], [d["C"]]])
for rm in rms:
    v, nr = cv.solvec(ring, v, rm)
    G = v[:nG]
    m = v[nG : nG + nO]
    m0 = v[nG + nO]
    C = v[nG + nO + 1]
    L = cv.logPc(ring.X, ring.Y, G, m, m0, rm)
    tt = np.linspace(0, np.pi / 2, 3601)
    Dc = cv.D(L, cv.pos(tt, rm)) - C
    reps = np.concatenate([[0.0] if ring.X else [], [np.pi / 2] if ring.Y else [], G])
    away = np.ones_like(tt, dtype=bool)
    for a in reps:
        away &= np.abs(tt - a) > 0.004
    Dd, d1, d2 = cv.Dt(L, reps, rm, 2)
    s = np.linspace(0, 0.995, 40)
    th = np.linspace(0, np.pi / 2, 31)
    S, TH = np.meshgrid(s, th)
    Qi = np.stack([(S * rp * np.cos(TH)).ravel(), (S * rm * np.sin(TH)).ravel()], 1)
    Di = cv.D(L, Qi) - C
    near0 = np.hypot(Qi[:, 0], Qi[:, 1]) > 0.05
    _, _, h0 = cv.D(L, np.zeros((1, 2)), 2)
    print(
        f"rm {rm}: residual {nr:.1e}; m0 {m0:.10f}; C {C:.12f}; K {ring.K} + centre; curve max D-C off atoms {Dc[away].max():+.2e}; "
        f"atom d2 max {d2.max():+.2e}; interior max D-C (|q|>0.05) {Di[near0].max():+.2e}; centre Hessian diag {h0[0,0]:+.3e} {h0[0,2]:+.3e}",
        flush=True,
    )
    st = {
        "rm": rm,
        "X": int(ring.X),
        "Y": int(ring.Y),
        "G": G.tolist(),
        "m": m.tolist(),
        "m0": float(m0),
        "C": float(C),
        "residual": float(nr),
    }
    open(f"G_work/{__import__('os').environ.get('OUTP', 'c28ctr')}_rm{rm}.json", "w").write(json.dumps(st))
