"""q920 (7 Oct 2026): ball-arithmetic (python-flint, 200 bits) implementation of the residual and the analytic Jacobian of q206_gauss_mp.py for the amplitude-constrained Gaussian channel, with the analytic dF/dA of q901;
the sums over the quadrature grid are done by one arb_mat product, which makes a Jacobian about twenty times faster than the mpmath loops of q206 (instrument for T61, recorded in an amendment with its test against q206).
Same discrete problem as q206: grid y_i = -A - 9 + H i for i = 0..n-1 with n = int((2A + 18)/H) + 1 (H = the double 0.02 as in q206), interior atoms +-u_j with per-atom mass w_j, wall atoms +-A with mass we, optional centre atom with mass wc;
unknown vector v = (u_1..u_m, w_1..w_m, we[, wc]); residual rows D(u_j) - D(A) (m), D'(u_j) (m), 2 (sum w + we) + wc - 1, and for odd K D(0) - D(A).
Usage as a library: from q920_cap_flint import Cap."""

import os, sys, time
from flint import arb, arb_mat, ctx
from mpmath import mp, mpf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q206_gauss_mp as G

ctx.prec = 200
mp.dps = 40
HD = arb(float(G.H))
L2P = (arb.pi() * 2).log() / 2


class Cap:
    def __init__(self, A_str, m, odd, grid_A=None):
        self.A = arb(A_str)
        self.m = m
        self.odd = odd
        self.n = 2 * m + 1
        self.nn = self.n + (1 if odd else 0)
        self.grid_A = grid_A
        gA = (
            grid_A if grid_A is not None else A_str
        )  # grid_A: the grid of q206 for this amplitude is used instead of the grid of A (a frozen grid for a trace in A)
        npts = int((2 * mpf(gA) + 18) / G.H) + 1
        self.N = npts
        self.y = [-arb(gA) - 9 + HD * i for i in range(npts)]

    def atom(self, x):
        S = [yy - x for yy in self.y]
        L = [-(s * s) / 2 - L2P for s in S]
        P = [l.exp() for l in L]
        return S, L, P

    def tables(self, u, w, we, wc):
        N = self.N
        m = self.m
        rows = []
        Q = [arb(0)] * N
        for j in range(m):
            Sp, Lp, Pp = self.atom(u[j])
            Sm, Lm, Pm = self.atom(-u[j])
            rows.append((Sp, Lp, Pp, Sm, Lm, Pm))
            wj = w[j]
            Q = [Q[i] + wj * (Pp[i] + Pm[i]) for i in range(N)]
        SA, LA, PA = self.atom(self.A)
        SmA, LmA, PmA = self.atom(-self.A)
        Q = [Q[i] + we * (PA[i] + PmA[i]) for i in range(N)]
        cen = None
        if wc is not None:
            Sc, Lc, Pc = self.atom(arb(0))
            cen = (Sc, Lc, Pc)
            Q = [Q[i] + wc * Pc[i] for i in range(N)]
        lQ = [q.log() for q in Q]
        return rows, (SA, LA, PA), (SmA, LmA, PmA), cen, Q, lQ

    def D(self, L, P, lQ):
        return HD * sum((P[i] * (L[i] - lQ[i]) for i in range(self.N)), arb(0))

    def Dp(self, S, L, P, lQ):
        return HD * sum((P[i] * S[i] * (L[i] - lQ[i] + 1) for i in range(self.N)), arb(0))

    def Dpp(self, S, L, P, lQ):
        return HD * sum(
            (P[i] * (S[i] * S[i] * (L[i] - lQ[i] + 2) - (L[i] - lQ[i] + 1)) for i in range(self.N)), arb(0)
        )

    def res_jac(self, v, want_J=True, want_b=False):
        m = self.m
        odd = self.odd
        N = self.N
        n = self.n
        u = v[:m]
        w = v[m : 2 * m]
        we = v[2 * m]
        wc = v[2 * m + 1] if odd else None
        rows, (SA, LA, PA), (SmA, LmA, PmA), cen, Q, lQ = self.tables(u, w, we, wc)
        D_A = self.D(LA, PA, lQ)
        D_u = [self.D(r[1], r[2], lQ) for r in rows]
        Dp_u = [self.Dp(r[0], r[1], r[2], lQ) for r in rows]
        res = [D_u[j] - D_A for j in range(m)] + Dp_u + [2 * (sum(w, arb(0)) + we) + (wc if odd else 0) - 1]
        if odd:
            res.append(self.D(cen[1], cen[2], lQ) - D_A)
        if not want_J and not want_b:
            return res, None, None
        invQ = [1 / q for q in Q]
        # source columns (scaled by 1/Q): u_k, w_k, we, [wc], [A]
        cols = []
        for k in range(m):
            Sp, Lp, Pp, Sm, Lm, Pm = rows[k]
            cols.append([w[k] * (Pp[i] * Sp[i] - Pm[i] * Sm[i]) * invQ[i] for i in range(N)])
        for k in range(m):
            Pp = rows[k][2]
            Pm = rows[k][5]
            cols.append([(Pp[i] + Pm[i]) * invQ[i] for i in range(N)])
        cols.append([(PA[i] + PmA[i]) * invQ[i] for i in range(N)])
        if odd:
            cols.append([cen[2][i] * invQ[i] for i in range(N)])
        if want_b:
            cols.append([we * (PA[i] * SA[i] - PmA[i] * SmA[i]) * invQ[i] for i in range(N)])
        ncol = len(cols)
        # row functions: P_j (m), P_A, [P_0], P_j S_j (m)
        R = (
            [rows[j][2] for j in range(m)]
            + [PA]
            + ([cen[2]] if odd else [])
            + [[rows[j][2][i] * rows[j][0][i] for i in range(N)] for j in range(m)]
        )
        nrow = len(R)
        Rm = arb_mat(nrow, N, [x for r in R for x in r])
        Ct = arb_mat(N, ncol, [cols[c][i] for i in range(N) for c in range(ncol)])
        Gm = Rm * Ct
        iPA = m
        iP0 = m + 1 if odd else None
        iPS = m + (2 if odd else 1)
        nc_J = n + (1 if odd else 0)
        J = arb_mat(nc_J, nc_J) if want_J else None
        if want_J:
            for j in range(m):
                for c in range(m + m + 1 + (1 if odd else 0)):
                    J[j, c] = -HD * (Gm[j, c] - Gm[iPA, c]) + (Dp_u[j] if c == j else 0)
                    J[m + j, c] = -HD * Gm[iPS + j, c] + (
                        self.Dpp(rows[j][0], rows[j][1], rows[j][2], lQ) if c == j else 0
                    )
            for k in range(m):
                J[2 * m, m + k] = arb(2)
            J[2 * m, 2 * m] = arb(2)
            if odd:
                J[2 * m, 2 * m + 1] = arb(1)
                for c in range(m + m + 2):
                    J[2 * m + 1, c] = -HD * (Gm[iP0, c] - Gm[iPA, c])
        b = None
        if want_b:
            cb = ncol - 1
            DpA = self.Dp(SA, LA, PA, lQ)
            b = (
                [-HD * (Gm[j, cb] - Gm[iPA, cb]) - DpA for j in range(m)]
                + [-HD * Gm[iPS + j, cb] for j in range(m)]
                + [arb(0)]
            )
            if odd:
                b.append(-HD * (Gm[iP0, cb] - Gm[iPA, cb]) - DpA)
        return res, J, b

    def signals(self, v, grid_pts=400):
        """centre signal (even K: D(0) - C; odd K: second difference of D at 0 with d = 0.01) and the largest off-atom value of D(x) - C, as in q206.signals"""
        m = self.m
        u = v[:m]
        w = v[m : 2 * m]
        we = v[2 * m]
        wc = v[2 * m + 1] if self.odd else None
        rows, (SA, LA, PA), _, cen, Q, lQ = self.tables(u, w, we, wc)
        C = self.D(LA, PA, lQ)
        Dx = lambda x: (lambda SLP: self.D(SLP[1], SLP[2], lQ))(self.atom(x))
        if wc is None:
            centre = Dx(arb(0)) - C
        else:
            d = arb("0.01")
            centre = (Dx(-d) - 2 * Dx(arb(0)) + Dx(d)) / (d * d)
        atoms = list(u) + [self.A] + ([arb(0)] if wc is not None else [])
        best = None
        for i in range(1, grid_pts):
            x = self.A * i / grid_pts
            if min(abs(x - a_) for a_ in atoms) < arb("0.05"):
                continue
            val = Dx(x) - C
            if best is None or val > best:
                best = val
        return centre, best

    def feasible(self, v):
        m = self.m
        u = v[:m]
        w = v[m : 2 * m]
        we = v[2 * m]
        wc = v[2 * m + 1] if self.odd else None
        return (
            all(u[i] < u[i + 1] for i in range(m - 1))
            and u[0] > 0
            and u[-1] < self.A
            and all(t > 0 for t in w)
            and we > 0
            and (wc is None or wc > 0)
        )


def vec(M, n):
    return [M[i, 0].mid() for i in range(n)]


def col(lst):
    return arb_mat(len(lst), 1, list(lst))


def norm2(r):
    return sum((t.mid() * t.mid() for t in r), arb(0)).sqrt()


def maxabs(r):
    return max(abs(t.mid()) for t in r)


def load_state(path, G_json=None):
    import json

    d = json.load(open(path))
    d = d[-1] if isinstance(d, list) else d
    A = str(d["A"]).replace("np.float64(", "").replace(")", "").strip()
    u = [arb(t) for t in d["u"]]
    w = [arb(t) for t in d["w"]]
    we = arb(d["we"])
    wc = None if d.get("wc") in (None, "None") else arb(d["wc"])
    return A, u, w, we, wc, d


def dump_state(path, A_str, cap, v, res, verdict, extra):
    import json

    m = cap.m
    odd = cap.odd
    out = {
        "A": A_str,
        "K": 2 * m + 2 + (1 if odd else 0),
        "residual": float(maxabs(res)),
        "verdict": verdict,
        "u": [t.str(40, radius=False) for t in v[:m]],
        "w": [t.str(40, radius=False) for t in v[m : 2 * m]],
        "we": v[2 * m].str(40, radius=False),
        "wc": (v[2 * m + 1].str(40, radius=False) if odd else None),
    }
    out.update(extra)
    json.dump(out, open(path, "w"), indent=1)
