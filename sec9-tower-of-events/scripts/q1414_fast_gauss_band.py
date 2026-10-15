"""q1414 (9 Oct 2026, INSTRUMENT, no item, no verdict): band-limited evaluation for q1410. The kernel phi(y|x) is below 1e-80 of its peak for |y - x| > 19.5, so every
atom contributes to the sums only on a band of about 780 of the 2500 to 3500 grid points; the entries outside the band are exact zeros of the matrices. The residual and
the Jacobian are the same functions as in q1410 up to terms smaller than 1e-80 relative; the Python-level work falls by a factor of about three.
Control (py q1414_fast_gauss_band.py control): residual and Jacobian at the K = 62 and K = 81 dumps of q1388 agree with q1410's dense evaluation to 1e-60 relative per entry
(max |dJ| over the row maximum) and with the 40-digit dump to 1e-39; timings of both evaluations are printed.
"""

import sys, time, json
from flint import arb, arb_mat, ctx
from q1410_fast_gauss import FastG


class FastGB(FastG):
    WB = 19.5

    def _band(self, y0f, Hf, N, x):
        xf = float(x.mid())
        return max(0, int((xf - self.WB - y0f) / Hf)), min(N, int((xf + self.WB - y0f) / Hf) + 2)

    def residual_and_jacobian(self, A, u, w, we, wc, want_J=True, y=None):
        H = self.H
        m = len(u)
        odd = wc is not None
        if y is None:
            y = self.grid(A)
        N = len(y)
        y0f = float(y[0].mid())
        Hf = self.Hf
        zero = arb(0)
        L2P = self.L2P

        def row(x):
            i0, i1 = self._band(y0f, Hf, N, x)
            ys = y[i0:i1]
            d = [yy - x for yy in ys]
            L = [-(dd * dd) / 2 - L2P for dd in d]
            return i0, i1, L, [l.exp() for l in L], d

        atoms = []  # (plus row, minus row) per interior atom
        Q = [zero] * N

        def addto(Q, r, wt):
            i0, i1, L, P, d = r
            Q[i0:i1] = [q + wt * p for q, p in zip(Q[i0:i1], P)]

        for uj, wj in zip(u, w):
            rp = row(uj)
            rm = row(-uj)
            atoms.append((rp, rm))
            addto(Q, rp, wj)
            addto(Q, rm, wj)
        rA = row(A)
        rmA = row(-A)
        addto(Q, rA, we)
        addto(Q, rmA, we)
        rc = None
        if odd:
            rc = row(arb(0))
            addto(Q, rc, wc)
        lQ = [q.log() for q in Q]

        def Dv(r):
            i0, i1, L, P, d = r
            return H * sum((p * (l - q) for p, l, q in zip(P, L, lQ[i0:i1])), zero)

        def Dpv(r):
            i0, i1, L, P, d = r
            return H * sum((p * s * (l - q + 1) for p, s, l, q in zip(P, d, L, lQ[i0:i1])), zero)

        def Dppv(r):
            i0, i1, L, P, d = r
            return H * sum(
                (p * (s * s * (l - q + 2) - (l - q + 1)) for p, s, l, q in zip(P, d, L, lQ[i0:i1])), zero
            )

        D_u = [Dv(a[0]) for a in atoms]
        D_A = Dv(rA)
        Dp_u = [Dpv(a[0]) for a in atoms]
        res = [D_u[j] - D_A for j in range(m)] + Dp_u + [2 * (sum(w, zero) + we) + (wc if odd else 0) - 1]
        if odd:
            res.append(Dv(rc) - D_A)
        if not want_J:
            return res, None
        Dpp_u = [Dppv(a[0]) for a in atoms]
        Qinv = [1 / q for q in Q]
        ncol = 2 * m + 1 + (1 if odd else 0)

        def pad(i0, i1, vals):
            return [zero] * i0 + vals + [zero] * (N - i1)

        evalrows = [a[0] for a in atoms] + [rA] + ([rc] if odd else [])
        RQ = arb_mat(
            len(evalrows),
            N,
            [
                v
                for r in evalrows
                for v in pad(r[0], r[1], [p * qi for p, qi in zip(r[3], Qinv[r[0] : r[1]])])
            ],
        )
        RSQ = arb_mat(
            m,
            N,
            [
                v
                for a in atoms
                for r in (a[0],)
                for v in pad(r[0], r[1], [p * s * qi for p, s, qi in zip(r[3], r[4], Qinv[r[0] : r[1]])])
            ],
        )
        cols = []
        for k in range(m):  # d Q / d u_k
            col = [zero] * N
            rp, rm = atoms[k]
            for i, p, s in zip(range(rp[0], rp[1]), rp[3], rp[4]):
                col[i] = w[k] * p * s
            for i, p, s in zip(range(rm[0], rm[1]), rm[3], rm[4]):
                col[i] = col[i] - w[k] * p * s
            cols.append(col)
        for k in range(m):  # d Q / d w_k
            col = [zero] * N
            rp, rm = atoms[k]
            for i, p in zip(range(rp[0], rp[1]), rp[3]):
                col[i] = p
            for i, p in zip(range(rm[0], rm[1]), rm[3]):
                col[i] = col[i] + p
            cols.append(col)
        col = [zero] * N  # d Q / d we
        for i, p in zip(range(rA[0], rA[1]), rA[3]):
            col[i] = p
        for i, p in zip(range(rmA[0], rmA[1]), rmA[3]):
            col[i] = col[i] + p
        cols.append(col)
        if odd:
            cols.append(pad(rc[0], rc[1], list(rc[3])))
        C = arb_mat(ncol, N, [v for c in cols for v in c]).transpose()
        G1 = RQ * C
        G2 = RSQ * C
        J = arb_mat(ncol, ncol)
        for j in range(m):
            for k in range(ncol):
                J[j, k] = -H * G1[j, k] + H * G1[m, k] + ((Dp_u[j] if (k == j) else 0))
                J[m + j, k] = -H * G2[j, k] + ((Dpp_u[j] if (k == j) else 0))
        for k in range(m):
            J[2 * m, m + k] = 2
        J[2 * m, 2 * m] = 2
        if odd:
            J[2 * m, 2 * m + 1] = 1
            for k in range(ncol):
                J[2 * m + 1, k] = -H * G1[m + 1, k] + H * G1[m, k]
        return res, J


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "control":
    ok = True
    for fn in ("q1388_jb_K62_end.json", "q1388_jb_K81_f0.97.json", "q1388_jb_K61_end.json"):
        d = json.load(open(fn))
        m = d["m"]
        odd = d["odd"]
        n = 2 * m + 1 + (1 if odd else 0)
        for Hs in (0.02, 0.05):
            Fd = FastG(192, H=Hs)
            Fb = FastGB(192, H=Hs)
            A = arb(d["A"])
            u = [arb(x) for x in d["u"]]
            w = [arb(x) for x in d["w"]]
            we = arb(d["we"])
            wc = None if d["wc"] is None else arb(d["wc"])
            y = Fd.grid(A)
            t0 = time.time()
            r1, J1 = Fd.residual_and_jacobian(A, u, w, we, wc, y=y)
            t1 = time.time() - t0
            t0 = time.time()
            r2, J2 = Fb.residual_and_jacobian(A, u, w, we, wc, y=y)
            t2 = time.time() - t0
            worst = 0.0
            rr = 0.0
            for a in range(n):
                rowmax = max(abs(float(J1[a, c].mid())) for c in range(n))
                for c in range(n):
                    worst = max(worst, abs(float((J1[a, c] - J2[a, c]).mid())) / rowmax)
                rr = max(rr, abs(float((r1[a] - r2[a]).mid())))
            worst40 = 0.0
            if Hs == 0.02:
                for a in range(n):
                    rowmax = max(abs(float(arb(d["J"][a][c]).mid())) for c in range(n))
                    for c in range(n):
                        worst40 = max(worst40, abs(float((J2[a, c] - arb(d["J"][a][c])).mid())) / rowmax)
            good = worst < 1e-50 and rr < 1e-50 and (Hs != 0.02 or worst40 < 1e-30)
            ok = ok and good
            print(
                f"{fn} H {Hs}: dense {t1:.2f}s, band {t2:.2f}s; max |dJ|/rowmax dense-band {worst:.1e}, max |dr| {rr:.1e}; vs 40-digit dump {worst40:.1e}: {good}"
            )
    print("CONTROL q1414:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 2)
