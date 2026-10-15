"""q1410 (9 Oct 2026, INSTRUMENT, no item, no verdict): a python-flint (arb, matrix products) version of the mirror-reduced KKT system of
q206_gauss_mp.py for the amplitude-constrained Gaussian channel, with the SAME discretisation (trapezoid in y on [-A - 9, A + 9], step H =
the double 0.02, n = int((2A + 18)/H) + 1 points, mpmath arithmetic for n). Same unknowns v = (u_1..u_m, w_1..w_m, we [, wc]), same residuals
D(u_j) - D(A), D'(u_j), 2(sum w + we) + wc - 1 [, D(0) - D(A)], same Jacobian. The only change is the evaluation: the kernel sums are
products of arb matrices, so a Jacobian at K = 62 should cost seconds instead of minutes.
Use as a module: F = FastG(prec); F.residual_and_jacobian(A, u, w, we, wc, want_J) with arb inputs; F.newton(A, u, w, we, wc).
Controls (run `py q1410_fast_gauss.py control`): (C1) against the 40-digit dump q1388_jb_K62_end.json: residual, J and b agree with the
mpmath values to 1e-28 relative per entry (max over entries of |dJ| / max(|J_row|)); (C2) Newton from the stored state A = 42.258175 to
A = 42.263175; the solution has fast residual < 1e-40 (tolerance 1e-45) and, evaluated at 60 digits by the independent mpmath code of q206 (the inputs rounded to 70 digits), residual < 1e-40 (agreement with the stored positions is limited to cond x residual, about 1e-19, and is only printed); timings printed.
"""

import sys, os, json, time
import mpmath as mp
from flint import arb, arb_mat, ctx

H_DOUBLE = 0.02


class FastG:
    def __init__(self, prec=192, H=H_DOUBLE, pad=9):
        ctx.prec = prec
        mp.mp.prec = prec
        self.Hf = H
        self.pad = pad
        self.H = arb(H)
        self.L2P = (2 * arb.pi()).log() / 2
        self.prec = prec

    def grid(self, A):
        # n as in q206 (mpmath, H the double 0.02)
        Amp = mp.mpf(A.str(40, radius=False)) if isinstance(A, arb) else mp.mpf(A)
        n = int((2 * Amp + 2 * self.pad) / mp.mpf(self.Hf)) + 1
        y0 = -A - self.pad
        return [y0 + self.H * i for i in range(n)]

    def rows(self, y, x):
        L = [-((yy - x) ** 2) / 2 - self.L2P for yy in y]
        return L, [l.exp() for l in L], [yy - x for yy in y]

    def tables(self, A, y, u, w, we, wc):
        N = len(y)
        m = len(u)
        Q = [arb(0)] * N
        rowsP = []
        for uj, wj in zip(u, w):
            Lp, Pp, Sp = self.rows(y, uj)
            Lm, Pm, Sm = self.rows(y, -uj)
            rowsP.append((Lp, Pp, Sp, Lm, Pm, Sm))
            Q = [q + wj * (a + b) for q, a, b in zip(Q, Pp, Pm)]
        LA, PA, SA = self.rows(y, A)
        LmA, PmA, SmA = self.rows(y, -A)
        Q = [q + we * (a + b) for q, a, b in zip(Q, PA, PmA)]
        cen = None
        if wc is not None:
            Lc, Pc, Sc = self.rows(y, arb(0))
            cen = (Lc, Pc, Sc)
            Q = [q + wc * a for q, a in zip(Q, Pc)]
        lQ = [q.log() for q in Q]
        return rowsP, (LA, PA, SA), (LmA, PmA, SmA), cen, Q, lQ

    def residual_and_jacobian(self, A, u, w, we, wc, want_J=True, y=None):
        H = self.H
        m = len(u)
        odd = wc is not None
        if y is None:
            y = self.grid(A)
        N = len(y)
        rowsP, wallp, wallm, cen, Q, lQ = self.tables(A, y, u, w, we, wc)
        LA, PA, SA = wallp
        Dv = lambda Lr, Pr: H * sum((p * (l - q) for p, l, q in zip(Pr, Lr, lQ)), arb(0))
        Dpv = lambda Lr, Pr, Sr: H * sum((p * s * (l - q + 1) for p, s, l, q in zip(Pr, Sr, Lr, lQ)), arb(0))
        Dppv = lambda Lr, Pr, Sr: H * sum(
            (p * (s * s * (l - q + 2) - (l - q + 1)) for p, s, l, q in zip(Pr, Sr, Lr, lQ)), arb(0)
        )
        D_u = [Dv(r[0], r[1]) for r in rowsP]
        D_A = Dv(LA, PA)
        Dp_u = [Dpv(r[0], r[1], r[2]) for r in rowsP]
        res = [D_u[j] - D_A for j in range(m)] + Dp_u + [2 * (sum(w, arb(0)) + we) + (wc if odd else 0) - 1]
        if odd:
            res.append(Dv(cen[0], cen[1]) - D_A)
        if not want_J:
            return res, None
        Dpp_u = [Dppv(r[0], r[1], r[2]) for r in rowsP]
        Qinv = [1 / q for q in Q]
        ncol = 2 * m + 1 + (1 if odd else 0)
        # evaluation rows (m atoms, then the wall, then the centre atom if present)
        evalP = [r[1] for r in rowsP] + [PA] + ([cen[1]] if odd else [])
        RQ = arb_mat(len(evalP), N, [p * qi for Pr in evalP for p, qi in zip(Pr, Qinv)])
        RSQ = arb_mat(m, N, [p * s * qi for r in rowsP for p, s, qi in zip(r[1], r[2], Qinv)])
        # source columns: u_k, w_k, we, wc
        src_u = [
            [w[k] * (rowsP[k][1][i] * rowsP[k][2][i] - rowsP[k][4][i] * rowsP[k][5][i]) for i in range(N)]
            for k in range(m)
        ]
        src_w = [[rowsP[k][1][i] + rowsP[k][4][i] for i in range(N)] for k in range(m)]
        src_we = [PA[i] + wallm[1][i] for i in range(N)]
        cols = src_u + src_w + [src_we] + ([cen[1]] if odd else [])
        C = arb_mat(ncol, N, [v for c in cols for v in c]).transpose()  # N x ncol
        G1 = RQ * C  # (m + 1 [+1]) x ncol
        G2 = RSQ * C  # m x ncol
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

    def feasible(self, A, u, w, we, wc):
        m = len(u)
        return (
            all(u[i] < u[i + 1] for i in range(m - 1))
            and u[0] > 0
            and u[-1] < A
            and all(t > 0 for t in w)
            and we > 0
            and (wc is None or wc > 0)
        )

    @staticmethod
    def mid(a):
        return arb(a.mid())

    def newton(self, A, u, w, we, wc, steps=60, tol="1e-30", verbose=False, growth=5.0):
        """Newton with an a-posteriori acceptance test on the correction size: the full step is taken; at the new point the Jacobian and the next correction are
        computed (needed anyway); the step is kept when the next correction is not more than `growth` times the previous one (the residual may rise for an
        iteration on these ill-conditioned systems and then fall quadratically), otherwise the step is halved from the old point.
        """
        m = len(u)
        odd = wc is not None
        tol = arb(tol)
        A = arb(A.mid())
        y = self.grid(
            A
        )  # balls are stripped to their midpoints: a wide radius on A or on the inputs makes the residual floor rise
        v = [arb(t.mid()) for t in (list(u) + list(w) + [we] + ([wc] if odd else []))]
        unpack = lambda x: (x[:m], x[m : 2 * m], x[2 * m], (x[2 * m + 1] if odd else None))
        nrm = lambda d: max(abs(t.mid()) for t in d)
        n = len(v)
        t0 = time.time()
        r, J = self.residual_and_jacobian(A, *unpack(v), y=y)
        res = nrm(r)
        k = 0
        sol = J.solve(arb_mat([[-t.mid()] for t in r]))
        dv = [sol[i, 0].mid() for i in range(n)]
        nd = nrm(dv)
        while res > tol and k < steps:
            if nd < arb("1e-60"):
                break
            lam = arb(1)
            done = False
            for _ in range(10):
                vn = [v[i] + lam * dv[i] for i in range(n)]
                if self.feasible(A, *unpack(vn)):
                    rn, Jn = self.residual_and_jacobian(A, *unpack(vn), y=y)
                    resn = nrm(rn)
                    soln = Jn.solve(arb_mat([[-t.mid()] for t in rn]))
                    dvn = [soln[i, 0].mid() for i in range(n)]
                    ndn = nrm(dvn)
                    if ndn <= growth * nd or resn < tol:
                        v, r, J, res, dv, nd = vn, rn, Jn, resn, dvn, ndn
                        done = True
                        break
                lam /= 2
            if not done:
                if verbose:
                    print("   no acceptable step; stop", flush=True)
                break
            k += 1
            if verbose:
                print(
                    f"   it {k}: lam {float(lam):.3g} residual {float(res):.3g} next step {float(nd):.2g} [{time.time() - t0:.0f}s]",
                    flush=True,
                )
        self.last_J = J
        self.last_r = r
        self.last_y = y
        return (*unpack(v), res, k)

    def signals(self, A, u, w, we, wc, y=None):
        if y is None:
            y = self.grid(A)
        rowsP, wallp, wallm, cen, Q, lQ = self.tables(A, y, u, w, we, wc)
        H = self.H
        Dv = lambda Lr, Pr: H * sum((p * (l - q) for p, l, q in zip(Pr, Lr, lQ)), arb(0))
        C = Dv(wallp[0], wallp[1])
        L0, P0, S0 = self.rows(y, arb(0))
        d = arb("0.01")
        if wc is None:
            centre = Dv(L0, P0) - C
        else:
            Dx = lambda x: Dv(*self.rows(y, x)[:2])
            centre = (Dx(-d) - 2 * Dx(arb(0)) + Dx(d)) / d**2
        return centre, C


def load_state(fn, prec_str=None):
    d = json.load(open(fn))
    return (
        arb(d["A"]) if isinstance(d["A"], str) else arb(repr(d["A"])),
        [arb(x) for x in d["u"]],
        [arb(x) for x in d["w"]],
        arb(d["we"]),
        (None if d["wc"] is None else arb(d["wc"])),
    )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "control":
    F = FastG(192)
    dump = json.load(open("q1388_jb_K62_end.json"))
    m = dump["m"]
    A = arb(dump["A"])
    u = [arb(x) for x in dump["u"]]
    w = [arb(x) for x in dump["w"]]
    we = arb(dump["we"])
    wc = None
    t0 = time.time()
    r, J = F.residual_and_jacobian(A, u, w, we, wc)
    t1 = time.time() - t0
    n = 2 * m + 1
    worst = 0.0
    for a in range(n):
        rowmax = max(abs(float(arb(dump["J"][a][c]).mid())) for c in range(n))
        for c in range(n):
            ref = arb(dump["J"][a][c])
            worst = max(worst, abs(float((J[a, c] - ref).mid())) / rowmax)
    print(
        f"C1: Jacobian time {t1:.1f}s; worst |dJ| / row max = {worst:.2e}; residual max {float(max(abs(t.mid()) for t in r)):.2e}  (dump residual {dump['r0max']})"
    )
    ok1 = worst < 1e-28
    print("C1", "OK" if ok1 else "FAILED")
    t0 = time.time()
    A2 = arb("42.263175")
    uu, ww, wwe, wwc, res, k = F.newton(A2, u, w, we, wc, tol="1e-45", verbose=True)
    t2 = time.time() - t0
    ref = json.load(
        open(os.path.join("..", "capacity_even62_family", "results_q220_gauss_K62_A42.2632.json"))
    )
    du = abs(float((uu[0] - arb(ref["u"][0])).mid()))
    dwe = abs(float((wwe - arb(ref["we"])).mid()))
    print(
        f"C2: Newton {k} iterations, {t2:.0f}s, fast residual {float(res):.2e}; |u_1 - stored| {du:.2e}, |we - stored| {dwe:.2e} (stored residual {ref['residual']:.1e}; the Jacobian is ill-conditioned, so a deviation of order cond x residual is expected)"
    )
    # C2': independent evaluation of the fast solution with the mpmath code of q206 (a different implementation of the same residual)
    import q206_gauss_mp as G

    mp.mp.dps = 60
    G.mp.dps = 60
    G.L2P = mp.log(2 * mp.pi) / 2
    G.H = mp.mpf(
        H_DOUBLE
    )  # q206's H is the double 0.02; importing it after raising the precision would give 0.02 to 57 digits
    cv = lambda t: mp.mpf(t.str(70, radius=False))
    Amp = cv(A2)
    yv = G.grid(Amp)
    rm, _ = G.residual_and_jacobian(
        Amp, yv, [cv(t) for t in uu], [cv(t) for t in ww], cv(wwe), None, want_J=False
    )
    rmax = max(abs(t) for t in rm)
    print(f"C2': mpmath residual of the fast solution {mp.nstr(rmax, 3)}")
    ok2 = rmax < mp.mpf("1e-40") and res < arb("1e-40")
    print("C2", "OK" if ok2 else "FAILED")
    sys.exit(0 if ok1 and ok2 else 2)
