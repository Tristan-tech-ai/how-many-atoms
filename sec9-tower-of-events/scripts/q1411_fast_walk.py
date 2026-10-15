"""q1411 (9 Oct 2026, INSTRUMENT / COMPUTATION, no item, no verdict): continuation in A of the capacity-achieving input of the Gaussian channel
with the arb solver q1410 (192 bits), tangent predictor from the implicit-function flow, adaptive step, and the family events: a BIRTH
(even K, the centre signal D(0) - C turns positive: a centre atom appears) and a SPLIT (odd K, D''(0) turns positive: the centre atom splits
into a pair). Output: one JSON per accepted state in the OUT directory (same fields as results_q220_*), a log line per state.
Usage: py q1411_fast_walk.py STATE.json OUTDIR A_END [HMAX]   (STATE.json: a stored state; the walk goes up in A through as many events as needed
until A_END). The first check (`py q1411_fast_walk.py control`) walks the K = 62 family of capacity_even62_family from A = 42.258175 through
its birth and compares the birth location (42.264024 by interpolation of the stored states, q1386c) and the first K = 63 states with the
stored tower/K63 states."""

import sys, os, json, time
import mpmath as mp
from flint import arb, arb_mat, ctx
from q1410_fast_gauss import FastG, H_DOUBLE
from q1414_fast_gauss_band import FastGB

say = lambda *a: print(*a, flush=True)
T0 = time.time()


def S(a, n=45):
    return a.str(n, radius=False)


class Walker:
    def __init__(self, prec=192, outdir="fastwalk", H=H_DOUBLE, pad=9):
        self.F = (FastGB if os.environ.get("Q1411_BAND", "1") == "1" else FastG)(prec, H=H, pad=pad)
        self.out = outdir
        os.makedirs(outdir, exist_ok=True)
        self.nsave = 0
        self.tol = arb("1e-30")

    # ---- state utilities
    def solve_at(self, A, u, w, we, wc, tol=None):
        F = self.F
        uu, ww, wwe, wwc, res, k = F.newton(A, u, w, we, wc, tol=(tol or "1e-30"))
        return uu, ww, wwe, wwc, res, k

    def signal(self, A, u, w, we, wc, y):
        """even K: D(0) - C (must be < 0); odd K: D''(0) (must be < 0)"""
        F = self.F
        H = F.H
        rowsP, wallp, wallm, cen, Q, lQ = F.tables(A, y, u, w, we, wc)
        Dv = lambda Lr, Pr: H * sum((p * (l - q) for p, l, q in zip(Pr, Lr, lQ)), arb(0))
        C = Dv(wallp[0], wallp[1])
        L0, P0, S0 = F.rows(y, arb(0))
        if wc is None:
            return Dv(L0, P0) - C
        return H * sum(
            (p * (s * s * (l - q + 2) - (l - q + 1)) for p, s, l, q in zip(P0, S0, L0, lQ)), arb(0)
        )

    def tangent(self, A, u, w, we, wc):
        """dz/dA from the implicit-function theorem at the state (u, w, we, wc) assumed converged at A; J from the last Newton call is NOT reused here"""
        F = self.F
        y = F.grid(A)
        r, J = F.residual_and_jacobian(A, u, w, we, wc, y=y)
        dl = arb("1e-25")
        rp, _ = F.residual_and_jacobian(A + dl, u, w, we, wc, want_J=False, y=y)
        rm, _ = F.residual_and_jacobian(A - dl, u, w, we, wc, want_J=False, y=y)
        b = [(a - c) / (2 * dl) for a, c in zip(rp, rm)]
        sol = J.solve(arb_mat([[-t.mid()] for t in b]))
        return [sol[i, 0].mid() for i in range(J.nrows())], J

    def tangent_from_J(self, A, u, w, we, wc, J, y):
        F = self.F
        dl = arb("1e-25")
        rp, _ = F.residual_and_jacobian(A + dl, u, w, we, wc, want_J=False, y=y)
        rm, _ = F.residual_and_jacobian(A - dl, u, w, we, wc, want_J=False, y=y)
        b = [(a - c) / (2 * dl) for a, c in zip(rp, rm)]
        sol = J.solve(arb_mat([[-t.mid()] for t in b]))
        return [sol[i, 0].mid() for i in range(J.nrows())]

    @staticmethod
    def pack(u, w, we, wc):
        return list(u) + list(w) + [we] + ([wc] if wc is not None else [])

    @staticmethod
    def unpack(v, m, odd):
        return v[:m], v[m : 2 * m], v[2 * m], (v[2 * m + 1] if odd else None)

    def save(self, K, A, u, w, we, wc, res, verdict, sig):
        d = {
            "A": S(A, 40),
            "K": K,
            "residual": float(res),
            "verdict": verdict,
            "centre_signal": float(sig.mid()),
            "u": [S(t, 40) for t in u],
            "w": [S(t, 40) for t in w],
            "we": S(we, 40),
            "wc": (None if wc is None else S(wc, 40)),
        }
        fn = os.path.join(self.out, f"results_q1411_gauss_K{K}_A{float(A):.6f}.json")
        json.dump(d, open(fn, "w"))
        self.nsave += 1

    # ---- one family
    def walk_family(self, A, u, w, we, wc, A_end, h=2e-4, hmax=0.03):
        """walk up from the converged state at A with a cubic Hermite predictor (previous state and tangent) and the tangent predictor at the first step;
        returns ('event', A_lo, state_lo, s_lo, A_hi, state_hi, s_hi) or ('end', A, state) or ('stuck', A, state)
        """
        F = self.F
        m = len(u)
        odd = wc is not None
        K = 2 * m + 2 + (1 if odd else 0)
        y = F.grid(A)
        sig = self.signal(A, u, w, we, wc, y)
        res = arb(0)
        say(
            f"family K = {K} ({'odd' if odd else 'even'}) from A = {float(A):.6f}, signal {float(sig.mid()):.3e}, h = {h}"
        )
        self.save(K, A, u, w, we, wc, res, "OPTIMAL" if sig < 0 else "event", sig)
        dz, J = self.tangent(A, u, w, we, wc)
        prev = None
        while float(A) < A_end - 1e-12:
            v0 = self.pack(u, w, we, wc)
            step = min(h, A_end - float(A))
            ok = False
            for attempt in range(14):
                hh = arb(step)
                An = A + hh
                if prev is None:
                    vp = [a + hh * d for a, d in zip(v0, dz)]
                else:
                    Ap, vprev, dzp = prev
                    hp = A - Ap
                    t = 1 + hh / hp
                    h00 = 2 * t**3 - 3 * t**2 + 1
                    h10 = t**3 - 2 * t**2 + t
                    h01 = -2 * t**3 + 3 * t**2
                    h11 = t**3 - t**2
                    vp = [
                        h00 * a0 + h10 * hp * d0 + h01 * a1 + h11 * hp * d1
                        for a0, d0, a1, d1 in zip(vprev, dzp, v0, dz)
                    ]
                up, wp, wep, wcp = self.unpack(vp, m, odd)
                if not F.feasible(An, up, wp, wep, wcp):
                    step /= 2
                    continue
                try:
                    un, wn, wen, wcn, res, k = F.newton(An, up, wp, wep, wcp, tol="1e-30", steps=12)
                except Exception as e:
                    say("   newton exception", e)
                    step /= 2
                    continue
                if res < self.tol and F.feasible(An, un, wn, wen, wcn):
                    ok = True
                    break
                step /= 2.5
            if not ok:
                say(f"   step failed at A = {float(A):.6f} (h down to {step:.2e}); stop")
                return ("stuck", A, (u, w, we, wc))
            yn = F.last_y
            Jn = F.last_J
            sign = self.signal(An, un, wn, wen, wcn, yn)
            if sign > 0:  # the event lies between A and An
                say(
                    f"   EVENT between A = {float(A):.6f} (signal {float(sig.mid()):.3e}) and {float(An):.6f} (signal {float(sign.mid()):.3e})"
                )
                return ("event", A, (u, w, we, wc), sig, An, (un, wn, wen, wcn), sign)
            dzn = self.tangent_from_J(An, un, wn, wen, wcn, Jn, yn)
            prev = (A, v0, dz)
            A, u, w, we, wc, sig, dz = An, un, wn, wen, wcn, sign, dzn
            self.save(K, A, u, w, we, wc, res, "OPTIMAL", sig)
            say(
                f"   A = {float(A):.6f}: residual {float(res):.1e} in {k} it, signal {sig.mid().str(6, radius=False)}, h {step:.2e}  [{time.time() - T0:.0f}s]"
            )
            h = min(hmax, step * (2.0 if k <= 6 else (1.4 if k <= 7 else (1.0 if k <= 8 else 0.5))))
        return ("end", A, (u, w, we, wc))

    # ---- locate the event by regula falsi on the signal along the family
    def locate(self, lo, hi, tolA=1e-14, smax=1e-30):
        """event location along the family: secant on the signal inside the bracket [lo, hi] (lo: signal < 0, hi: signal > 0), states re-solved by Newton from the
        tangent predictor of the nearer end; returns (A0, st0, s0, A1, st1, s1, A_event) with A_event the linear interpolation of the final bracket
        """
        A0, st0, s0, A1, st1, s1 = lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
        F = self.F
        m = len(st0[0])
        odd = st0[3] is not None
        pts = [(A0, s0), (A1, s1)]
        cache = {}

        def pred(Aend, st, Am):
            key = float(Aend)
            if key not in cache:
                cache[key] = self.tangent(Aend, *st)[0]
            dz = cache[key]
            v0 = self.pack(*st)
            return self.unpack([a + (Am - Aend) * d for a, d in zip(v0, dz)], m, odd)

        for it in range(40):
            if float(A1 - A0) < tolA:
                break
            (Aa, sa), (Ab, sb) = pts[-2], pts[-1]
            if sb != sa and float(abs(sb - sa).mid()) > 0:
                Am = Ab - sb * (Ab - Aa) / (sb - sa)
            else:
                Am = (A0 + A1) / 2
            if not (A0 < Am < A1):
                Am = (A0 + A1) / 2
            Am = arb(Am.mid())
            near_lo = (Am - A0) < (A1 - Am)
            Aend, st = (A0, st0) if near_lo else (A1, st1)
            up, wp, wep, wcp = pred(Aend, st, Am)
            un, wn, wen, wcn, res, k = F.newton(
                Am, up, wp, wep, wcp, tol="1e-36", steps=30, verbose=(os.environ.get("LOCV") == "1")
            )
            if not (res < arb("1e-30")):
                say(
                    f"   locate: Newton failed (res {float(res):.2e} in {k} it, Am - Aend = {float(Am - Aend):.2e}); bisect"
                )
                Am = (A0 + A1) / 2
                up, wp, wep, wcp = pred(A0, st0, Am)
                un, wn, wen, wcn, res, k = F.newton(Am, up, wp, wep, wcp, tol="1e-36", steps=40)
            if not (res < arb("1e-30")):
                say("   locate: failed")
                break
            sm = self.signal(Am, un, wn, wen, wcn, F.last_y)
            say(f"   locate: A = {Am.str(16, radius=False)} signal {float(sm.mid()):.3e}")
            pts.append((Am, sm))
            if sm < 0:
                A0, st0, s0 = Am, (un, wn, wen, wcn), sm
                cache.clear()
            else:
                A1, st1, s1 = Am, (un, wn, wen, wcn), sm
                cache.clear()
            if abs(sm) < arb(smax):
                break
        f0 = abs(s0)
        f1 = abs(s1)
        Aev = A0 + (A1 - A0) * f0 / (f0 + f1)
        return A0, st0, s0, A1, st1, s1, Aev

    # ---- the new family after an event
    def after_birth(self, A_lo, st_lo, A_hi):
        """even -> odd: add a centre atom; tangent of the odd branch at wc = 0 from the odd Jacobian"""
        F = self.F
        u, w, we, wc = st_lo
        assert wc is None
        m = len(u)
        bsteps = int(
            os.environ.get("Q1411_BIRTH_STEPS", "90")
        )  # cap of the Newton of the odd start (switch added 9 Oct 2026, amendment 2197); the default 90 is the old value
        diag = (
            os.environ.get("Q1411_BIRTH_DIAG") == "1"
        )  # print-only diagnostics (switch added 9 Oct 2026, amendment 2194); default off
        for delta in (1e-5, 3e-6, 3e-5, 1e-6, 1e-4) + tuple(
            float(t) for t in os.environ.get("Q1411_BIRTH_EXTRA", "").split(",") if t.strip()
        ):  # extra offsets after the five old ones, default none
            A1 = A_lo + arb(delta)
            dz, J = self.tangent(
                A_lo, u, w, we, arb(0)
            )  # odd system at wc = 0 (J includes the wc column; the extra row D(0) - D(A))
            v0 = self.pack(u, w, we, arb(0))
            vp = [a + arb(delta) * d for a, d in zip(v0, dz)]
            up, wp, wep, wcp = self.unpack(vp, m, True)
            say(f"   birth seed delta {delta}: predicted wc {float(wcp.mid()):.3e}")
            if diag:
                say(
                    f"   birth diag delta {delta}: predicted wc positive {bool(wcp > 0)}, feasible {F.feasible(A1, up, wp, wep, wcp)}"
                )
            if not (wcp > 0 and F.feasible(A1, up, wp, wep, wcp)):
                for wseed in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
                    wcp = arb(wseed)
                    up, wp, wep = u, w, we
                    try:
                        un, wn, wen, wcn, res, k = F.newton(A1, up, wp, wep, wcp, tol="1e-30", steps=90)
                    except Exception as e:
                        continue
                    if res < self.tol and F.feasible(A1, un, wn, wen, wcn):
                        return A1, (un, wn, wen, wcn)
                continue
            try:
                un, wn, wen, wcn, res, k = F.newton(A1, up, wp, wep, wcp, tol="1e-30", steps=bsteps)
            except Exception as e:
                if diag:
                    say(f"   birth diag delta {delta}: Newton exception {type(e).__name__}")
                continue
            if diag:
                say(
                    f"   birth diag delta {delta}: Newton residual {float(res):.2e} in {k} it, feasible {F.feasible(A1, un, wn, wen, wcn)}, wc positive {bool(wcn > 0)}"
                )
            if res < arb(os.environ.get("Q1411_BIRTH_TOL", "1e-30")) and F.feasible(
                A1, un, wn, wen, wcn
            ):  # Q1411_BIRTH_TOL added 10 Oct 2026 (reading 179): acceptance residual of the odd start after a birth, default 1e-30 = self.tol, so the old path is unchanged
                sg = self.signal(A1, un, wn, wen, wcn, F.grid(A1))
                say(
                    f"   birth: odd family start at A = {float(A1):.7f}, wc {float(wcn.mid()):.3e}, D''(0) {float(sg.mid()):.3e}"
                )
                if diag:
                    say(f"   birth diag delta {delta}: signal negative {bool(sg < 0)}")
                if sg < 0:
                    return A1, (un, wn, wen, wcn)
        return None

    def bordered_newton(self, A0, u, w, we, eps, steps=40, verbose=False, growth=5.0):
        """pitchfork branch of the even system: unknowns (v, A) with the extra equation u_1 = eps (the branch is regular in eps, not in A: A - A_s ~ eps^2).
        The grid is frozen at A0."""
        F = self.F
        m = len(u)
        n = 2 * m + 1
        A = arb(A0.mid())
        y = F.grid(A)
        v = [arb(t.mid()) for t in (list(u) + list(w) + [we])]
        dl = arb("1e-25")

        def system(v, A):
            up, wp, wep = v[:m], v[m : 2 * m], v[2 * m]
            r, J = F.residual_and_jacobian(A, up, wp, wep, None, y=y)
            rp, _ = F.residual_and_jacobian(A + dl, up, wp, wep, None, want_J=False, y=y)
            rm, _ = F.residual_and_jacobian(A - dl, up, wp, wep, None, want_J=False, y=y)
            M = arb_mat(n + 1, n + 1)
            for i in range(n):
                for j in range(n):
                    M[i, j] = J[i, j]
                M[i, n] = (rp[i] - rm[i]) / (2 * dl)
            M[n, 0] = 1
            rhs = [-t.mid() for t in r] + [-(v[0] - eps)]
            sc = (
                1 / eps
            )  # the row D'(u_1) vanishes with u_1 at the pitchfork: use D'(u_1) / u_1 (regular), u_1 = eps being fixed
            for j in range(n + 1):
                M[m, j] = M[m, j] * sc
            rhs[m] = rhs[m] * sc
            return r, M, rhs

        nrm = lambda d: max(abs(t.mid()) for t in d)
        r, M, rhs = system(v, A)
        sol = M.solve(arb_mat([[t] for t in rhs]))
        dz = [sol[i, 0].mid() for i in range(n + 1)]
        nd = nrm(dz)
        res = nrm(r)
        k = 0
        while (
            k < steps and nd > arb("1e-40") and res > arb("1e-45")
        ):  # stop at the residual floor: the correction size stays near cond x residual and never reaches 1e-40 (40 idle iterations per pitchfork step before)
            lam = arb(1)
            done = False
            for _ in range(10):
                vn = [v[i] + lam * dz[i] for i in range(n)]
                An = A + lam * dz[n]
                if (
                    all(vn[i] > 0 for i in range(m, 2 * m + 1))
                    and vn[0] > 0
                    and all(vn[i] < vn[i + 1] for i in range(m - 1))
                    and vn[m - 1] < An
                ):
                    rn, Mn, rhsn = system(vn, An)
                    soln = Mn.solve(arb_mat([[t] for t in rhsn]))
                    dzn = [soln[i, 0].mid() for i in range(n + 1)]
                    ndn = nrm(dzn)
                    if ndn <= growth * nd or nrm(rn) < arb("1e-40"):
                        v, A, r, dz, nd, res, done = vn, An, rn, dzn, ndn, nrm(rn), True
                        break
                lam /= 2
            if not done:
                break
            k += 1
            if verbose:
                say(
                    f"      bordered it {k}: lam {float(lam):.3g}, residual {float(res):.2e}, next step {float(nd):.2e}, A = {A.str(15, radius=False)}, u_1 = {float(v[0].mid()):.4f}"
                )
        return v[:m], v[m : 2 * m], v[2 * m], A, res, k

    def after_split(self, A_lo, st_lo, verbose=True):
        """odd -> even: the centre atom (mass wc) is replaced by a pair at +-eps, each of mass wc/2 (mirror-reduced: u_1 = eps, w_1 = wc/2). Near the pitchfork the new
        branch is z(t) = z_s + t z_2, A(t) = A_s + t A_2 with t = eps^2, so it is followed with u_1 = eps fixed and A unknown (bordered Newton) along a geometric
        sequence of eps from 1e-3 to 0.12, with a linear predictor in t from the last two solutions; the last state is re-solved at fixed A.
        """
        F = self.F
        u, w, we, wc = st_lo
        assert wc is not None
        v_s = [arb(t.mid()) for t in ([arb(0)] + list(u) + [wc / 2] + list(w) + [we])]
        A_s = arb(A_lo.mid())
        sol = [
            (arb(0), v_s, A_s)
        ]  # list of (t, vector (u, w, we), A); t = 0 is the odd state at the split (the limit of the branch)
        eps = 1e-4
        ratio = 1.7
        m = len(u) + 1
        fails = 0
        EPS_END = float(os.environ.get("Q1411_EPS_END", "0.3"))
        while True:
            t = arb(eps) ** 2
            if len(sol) >= 3:  # quadratic (Lagrange) predictor in t from the last three solutions
                (t0_, v0_, a0_), (t1, v1, a1), (t2, v2, a2) = sol[-3], sol[-2], sol[-1]
                l0 = (t - t1) * (t - t2) / ((t0_ - t1) * (t0_ - t2))
                l1 = (t - t0_) * (t - t2) / ((t1 - t0_) * (t1 - t2))
                l2 = (t - t0_) * (t - t1) / ((t2 - t0_) * (t2 - t1))
                vp = [l0 * x0 + l1 * x1 + l2 * x2 for x0, x1, x2 in zip(v0_, v1, v2)]
                Ap = l0 * a0_ + l1 * a1 + l2 * a2
            else:
                (t1, v1, a1), (t2, v2, a2) = (sol[-2], sol[-1]) if len(sol) >= 2 else (sol[0], sol[0])
                if t2 != t1:
                    f = (t - t2) / (t2 - t1)
                    vp = [x2 + f * (x2 - x1) for x1, x2 in zip(v1, v2)]
                    Ap = a2 + f * (a2 - a1)
                else:
                    vp = list(v2)
                    Ap = a2
            vp[0] = arb(eps)
            uu, ww, wwe = vp[:m], vp[m : 2 * m], vp[2 * m]
            try:
                u2, w2, we2, Ab, res, k = self.bordered_newton(Ap, uu, ww, wwe, arb(eps))
            except Exception as e:
                u2 = None
                res = arb(1)
            ok = (u2 is not None) and res < arb("1e-25") and F.feasible(Ab, u2, w2, we2, None)
            if verbose:
                say(
                    f"   split eps {eps:.4g}: "
                    + (
                        f"residual {float(res):.2e} in {k} it, A - A_s = {float(Ab - A_s):.4e}, u_1 {float(u2[0].mid()):.5f}"
                        if u2 is not None
                        else "exception"
                    )
                )
            if ok:
                sol.append((t, [x.mid() for x in (list(u2) + list(w2) + [we2])], arb(Ab.mid())))
                if eps >= EPS_END:
                    break
                eps = min(EPS_END * 1.0001, eps * ratio)
            else:
                fails += 1
                ratio = 1 + (ratio - 1) / 2
                if (
                    fails > int(os.environ.get("Q1411_SPLIT_FAILS_MAX", "8"))
                    or len(sol) < 2
                    and fails > int(os.environ.get("Q1411_SPLIT_FIRST_FAILS", "3"))
                ):
                    say("   split branch lost")
                    return None  # switch added 9 Oct 2026 (amendment 2191); the default 3 is the old value, so the code path is unchanged unless the variable is set; Q1411_SPLIT_FAILS_MAX added 10 Oct 2026 (cumulative failure limit, default 8 is the old value; reading 160)
                eps = float(sol[-1][0].sqrt().mid()) * ratio if len(sol) > 1 else eps / 1.5
        t, v, Ab = sol[-1]
        m = len(u) + 1
        try:
            un, wn, wen, wcn, res2, k2 = F.newton(
                Ab, v[:m], v[m : 2 * m], v[2 * m], None, tol="1e-30", steps=20
            )
        except Exception as e:
            return None
        if res2 < self.tol and F.feasible(Ab, un, wn, wen, wcn):
            sg = self.signal(Ab, un, wn, wen, wcn, F.last_y)
            say(
                f"   split: even family start at A = {Ab.str(12, radius=False)}, u_1 {float(un[0].mid()):.4f}, centre signal {float(sg.mid()):.3e}"
            )
            if sg < 0:
                return Ab, (un, wn, wen, wcn)
        return None

    def run(self, A, st, A_end, hmax=0.03):
        u, w, we, wc = st
        while float(A) < A_end - 1e-12:
            out = self.walk_family(A, u, w, we, wc, A_end, hmax=hmax)
            if out[0] != "event":
                return out
            _, A_lo, st_lo, s_lo, A_hi, st_hi, s_hi = out
            A_lo2, st_lo2, s_lo2, A_hi2, st_hi2, s_hi2, A_ev = self.locate(
                (A_lo, st_lo, s_lo), (A_hi, st_hi, s_hi)
            )
            K_before = 2 * len(st_lo[0]) + 2 + (1 if st_lo[3] is not None else 0)
            say(
                f"EVENT located: K_b = {K_before} ({'split' if st_lo[3] is not None else 'birth'}) A_event = {A_ev.str(20, radius=False)} (bracket {float(A_hi2 - A_lo2):.2e})"
            )
            json.dump(
                {
                    "K_b": K_before,
                    "kind": "birth" if st_lo[3] is None else "split",
                    "A_event": A_ev.str(20, radius=False),
                    "A_lo": S(A_lo2, 30),
                    "A_hi": S(A_hi2, 30),
                    "s_lo": float(s_lo2.mid()),
                    "s_hi": float(s_hi2.mid()),
                },
                open(os.path.join(self.out, f"event_Kb{K_before}.json"), "w"),
            )
            nxt = (
                self.after_birth(A_lo2, st_lo2, A_hi2)
                if st_lo2[3] is None
                else self.after_split(A_lo2, st_lo2)
            )
            if nxt is None:
                say("new family could not be started; stop")
                return ("stuck-event", A_lo2, st_lo2)
            A, (u, w, we, wc) = nxt
        return ("end", A, (u, w, we, wc))


def load(fn, F):
    d = json.load(open(fn))
    d = d[-1] if isinstance(d, list) else d
    return (
        arb(d["A"]),
        [arb(x) for x in d["u"]],
        [arb(x) for x in d["w"]],
        arb(d["we"]),
        (None if d.get("wc") in (None, "None") else arb(d["wc"])),
    )


if __name__ == "__main__":
    if sys.argv[1] == "control":
        W = Walker(192, "fastwalk_control")
        fn = os.path.join("..", "capacity_even62_family", "results_q220_gauss_K62_A42.2582.json")
        A, u, w, we, wc = load(fn, W.F)
        say("start state A =", float(A))
        u, w, we, wc, res, k = W.solve_at(A, u, w, we, wc)
        say(f"re-solved at the decimal A: residual {float(res):.1e} in {k} it")
        out = W.run(A, (u, w, we, wc), 42.2800, hmax=0.01)
        say("run returned", out[0], "at A =", float(out[1]))
    else:
        W = Walker(192, sys.argv[2], H=float(os.environ.get("Q1411_H", "0.05")))
        A, u, w, we, wc = load(sys.argv[1], W.F)
        u, w, we, wc, res, k = W.solve_at(A, u, w, we, wc)
        say(f"start re-solved: residual {float(res):.1e}")
        out = W.run(
            A, (u, w, we, wc), float(sys.argv[3]), hmax=(float(sys.argv[4]) if len(sys.argv) > 4 else 0.03)
        )
        say("run returned", out[0], "at A =", float(out[1]))
