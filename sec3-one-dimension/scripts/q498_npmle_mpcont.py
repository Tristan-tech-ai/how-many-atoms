"""Q498: arb continuation of the population NPMLE (Gaussian location mixture, data uniform on [-a, a]) past a = 33.65,
where the arb run of q496 (mpsweep) stopped. Same problem, same symmetric parametrisation, same quadrature
(composite Gauss-Legendre, q = 16 nodes per panel of width <= 0.5) and the same arb system (q496.MPC.state / FJ).

WHAT IS NEW (method only; the problem and the stationarity system are those of q496):
 * NEWTON on the augmented system. Unknowns (z, a) with one linear constraint g.z = par (the young pair's half-separation
   s, the new centre weight c, or the half-gap of a split pair), the analytic dF/da column, exact Jacobian, backtracking on
   the 2-norm; with no constraint it is the plain fixed-a Newton.
 * HOMOTOPY fallback (as q491 MP.polish_h): solve F(u) = (1 - tau) F(u0), tau 0 -> 1 in adaptive steps (x2 on success,
   /4 on failure), Newton corrector with exact Jacobian and 2-norm backtracking at each step.
 * ORDINARY STEPS of 0.05 in a: Newton from the quadratic extrapolation through the last three states, then from the
   secant; then (v2) LOCAL-PARAMETER CONTINUATION (LPC): at each step the unknown of (z, a) with the largest component of
   the secant tangent is held fixed (a free otherwise), step h along the tangent (x1.5 on success, /2 on failure, h <=
   0.05), Lagrange predictor in that parameter through the last three points, points accepted at max |F| < 1e-18, a flag
   raised on an LPC point is located along the LPC parameter; landing on the step point by interpolation in a, else by
   Illinois regula falsi in arb in the path parameter on a(par) - a_target, then the fixed-a Newton polish; then the
   homotopy from each seed; a failed step is halved. v1 (quadratic, secant, homotopy only) could not step 33.70 -> 33.75
   after the 45 -> 46 split: there the young pair moves with ds/da ~ 3 and the fixed-a Newton basin is tiny.
   v2b (speed only, resumed at 35.05): the seed Newton is capped at 10 iterations, and LPC is tried first right after an
   event and while the previous LPC step needed a parameter other than a. v2c (speed only, resumed at 36.45): Newton
   backtracking evaluates F without the Jacobian; LPC keeps a as its parameter unless another unknown moves at least twice
   as fast along the tangent (the outermost atoms move with dp/da close to 1). v2d (speed only, resumed at 40.85): factor
   1.25 instead of 2 (with 2, the first steps after a split were fixed-a LPC steps, up to 416 s for 0.05 at 40.35).
 * EVENTS: flags as in q496 (weight < 0, D''(p_k) > 0, D''(0) > 0 with a centre atom, D(0) > 1 without). The crossing of
   the flagged signal along the OLD family is located by Illinois regula falsi to 1e-9 in a. The new family is followed
   by NATURAL-PARAMETER continuation (as q492 young_mp): s (split) or c (birth) held fixed, a free, from s0 = 1e-3 or
   c0 = 1e-6 (divided by 10, then 100, if the first fixed-parameter Newton at a* fails), ratio 1.05 per step, QUADRATIC predictor in the parameter through the last three path points, intermediate
   points accepted at max |F| < 1e-18, at most 600 steps; then the landing on a_target (interpolation in a, else Illinois
   regula falsi in arb on a(par) - a_target, then the fixed-a polish; the homotopy as the fallback). A failed path step is
   retried with the ratio step halved (to 1.05^(1/16)), then by the homotopy at the fixed parameter, then (v2) by LPC
   from the last three path points. A change less than 0.01 below the step point starts the new family at the next step
   point.
 * REPORT ROWS every 0.25 in a: full arb KKT scan (16 points per gap between consecutive atoms, the minimum of each gap
   refined by Newton on D', every interior local maximum of the samples refined and checked, 30 points beyond the last
   atom, D'' < 0 at every atom, weights > 0). Signal S = the shallowest dip of D - 1 over all gaps (the centre gap).
   Noise nu = max(max |F|, radius of the arb value of D - 1 at the dip, |change of S when the quadrature is changed from
   q = 16 to q = 24 nodes per panel and the state re-polished|). The count is reported as RESOLVED only if max |F| < the
   Newton tolerance, the KKT scan is clean and S >= 1e3 nu (q496 P3).
 * PRECISION: 128 bits for a < 36, 160 bits for 36 <= a < 40, 192 bits for a >= 40 (the signal falls like
   exp(-2.845 a^(2/3))); raised by 32 more bits at a report row whose arb noise max(max |F|, radius) exceeds 1e-9 S.

Usage:
  py q498_npmle_mpcont.py oldfamily          (the K = 45 family continued past 33.6487: sign changes of D'' at the atoms)
  py q498_npmle_mpcont.py control            (dF/da vs central differences; q498 rows vs the q496 arb rows at 33.25, 33.5)
  py q498_npmle_mpcont.py run AEND [resume]  (continuation from the q496 arb row at a = 33.5; results_q498_npmle_mpcont.json)
  py q498_npmle_mpcont.py summarylog         (table of the change points and the rows, into the log)
  py q498_npmle_mpcont.py compare            (AFTER the run only: the frozen npmle_forward_predictions.json, rule 642(5))

RESULT (run of 23 Sep, 16:29 - 23:27, log q498_npmle_mpcont.log): 21 changes located in (33.5, 45], all at the centre,
alternating split / birth, K = 45 -> 66; all 46 report rows (33.5 .. 45.0; 43.5 skipped by the 0.01 rule) KKT-clean and
resolved (min S/nu 1.5e12). The event at 33.6487 is an ordinary centre split; the D'' sign changes at p1, p2, ... occur
only along the old K = 45 family continued past a*.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv
sys.argv = [sys.argv[0]]
import q496_npmle_uniform as Q  # main-guarded; nothing runs at import

sys.argv = _argv
from flint import arb, arb_mat, ctx

say = lambda *a: print(*a, flush=True)
OUT = os.path.join(HERE, "results_q498_npmle_mpcont.json")
LOGF = os.path.join(HERE, "q498_npmle_mpcont.log")
SRC = os.path.join(HERE, "results_q496_npmle_uniform_mpsweep.json")
TOLY = 1e-18  # intermediate natural-parameter points (seeds only)
RATIO = 1.05
DA = 0.05
_log = [None]


def out(s):
    say(s)
    if _log[0] is not None:
        _log[0].write(s + "\n")
        _log[0].flush()


def prec_for(a):
    return 128 if a < 36 else (160 if a < 40 else 192)


def K_of(n, centre):
    return 2 * n + (1 if centre else 0)


def fl(x):
    return float(x.mid()) if isinstance(x, arb) else float(x)


class MPX:
    def __init__(self, prec=128, q=16):
        self.q = q
        self.set_prec(prec)

    def set_prec(self, prec):
        self.prec = prec
        self.M = Q.MPC(prec, self.q)  # sets ctx.prec
        self.tol = 10.0 ** (-(0.30103 * prec - 13.5))  # 128 -> 1e-25 (q496), 160 -> 2e-35, 192 -> 2e-44

    # ---------------------------------------------------------------- system
    def evalF(self, z, n, centre, A, jac=True):
        st = self.M.state(z, n, centre, A)
        if st is None:
            return None
        o = self.M.FJ(st, jac=jac)
        o["st"] = st
        return o

    def dFda(self, o, n, centre):
        st = o["st"]
        A = st["A"]
        p, v, c = st["p"], st["v"], st["c"]
        phi = self.M.phi
        F = o["F"]
        mA = sum((v[k] * (phi(A - p[k]) + phi(A + p[k])) for k in range(n)), arb(0)) + (
            c * phi(A) if centre else 0
        )
        col = [(phi(A - p[k]) + phi(A + p[k])) / (2 * A * mA) - (F[k] + 1) / A for k in range(n)]
        col += [
            ((A - p[k]) * phi(A - p[k]) - (A + p[k]) * phi(A + p[k])) / (2 * A * mA) - F[n + k] / A
            for k in range(n)
        ]
        if centre:
            col.append(2 * phi(A) / (2 * A * mA) - (F[2 * n] + 1) / A)
        return col

    @staticmethod
    def valid(z, n, A):
        pn = [fl(t) for t in z[:n]]
        return (not n or (pn[0] > 0 and all(pn[i] < pn[i + 1] for i in range(n - 1)))) and fl(A) > 0.5

    def _step_matrix(self, o, n, centre, con):
        J = o["J"]
        L = J.nrows()
        if con is None:
            return J
        col = self.dFda(o, n, centre)
        Ja = arb_mat(L + 1, L + 1)
        for i in range(L):
            for j in range(L):
                Ja[i, j] = J[i, j]
            Ja[i, L] = col[i]
        for idx, cf in con:
            Ja[L, idx] = arb(cf)
        return Ja

    def solve(self, z, n, centre, A, con=None, maxit=30, tol=None, shift=None):
        """Newton on F(z, a) - shift = 0 (shift: the homotopy offset (1 - tau) F0, or None). con None: a fixed.
        con = [(index, coeff)]: a free and g.z held at its seed value. Backtracking on the 2-norm (12 halvings).
        Returns (z, A, max|F - shift|, o)."""
        tol = self.tol if tol is None else tol
        A = arb(A) if not isinstance(A, arb) else A
        z = [t.mid() if isinstance(t, arb) else arb(t) for t in z]
        o = self.evalF(z, n, centre, A)
        if o is None:
            return z, A, np.inf, None
        L = len(z)
        G = lambda o_: [o_["F"][i] - (shift[i] if shift is not None else 0) for i in range(L)]
        g = G(o)
        s2 = sum((x * x for x in g), arb(0)).mid()
        nF = max(abs(fl(x)) for x in g)
        for it in range(maxit):
            if nF < tol:
                break
            if "J" not in o:  # v2c: backtracking evaluates F only; J once per accepted point
                o2 = self.M.FJ(o["st"], jac=True)
                o2["st"] = o["st"]
                o = o2
            Mx = self._step_matrix(o, n, centre, con)
            rhs = arb_mat([[x.mid()] for x in g] + ([[arb(0)]] if con is not None else []))
            try:  # approximate LU solve (midpoints): a Newton step needs no bounds
                d = Mx.solve(rhs, algorithm="approx")
            except (ZeroDivisionError, ValueError):
                break
            if not all(d[i, 0].is_finite() for i in range(d.nrows())):
                break
            lam = arb(1)
            ok = False
            for _ in range(12):
                zn = [(z[i] - lam * d[i, 0]).mid() for i in range(L)]
                An = (A - lam * d[L, 0]).mid() if con is not None else A
                if self.valid(zn, n, An):
                    on = self.evalF(zn, n, centre, An, jac=False)
                    if on is not None:
                        gn = G(on)
                        sn = sum((x * x for x in gn), arb(0)).mid()
                        if sn < s2:
                            z, A, o, g, s2, ok = zn, An, on, gn, sn, True
                            break
                lam = lam / 2
            if not ok:
                break
            nF = max(abs(fl(x)) for x in g)
        return z, A, nF, o

    def homotopy(self, z0, n, centre, A0, con=None, tol=None, max_steps=200, verbose=False):
        """F(u) = (1 - tau) F(u0), tau 0 -> 1; returns (z, A, max|F|, o)."""
        tol = self.tol if tol is None else tol
        A0 = arb(A0) if not isinstance(A0, arb) else A0
        z = [t.mid() if isinstance(t, arb) else arb(t) for t in z0]
        A = A0
        o0 = self.evalF(z, n, centre, A, jac=False)
        if o0 is None:
            return z, A, np.inf, None
        F0 = [f.mid() for f in o0["F"]]
        nF0 = max(abs(fl(f)) for f in F0)
        tau = 0.0
        dtau = 1.0
        steps = 0
        while tau < 1 and steps < max_steps:
            steps += 1
            tn = min(tau + dtau, 1.0)
            sh = [(f * arb(1 - tn)).mid() for f in F0] if tn < 1 else None
            ctol = 1e-8 * nF0 * (1 - tn) if tn < 1 else tol
            zn, An, nFn, on = self.solve(z, n, centre, A, con=con, maxit=12, tol=ctol, shift=sh)
            if nFn < ctol or (tn == 1 and nFn < tol):
                z, A, tau = zn, An, tn
                dtau *= 2
                if verbose:
                    out(f"         homotopy tau {tau:.5f} ok")
            else:
                dtau /= 4
                if dtau < 1e-6:
                    break
        z, A, nF, o = self.solve(z, n, centre, A, con=con, maxit=4)
        return z, A, nF, o

    # ---------------------------------------------------------------- D at many points (arb_mat products)
    def D_many(self, st, ts):
        """D, D', D'' at the points ts (arb), vectorised: D0 = P r, D1 = Mn xr - t P r, D2 = P x2r - 2t Mn xr + (t^2 - 1) P r
        with P = phi(x - t) + phi(x + t), Mn = phi(x - t) - phi(x + t)."""
        xs, r, N = st["xs"], st["r"], st["N"]
        phi = self.M.phi
        rv = arb_mat([[y] for y in r])
        xr = arb_mat([[xs[i] * r[i]] for i in range(N)])
        x2r = arb_mat([[xs[i] * xs[i] * r[i]] for i in range(N)])
        res = []
        for s in range(0, len(ts), 64):
            tt = ts[s : s + 64]
            E1 = [[phi(x - t) for x in xs] for t in tt]
            E2 = [[phi(x + t) for x in xs] for t in tt]
            P = arb_mat([[E1[j][i] + E2[j][i] for i in range(N)] for j in range(len(tt))])
            Mn = arb_mat([[E1[j][i] - E2[j][i] for i in range(N)] for j in range(len(tt))])
            Pr = P * rv
            Px2 = P * x2r
            Mx = Mn * xr
            for j, t in enumerate(tt):
                res.append(
                    (Pr[j, 0], Mx[j, 0] - t * Pr[j, 0], Px2[j, 0] - 2 * t * Mx[j, 0] + (t * t - 1) * Pr[j, 0])
                )
        return res

    def refine_ext(self, st, t0, lo, hi, iters=8):
        t = arb(t0)
        for _ in range(iters):
            ((d0, d1, d2),) = self.D_many(st, [t])
            if fl(d2) == 0:
                break
            tn = (t - d1 / d2).mid()
            if not (lo <= fl(tn) <= hi):
                break
            t = tn
        ((d0, d1, d2),) = self.D_many(st, [t])
        return t, d0, d1, d2

    def flags(self, z, n, centre, o):
        return self.M.flags(z, n, centre, o)

    def signals(self, z, n, centre, o):
        """the flag signals (positive = change needed)"""
        s = {}
        v = [fl(t) for t in z[n : 2 * n]] + ([fl(z[2 * n])] if centre else [])
        s["death"] = -min(v) if v else -1.0
        d2 = [fl(t) for t in o["D2"]]
        s["split"] = max(d2) if d2 else -1.0
        s["split0"] = fl(o["D2c"]) if centre else None
        s["birth0"] = fl(o["D0c"] - 1) if not centre else None
        return s

    def kkt(self, z, n, centre, o, per_gap=16):
        st = o["st"]
        A = st["A"]
        a = fl(A)
        at = ([0.0] if centre else []) + [fl(t) for t in z[:n]]
        edges = at if centre else ([-at[0]] + at if n else [])
        gmax, dips, lmax, dips_arb = -np.inf, [], [], []
        for g in range(len(edges) - 1):
            lo, hi = max(edges[g], 0.0), edges[g + 1]
            ts = [lo + (hi - lo) * (k + 0.5) / per_gap for k in range(per_gap)]
            if not centre and g == 0:
                ts = [0.0] + ts
            vals = self.D_many(st, [arb(t) for t in ts])
            dv = [fl(d0 - 1) for d0, _, _ in vals]
            gmax = max(gmax, max(dv))
            k = int(np.argmin(dv))
            if ts[k] == 0.0:
                t, d0, d1, d2 = arb(0), *vals[k]
            else:
                t, d0, d1, d2 = self.refine_ext(st, ts[k], lo, hi)
            dips.append((fl(d0 - 1), fl(t), float(d0.rad())))
            dips_arb.append(d0 - 1)
            # interior local maxima of the samples (a second hump inside a gap would signal a birth there)
            for j in range(1, len(dv) - 1):
                if dv[j] >= dv[j - 1] and dv[j] > dv[j + 1]:
                    tm, e0, _, _ = self.refine_ext(st, ts[j], lo, hi)
                    lmax.append((fl(e0 - 1), fl(tm)))
        pn = at[-1] if at else 0.0
        outer = [pn + 0.05 + (a + 6 - pn - 0.05) * k / 29 for k in range(30)]
        ov = max(fl(d0 - 1) for d0, _, _ in self.D_many(st, [arb(t) for t in outer]))
        v = [fl(t) for t in z[n : 2 * n]] + ([fl(z[2 * n])] if centre else [])
        d2 = [fl(t) for t in o["D2"]] + ([fl(o["D2c"])] if centre else [])
        S = min(-d for d, _, _ in dips) if dips else np.nan
        kmin = int(np.argmax([d for d, _, _ in dips])) if dips else None
        lm = max(lmax)[0] if lmax else -np.inf
        ok = gmax < 0 and ov < 0 and min(v) > 0 and max(d2) < 0 and all(d < 0 for d, _, _ in dips) and lm < 0
        return dict(
            ok=bool(ok),
            S=S,
            S_at=(dips[kmin][1] if dips else np.nan),
            S_rad=(dips[kmin][2] if dips else np.nan),
            S_arb=(-dips_arb[kmin] if dips else None),
            gmax=gmax,
            lmax=lm,
            outer=ov,
            minw=min(v),
            maxd2=max(d2),
            dips=[(d, t) for d, t, _ in dips],
            d2c=(fl(o["D2c"]) if centre else None),
            D0m1=(fl(o["D0c"] - 1) if not centre else None),
        )


# -------------------------------------------------------------------------------------------- helpers
def A_(x):
    return x if isinstance(x, arb) else arb(x)


def lagrange(pts, x):
    """pts = [(x_i, z_i list, A_i)] (1 to 3 of them); interpolated (z, A) at x; all weights in arb"""
    xs = [A_(p[0]) for p in pts]
    x = A_(x)
    Ls = []
    for i in range(len(pts)):
        L = arb(1)
        for j in range(len(pts)):
            if j != i:
                L = L * (x - xs[j]) / (xs[i] - xs[j])
        Ls.append(L)
    z = [sum((pts[i][1][k] * Ls[i] for i in range(len(pts))), arb(0)).mid() for k in range(len(pts[0][1]))]
    A = sum((A_(pts[i][2]) * Ls[i] for i in range(len(pts))), arb(0)).mid() if pts[0][2] is not None else None
    return z, A


def zstr(z):
    return [t.mid().str(45, radius=False) for t in z]


# ---- parameters of a path: con None = the half-width a; con = [(index, coeff)] = the linear functional g.z
def gpar(z, A, con):
    return A_(A) if con is None else sum((arb(cf) * z[i] for i, cf in con), arb(0))


def setpar(z, con, par):
    """minimum-norm change of z that gives g.z = par"""
    z = list(z)
    if con is None:
        return z
    g2 = arb(sum(cf * cf for _, cf in con))
    dl = A_(par) - gpar(z, None, con)
    for i, cf in con:
        z[i] = (z[i] + dl * arb(cf) / g2).mid()
    return z


def solve_at(X, sd, A0, n, centre, con, par, maxit=14, tol=None):
    if con is None:
        return X.solve(sd, n, centre, A_(par), maxit=maxit, tol=tol)
    return X.solve(setpar(sd, con, par), n, centre, A0, con=con, maxit=maxit, tol=tol)


def predict(P, con, par):
    """Lagrange in the parameter through the last 3 (or 2) points of P = [(z, A)], monotone in the parameter"""
    use = P[-3:]
    xs = [fl(gpar(z, A, con)) for z, A in use]
    if len(use) == 3 and not (xs[0] < xs[1] < xs[2] or xs[0] > xs[1] > xs[2]):
        use = use[-2:]
        xs = xs[-2:]
    if len(use) == 2 and xs[0] == xs[1]:
        use = use[-1:]
    if len(use) == 1:
        return list(use[0][0]), use[0][1]
    return lagrange([(gpar(z, A, con), z, A) for z, A in use], par)


def advance(X, hist, n, centre, a1, homotopy=False):
    """old-structure step to a1. Seeds: quadratic (3 states), secant (2), previous. Newton each (or the homotopy)."""
    seeds = []
    h = [(ha, hz, arb(ha)) for ha, hz in hist if abs(ha - a1) > 1e-12]
    if len(h) >= 3 and len({round(t[0], 12) for t in h[-3:]}) == 3:
        seeds.append(("quadratic", lagrange(h[-3:], a1)[0]))
    if len(h) >= 2 and abs(h[-1][0] - h[-2][0]) > 1e-12:
        seeds.append(("secant", lagrange(h[-2:], a1)[0]))
    if not seeds:
        seeds.append(("previous", list(hist[-1][1])))
    nF = np.inf
    for nm, sd in seeds:
        if homotopy:
            z, A, nF, o = X.homotopy(sd, n, centre, arb(a1))
            nm = "homotopy/" + nm
        else:
            z, A, nF, o = X.solve(
                sd, n, centre, arb(a1), maxit=10
            )  # v2b: a good seed converges in < 10 iterations
        if nF < X.tol:
            return z, o, nF, nm
    return None, None, nF, "failed"


def land_at(X, P, n, centre, a_goal, con):
    """P[-2], P[-1] bracket a_goal in a (the path parameter of the last step: con). Returns (z, o, nF, how)."""
    aG = arb(a_goal)
    zl, Al = P[-1]
    if abs(fl(Al - aG)) < 1e-30:
        z, A, nF, o = X.solve(zl, n, centre, aG)
        if nF < X.tol:
            return z, o, nF, "on target"
    Pa = [(zz, aa) for zz, aa in P[-3:]]
    sd, _ = predict(Pa, None, aG)  # interpolation in a
    z, A, nF, o = X.solve(sd, n, centre, aG)
    if nF < X.tol:
        return z, o, nF, "a-interpolation"
    if con is not None:  # Illinois regula falsi (arb) in the path parameter on a(par) - a_goal
        pts = list(P[-3:])
        lo, hi = P[-2], P[-1]
        flo, fhi = lo[1] - aG, hi[1] - aG
        side = 0
        ok = False
        for it in range(40):
            if abs(fl(fhi)) < 1e-30:
                ok = True
                break
            plo, phi_ = gpar(*lo, con), gpar(*hi, con)
            pn = (phi_ - fhi * (phi_ - plo) / (fhi - flo)).mid()
            near = sorted(pts, key=lambda q: abs(fl(gpar(*q, con) - pn)))[:3]
            near = sorted(near, key=lambda q: fl(gpar(*q, con)))
            sd2, A0 = predict(near, con, pn)
            zn, An, nn, on = solve_at(X, sd2, A0, n, centre, con, pn, maxit=14, tol=TOLY * 1e-6)
            if not (nn < TOLY):
                break
            pts.append((zn, An))
            fn = An - aG
            if fl(fn) > 0:
                hi, fhi = (zn, An), fn
                if side == 1:
                    flo = flo / 2
                side = 1
            else:
                lo, flo = (zn, An), fn
                if side == -1:
                    fhi = fhi / 2
                side = -1
            if abs(fl(fn)) < 1e-30:
                hi = (zn, An)
                ok = True
                break
        if ok:
            z, A, nF, o = X.solve(hi[0], n, centre, aG)
            if nF < X.tol:
                return z, o, nF, f"regula falsi in the parameter ({it + 1} iterates)"
    z, A, nF, o = X.homotopy(sd, n, centre, aG)
    if nF < X.tol:
        return z, o, nF, "homotopy from the a-interpolation"
    z, A, nF, o = X.homotopy(zl, n, centre, aG)
    if nF < X.tol:
        return z, o, nF, "homotopy from the last path point"
    return None, None, nF, "landing failed"


def lpc(X, P, n, centre, a_goal, hmax=0.05, max_steps=800):
    """LOCAL-PARAMETER continuation (the unknown of (z, a) with the largest component of the secant tangent is held
    fixed at each step; step h in that tangent, x1.5 on success, /2 on failure; predictor: Lagrange in the parameter
    through the last three points) from the points P = [(z, A)] (>= 2, newest last) until a >= a_goal, then landing on
    a_goal. Stops early when a flag is raised."""
    P = [(list(z), A_(A)) for z, A in P][-3:]
    L = len(P[-1][0])
    U = lambda p: np.array([fl(t) for t in p[0]] + [fl(p[1])])
    h = min(hmax, float(np.linalg.norm(U(P[-1]) - U(P[-2]))))
    steps = fails = 0
    used = {}
    while steps < max_steps:
        t = U(P[-1]) - U(P[-2])
        t /= np.linalg.norm(t)
        wt = np.ones(len(t))
        wt[L] = 1.25  # v2c: 2.0; v2d: 1.25 (a is kept unless another unknown moves 1.25x faster)
        i = int(np.argmax(np.abs(t) * wt))
        con = None if i == L else [(i, 1.0)]
        par = fl(gpar(*P[-1], con)) + h * t[i]
        land = False
        if con is None and par >= a_goal - 1e-12:
            par = a_goal
            land = True
        sd, A0 = predict(P, con, par)
        z, A, nF, o = solve_at(X, sd, A0, n, centre, con, par)
        if nF < TOLY:
            P = (P + [(z, A)])[-4:]
            steps += 1
            h = min(h * 1.5, hmax)
            used[
                (
                    "a"
                    if con is None
                    else ("p%d" % (i + 1) if i < n else ("v%d" % (i - n + 1) if i < 2 * n else "c"))
                )
            ] = 1
            fl_ = X.flags(z, n, centre, o)
            if fl_:
                return dict(
                    status="flag", P=P, o=o, flags=fl_, con=con, steps=steps, fails=fails, used=list(used)
                )
            if land or fl(A) >= a_goal:
                zf, of, nFf, how = land_at(X, P, n, centre, a_goal, con)
                if zf is None:
                    return dict(
                        status="fail", P=P, steps=steps, fails=fails, nF=nFf, how=how, used=list(used)
                    )
                return dict(
                    status="ok", P=P, z=zf, o=of, nF=nFf, how=how, steps=steps, fails=fails, used=list(used)
                )
        else:
            h /= 2
            fails += 1
            if h < 1e-10:
                return dict(
                    status="fail",
                    P=P,
                    steps=steps,
                    fails=fails,
                    nF=nF,
                    how="step size below 1e-10",
                    used=list(used),
                )
    return dict(status="fail", P=P, steps=steps, fails=fails, nF=np.inf, how="max steps", used=list(used))


def sig_value(X, kind, z, n, centre, o, k=None):
    if kind == "split0":
        return fl(o["D2c"])
    if kind == "birth0":
        return fl(o["D0c"] - 1)
    if kind == "split":
        return fl(o["D2"][k])
    if kind == "death":
        allv = [fl(t) for t in z[n : 2 * n]] + ([fl(z[2 * n])] if centre else [])
        return -allv[k]
    raise ValueError(kind)


def locate(X, kind, k, lo_pt, hi_pt, n, centre, con, flo, fhi, before=(), tol_par=1e-9):
    """Illinois regula falsi on the flagged signal of the OLD family along its path parameter (con; None = a) between
    lo_pt (signal < 0) and hi_pt (> 0), points (z, A). Returns a*, a-bracket, the old state at a*."""
    xlo, xhi = fl(gpar(*lo_pt, con)), fl(gpar(*hi_pt, con))
    side = 0
    base = list(before)[-1:]
    for it in range(60):
        if abs(xhi - xlo) < tol_par:
            break
        mid = xhi - fhi * (xhi - xlo) / (fhi - flo) if fhi != flo else 0.5 * (xlo + xhi)
        lo_, hi_ = min(xlo, xhi), max(xlo, xhi)
        mid = (
            min(max(mid, lo_ + 0.01 * (hi_ - lo_)), hi_ - 0.01 * (hi_ - lo_))
            if it < 45
            else 0.5 * (xlo + xhi)
        )
        pts = sorted(base + [lo_pt, hi_pt], key=lambda q: fl(gpar(*q, con)))
        sd, A0 = (
            predict(pts, con, mid)
            if len(pts) == 3
            else lagrange([(gpar(*q, con), q[0], q[1]) for q in pts], mid)
        )
        zm, Am, nm, om = solve_at(X, sd, A0, n, centre, con, mid, maxit=30)
        if not (nm < X.tol):
            if con is None:
                zm, Am, nm, om = X.homotopy(sd, n, centre, arb(mid))
            else:
                zm, Am, nm, om = X.homotopy(setpar(sd, con, mid), n, centre, A0, con=con)
        if not (nm < X.tol):
            out(f"      locate: old family did not converge at parameter {mid:.10f} (max|F| {nm:.1e})")
            break
        sm = sig_value(X, kind, zm, n, centre, om, k)
        if sm > 0:
            xhi, hi_pt, fhi = mid, (zm, Am), sm
            if side == 1:
                flo /= 2
            side = 1
        else:
            xlo, lo_pt, flo = mid, (zm, Am), sm
            if side == -1:
                fhi /= 2
            side = -1
        if abs(sm) < 1e3 * X.tol:
            xlo = xhi = mid
            break
    xs = 0.5 * (xlo + xhi)
    sd, A0 = (
        lagrange([(gpar(*q, con), q[0], q[1]) for q in (lo_pt, hi_pt)], xs)
        if xhi != xlo
        else (list(lo_pt[0]), lo_pt[1])
    )
    zs, As, ns, os_ = solve_at(X, sd, A0, n, centre, con, xs, maxit=30)
    return fl(As), fl(lo_pt[1]), fl(hi_pt[1]), zs, os_, ns


def young_path(X, kind, zst, n, centre, a_star, a_target, k=None, ratio=RATIO):
    """natural-parameter continuation of the new family from the old state zst at a_star to a_target."""
    p = zst[:n]
    v = zst[n : 2 * n]
    c = zst[2 * n] if centre else None
    if kind == "split0":  # centre atom -> pair +-s
        n2, c2, con, par0 = n + 1, False, [(0, 1.0)], 1e-3
        mk = lambda par: [arb(par)] + list(p) + [c / 2] + list(v)
    elif kind == "birth0":  # new centre atom of weight c
        n2, c2, con, par0 = n, True, [(2 * n, 1.0)], 1e-6
        mk = lambda par: list(p) + list(v) + [arb(par)]
    elif kind == "split":  # atom p_k -> pair p_k -+ h (half-gap h)
        n2, c2, con, par0 = n + 1, centre, [(k + 1, 0.5), (k, -0.5)], 1e-3
        mk = lambda par: (
            list(p[:k])
            + [p[k] - arb(par), p[k] + arb(par)]
            + list(p[k + 1 :])
            + list(v[:k])
            + [v[k] / 2, v[k] / 2]
            + list(v[k + 1 :])
            + ([c] if centre else [])
        )
    elif kind == "death":  # remove the atom whose weight crossed 0 (k = n: the centre)
        if k == n:
            z0 = list(p) + list(v)
            n2, c2 = n, False
        else:
            z0 = (
                [t for i, t in enumerate(p) if i != k]
                + [t for i, t in enumerate(v) if i != k]
                + ([c] if centre else [])
            )
            n2, c2 = n - 1, centre
        z, A, nF, o = X.solve(z0, n2, c2, arb(a_target))
        if not (nF < X.tol):
            z, A, nF, o = X.homotopy(z0, n2, c2, arb(a_target))
        if not (nF < X.tol):
            return dict(fail=True, notes=["death: direct solve failed"])
        return dict(
            z=z,
            n=n2,
            centre=c2,
            o=o,
            nF=nF,
            steps=0,
            par_end=None,
            tail=[],
            notes=["death: direct solve"],
            par_start=None,
            a_path_start=None,
        )
    else:
        raise ValueError(kind)
    notes = []
    for p0 in (par0, par0 / 10, par0 / 100):
        z, A, nF, o = X.solve(mk(p0), n2, c2, arb(a_star), con=con, maxit=40)
        if nF < TOLY:
            par0 = p0
            break
    if not (nF < TOLY):
        notes.append(f"path start failed (max|F| {nF:.1e})")
        return dict(fail=True, notes=notes)
    pts = [(par0, z, A)]
    lr = np.log(ratio)
    cur_lr = lr
    steps = 0
    lpc_used = None
    while steps < 600 and fl(pts[-1][2]) < a_target:
        par = pts[-1][0] * np.exp(cur_lr)
        if len(pts) >= 2:
            sd, A0 = lagrange(pts[-3:], par)
        else:
            pb, zb, Ab = pts[-1]
            e = 2 if kind in ("split0", "split") else 1
            A0 = (arb(a_star) + (Ab - arb(a_star)) * arb((par / pb) ** e)).mid()
            sd = list(zb)
        zt, At, nFt, ot = solve_at(X, sd, A0, n2, c2, con, par)
        how = "newton"
        if not (nFt < TOLY) and cur_lr <= lr / 16 + 1e-15:
            zt, At, nFt, ot = X.homotopy(setpar(sd, con, par), n2, c2, A0, con=con, tol=TOLY)
            how = "homotopy"
        if nFt < TOLY:
            pts.append((par, zt, At))
            steps += 1
            if how == "homotopy":
                notes.append(f"step to par {par:.4e} (a {fl(At):.8f}) by the a-free homotopy")
            cur_lr = min(cur_lr * 2, lr)
        else:
            if cur_lr <= lr / 16 + 1e-15:
                notes.append(
                    f"natural-parameter path failed at par {par:.4e} (a {fl(A0):.8f}), max|F| {nFt:.1e}; "
                    f"local-parameter continuation from the last {min(3, len(pts))} path points"
                )
                if len(pts) < 2:
                    return dict(fail=True, notes=notes)
                r = lpc(X, [(z_, A__) for _, z_, A__ in pts[-3:]], n2, c2, a_target)
                notes.append(
                    f"LPC: {r['status']} after {r['steps']} steps ({r['fails']} halvings), parameters used {r.get('used')}, landing {r.get('how')}"
                )
                if r["status"] != "ok":
                    return dict(fail=True, notes=notes)
                tail = [(fl(A__), z_) for z_, A__ in r["P"] if fl(A__) < a_target - 1e-6][-2:]
                return dict(
                    z=r["z"],
                    n=n2,
                    centre=c2,
                    o=r["o"],
                    nF=r["nF"],
                    steps=steps + r["steps"],
                    par_end=fl(gpar(r["z"], None, con)),
                    tail=tail,
                    notes=notes,
                    par_start=pts[0][0],
                    a_path_start=fl(pts[0][2]),
                )
            cur_lr /= 2
            notes.append(
                f"step to par {par:.4e} failed (max|F| {nFt:.1e}); ratio step halved to {np.exp(cur_lr):.5f}"
            )
    if fl(pts[-1][2]) < a_target:
        notes.append("600 steps without reaching the target")
        return dict(fail=True, notes=notes)
    P = [(z_, A__) for _, z_, A__ in pts[-3:]]
    zf, of, nFf, how = land_at(X, P, n2, c2, a_target, con)
    notes.append(f"landing on a = {a_target}: {how}, max|F| {nFf:.1e}")
    if zf is None:
        return dict(fail=True, notes=notes)
    tail = [(fl(A__), z_) for z_, A__ in P if fl(A__) < a_target - 1e-6][-2:]
    return dict(
        z=zf,
        n=n2,
        centre=c2,
        o=of,
        nF=nFf,
        steps=steps,
        par_end=fl(gpar(zf, None, con)),
        tail=tail,
        notes=notes,
        par_start=pts[0][0],
        a_path_start=fl(pts[0][2]),
    )


def report(X, z, n, centre, o, a, nF):
    T = time.time()
    k = X.kkt(z, n, centre, o)
    # quadrature check: q = 24 nodes per panel, re-polished, S at the same dip
    Xq = MPX(X.prec, q=24)
    zq, Aq, nFq, oq = Xq.solve(z, n, centre, arb(a), maxit=3, tol=Xq.tol * 1e-8)  # iterate to the floor
    dSq = np.inf
    dz = np.inf
    if nFq < Xq.tol:
        t0 = k["S_at"]
        if abs(t0) < 1e-12:
            ((d0, _, _),) = Xq.D_many(oq["st"], [arb(0)])
        else:
            _, d0, _, _ = Xq.refine_ext(oq["st"], t0, t0 - 0.2, t0 + 0.2)
        dSq = abs(fl(-(d0 - 1) - k["S_arb"]))
        dz = max(abs(fl(zq[i] - z[i])) for i in range(len(z)))
    X.M = Q.MPC(X.prec, X.q)  # restore ctx.prec (MPC sets it; same prec)
    noise_arb = max(nF, k["S_rad"])
    nu = max(noise_arb, dSq)
    resolved = bool(nF < X.tol and k["ok"] and k["S"] >= 1e3 * nu)
    row = dict(
        a=a,
        K=K_of(n, centre),
        n=n,
        centre=centre,
        prec=X.prec,
        maxF=nF,
        S=k["S"],
        S_at=k["S_at"],
        S_rad=k["S_rad"],
        dS_q24=dSq,
        dz_q24=dz,
        nu=nu,
        ratio=(k["S"] / nu if nu > 0 else np.inf),
        gmax=k["gmax"],
        lmax=k["lmax"],
        outer=k["outer"],
        minw=k["minw"],
        maxd2=k["maxd2"],
        d2c=k["d2c"],
        D0m1=k["D0m1"],
        kkt_ok=k["ok"],
        resolved=resolved,
        p=[fl(t) for t in z[:n]],
        v=[fl(t) for t in z[n : 2 * n]],
        c=(fl(z[2 * n]) if centre else None),
        z_str=zstr(z),
        dips=k["dips"][:4],
    )
    out(
        f"MP a = {a:6.2f}: K = {row['K']:3d}, prec {X.prec}, max|F| {nF:.1e}, S = {k['S']:.3e} at {k['S_at']:.3f} (rad {k['S_rad']:.0e}), "
        f"|dS q16->24| {dSq:.1e}, nu {nu:.1e}, S/nu {row['ratio']:.1e}, off-atom max(D-1) {k['gmax']:+.2e}, outer {k['outer']:+.1e}, "
        f"min w {k['minw']:.2e}, max D'' {k['maxd2']:+.2e}: {'KKT ok' if k['ok'] else 'KKT FAILS'}, "
        f"{'RESOLVED' if resolved else 'NOT RESOLVED'}  [{time.time() - T:.0f}s]"
    )
    return row, noise_arb


def save(rows, events, extra):
    json.dump(dict(rows=rows, events=events, **extra), open(OUT, "w"), indent=0)


# -------------------------------------------------------------------------------------------- run
def load_start(X):
    d = json.load(open(SRC))
    r = d["rows"][-1]
    z = [arb(t) for t in r["p_str"]] + [arb(t) for t in r["v"]] + ([arb(r["c"])] if r["centre"] else [])
    n, centre, a = r["n"], r["centre"], r["a"]
    z, A, nF, o = X.solve(z, n, centre, arb(a), maxit=40)
    return a, n, centre, z, o, nF


TYPES = {"split0": "split-centre", "birth0": "birth-centre", "split": "split-pair", "death": "death"}


def run(a_end, resume=False):
    T0 = time.time()
    el = lambda: f"[{time.time() - T0:.0f}s]"
    if resume and os.path.exists(OUT):
        d = json.load(open(OUT))
        rows, events = d["rows"], d["events"]
        extra = d.get("extra", {})
        r = [r_ for r_ in rows if not r_.get("skipped")][-1]
        st = d.get("state") or dict(
            a=r["a"], n=r["n"], centre=r["centre"], prec=r["prec"], z=r["z_str"], hist=[]
        )
        a, n, centre = st["a"], st["n"], st["centre"]
        X = MPX(st["prec"])
        z = [arb(t) for t in st["z"]]
        z, A, nF, o = X.solve(z, n, centre, arb(a), maxit=40)
        hist = [(ha, [arb(t) for t in hz]) for ha, hz in st.get("hist", [])][-2:] + [(a, z)]
        rows = [r_ for r_ in rows if r_["a"] <= a + 1e-9]
        events = [e for e in events if e["a_star"] <= a + 1e-9]
        out(
            f"=== q498 RESUME from a = {a} (K = {K_of(n, centre)}), prec {X.prec}, max|F| {nF:.1e}, to {a_end}  ({time.ctime()}) ==="
        )
    else:
        X = MPX(prec_for(33.5))
        a, n, centre, z, o, nF = load_start(X)
        out(
            f"=== q498 arb continuation from the q496 arb row a = {a} (K = {K_of(n, centre)}), prec {X.prec}, "
            f"polished max|F| {nF:.1e}, to {a_end}  ({time.ctime()}) ==="
        )
        rows, events, extra = [], [], {}
        row, _ = report(X, z, n, centre, o, a, nF)
        rows.append(row)
        hist = [(a, z)]

    def state():
        return dict(
            a=a,
            n=n,
            centre=centre,
            prec=X.prec,
            z=zstr(z),
            hist=[(ha, zstr(hz)) for ha, hz in hist[:-1]][-2:],
        )

    save(rows, events, dict(extra=extra, state=state()))
    da = DA
    prefer_lpc = False
    while a < a_end - 1e-9:
        pr = prec_for(a + 1e-9)  # precision schedule
        if pr > X.prec:
            X.set_prec(pr)
            z, A, nF, o = X.solve(z, n, centre, arb(a), maxit=40)
            out(f"   precision raised to {pr} bits at a = {a}: re-polished max|F| {nF:.1e}")
            hist = [(ha, X.solve(hz, n, centre, arb(ha), maxit=40)[0]) for ha, hz in hist[:-1]] + [(a, z)]
        a1 = round(min(a + da, np.floor(a / DA + 1e-6) * DA + DA), 10)
        t_step = time.time()
        z1 = o1 = None
        nF1 = np.inf
        how = "failed"
        if not (
            prefer_lpc and len(hist) >= 2
        ):  # v2b: LPC first while the previous step needed a non-a parameter
            z1, o1, nF1, how = advance(X, hist, n, centre, a1)
        bracket = None
        lpcP = None
        lpc_non_a = False
        if z1 is None and len(hist) >= 2:  # local-parameter continuation
            r = lpc(X, [(hz, arb(ha)) for ha, hz in hist][-3:], n, centre, a1)
            if r["status"] == "ok":
                z1, o1, nF1 = r["z"], r["o"], r["nF"]
                lpcP = r["P"]
                lpc_non_a = any(u_ != "a" for u_ in r["used"])
                how = f"LPC, {r['steps']} steps, {r['fails']} halvings, parameters {r['used']}, landing: {r['how']}"
            elif r["status"] == "flag":
                bracket = (r["P"][-2], r["P"][-1], r["con"], r["o"], r["flags"])
                how = f"LPC, {r['steps']} steps, flag {list(r['flags'])}"
            else:
                out(
                    f"      step {a} -> {a1}: LPC failed ({r['how']}, {r['steps']} steps, {r['fails']} halvings)"
                )
        if z1 is None and bracket is None and prefer_lpc:
            z1, o1, nF1, how = advance(X, hist, n, centre, a1)
        prefer_lpc = lpc_non_a
        if z1 is None and bracket is None:
            z1, o1, nF1, how = advance(X, hist, n, centre, a1, homotopy=True)
        if z1 is None and bracket is None:
            da /= 2
            out(f"   step {a} -> {a1} failed (max|F| {nF1:.1e}); da -> {da}  [{time.time() - t_step:.0f}s]")
            if da < 1e-4:
                out(f"   STOP: no convergence of the K = {K_of(n, centre)} family beyond a = {a} {el()}")
                break
            continue
        if how != "quadratic":
            out(
                f"      step {a} -> {a1}: {how}, max|F| {nF1 if z1 is not None else np.nan:.1e}  [{time.time() - t_step:.0f}s]"
            )
        if bracket is None:
            fl1 = X.flags(z1, n, centre, o1)
            if fl1:
                bracket = ((z, arb(a)), (z1, arb(a1)), None, o1, fl1)
        if bracket is None:
            if lpcP is not None:
                hist = [(fl(A__), z_) for z_, A__ in lpcP if fl(A__) < a1 - 1e-9][-2:] + [(a1, z1)]
            else:
                hist = (hist + [(a1, z1)])[-3:]
            a, z, o, nF = a1, z1, o1, nF1
            da = DA if abs(a / DA - round(a / DA)) < 1e-9 else da
        else:
            # ---------------- event ----------------
            Te = time.time()
            Kb = K_of(n, centre)
            lo_pt, hi_pt, con, o_hi, flg = bracket
            o_lo = X.evalF(lo_pt[0], n, centre, lo_pt[1])
            before = (
                [(hz, arb(ha)) for ha, hz in hist if ha < fl(lo_pt[1]) - 1e-9][-1:] if con is None else []
            )
            cands = []
            for kind in flg:
                kk = flg[kind] if kind in ("split", "death") else None
                flo = sig_value(X, kind, lo_pt[0], n, centre, o_lo, kk)
                fhi = sig_value(X, kind, hi_pt[0], n, centre, o_hi, kk)
                a_st, alo, ahi, zst, ost, nst = locate(
                    X, kind, kk, lo_pt, hi_pt, n, centre, con, flo, fhi, before=before
                )
                cands.append((a_st, kind, kk, alo, ahi, zst, ost, nst))
                out(
                    f"   flag {kind}{'' if kk is None else f'[{kk}]'}: old-family signal {flo:+.3e} at a = {fl(lo_pt[1]):.6f} -> {fhi:+.3e} "
                    f"at {fl(hi_pt[1]):.6f}; crossing at a* = {a_st:.10f} (bracket in a [{alo:.10f}, {ahi:.10f}], "
                    f"path parameter {'a' if con is None else con}, old-family max|F| {nst:.1e})"
                )
            cands.sort(key=lambda t: t[0])
            a_st, kind, kk, alo, ahi, zst, ost, nst = cands[0]
            a_t = round(np.ceil((a_st + 0.01) / DA - 1e-9) * DA, 10)
            if a_t > round(np.ceil(a_st / DA - 1e-9) * DA, 10) + 1e-9:
                out(
                    f"      change {round(np.ceil(a_st/DA - 1e-9)*DA, 10) - a_st:.1e} below the step point "
                    f"{round(np.ceil(a_st/DA - 1e-9)*DA, 10)}: the new family is started at {a_t}"
                )
            yp = young_path(X, kind, zst, n, centre, a_st, a_t, kk)
            for nt in yp.get("notes", []):
                out(f"      young path: {nt}")
            typ = TYPES[kind]
            ok_ev = not yp.get("fail") and yp["nF"] < X.tol
            if not ok_ev:
                ev = dict(
                    type=typ,
                    a_star=a_st,
                    a_lo=alo,
                    a_hi=ahi,
                    K_before=Kb,
                    K_after=None,
                    failed=True,
                    notes=yp.get("notes", []),
                )
                events.append(ev)
                save(rows, events, dict(extra=extra, state=state()))
                out(
                    f"   STOP: the new family after the {typ} at a* = {a_st:.8f} could not be followed to a = {a_t} {el()}"
                )
                break
            zn_, nn_, cn_, on_ = yp["z"], yp["n"], yp["centre"], yp["o"]
            fl2 = X.flags(zn_, nn_, cn_, on_)
            ev = dict(
                type=typ,
                a_star=a_st,
                a_lo=alo,
                a_hi=ahi,
                K_before=Kb,
                K_after=K_of(nn_, cn_),
                a_target=a_t,
                path_steps=yp["steps"],
                par_start=yp.get("par_start"),
                a_path_start=yp.get("a_path_start"),
                par_end=yp["par_end"],
                maxF=yp["nF"],
                flags_after=list(fl2.keys()),
                runtime_s=time.time() - Te,
                other_flags=[(c_[1], c_[0]) for c_ in cands[1:]],
                notes=yp["notes"],
                prec=X.prec,
                d2c_old_at_astar=(fl(ost["D2c"]) if centre else None),
                D0m1_old_at_astar=(fl(ost["D0c"] - 1) if not centre else None),
            )
            events.append(ev)
            out(
                f"   EVENT {typ:13s} at a* = {a_st:.8f}: K {Kb} -> {K_of(nn_, cn_)}; new family at a = {a_t} after {yp['steps']} "
                f"path steps (parameter {yp.get('par_start')} at a = {yp.get('a_path_start')} -> {yp['par_end']:.6f}), "
                f"max|F| {yp['nF']:.1e}, flags after {list(fl2.keys())}, event time {time.time() - Te:.0f}s  {el()}"
            )
            if fl2:
                out(f"   STOP: the new structure is still flagged at a = {a_t}: {fl2}")
                save(rows, events, dict(extra=extra, state=state()))
                break
            a, z, n, centre, o, nF = a_t, zn_, nn_, cn_, on_, yp["nF"]
            hist = yp["tail"][-2:] + [(a, z)]
            da = DA
            prefer_lpc = True  # v2b: the young family moves fast in a
            for g in np.arange(np.ceil((a_st - 1e-9) / 0.25) * 0.25, a - 1e-9, 0.25):
                out(f"      grid point {g:.2f} lies between a* and the start of the new family: no row")
                rows.append(dict(a=float(g), K=None, skipped=True, note=f"within the {typ} at {a_st:.6f}"))
        if abs(a / 0.25 - round(a / 0.25)) < 1e-9:
            row, noise_arb = report(X, z, n, centre, o, a, nF)
            if row["S"] > 0 and noise_arb > 1e-9 * row["S"]:
                X.set_prec(X.prec + 32)
                z, A, nF, o = X.solve(z, n, centre, arb(a), maxit=40)
                out(
                    f"   arb noise {noise_arb:.1e} > 1e-9 S: precision raised to {X.prec}; re-polished max|F| {nF:.1e}"
                )
                row, noise_arb = report(X, z, n, centre, o, a, nF)
                hist = [(ha, X.solve(hz, n, centre, arb(ha), maxit=40)[0]) for ha, hz in hist[:-1]] + [(a, z)]
            rows.append(row)
            if not row["kkt_ok"]:
                out(f"   STOP: the full KKT scan fails at a = {a} without a flagged event")
                save(rows, events, dict(extra=extra, state=state()))
                break
        save(rows, events, dict(extra=extra, state=state()))
    out(f"=== end at a = {a}, K = {K_of(n, centre)} {el()} ===")
    return rows, events


# -------------------------------------------------------------------------------------------- diagnostics
def oldfamily():
    """the K = 45 family continued past the centre split: where D'' changes sign at 0, p1, p2, p3 (regula falsi)"""
    X = MPX(128)
    a, n, centre, z, o, nF = load_start(X)
    out(
        f"=== oldfamily: the K = {K_of(n, centre)} family from a = {a} continued past the centre split (prec {X.prec}) ==="
    )
    hist = [(a, z)]
    prev = (a, z, o)
    names = ["0"] + [f"p{k + 1}" for k in range(n)]
    sig = lambda o_: [fl(o_["D2c"])] + [fl(t) for t in o_["D2"]]
    crossings = {}
    for a1 in [round(33.5 + 0.01 * k, 6) for k in range(1, 31)]:
        z1, o1, nF1, how = advance(X, hist, n, centre, a1)
        if z1 is None:
            out(f"   old family failed at {a1}")
            break
        s0, s1 = sig(prev[2]), sig(o1)
        for j in range(len(s1)):
            if s0[j] < 0 <= s1[j] and names[j] not in crossings:
                kind, kk = ("split0", None) if j == 0 else ("split", j - 1)
                a_st, lo, hi, zst, ost, nst = locate(
                    X,
                    kind,
                    kk,
                    (prev[1], arb(prev[0])),
                    (z1, arb(a1)),
                    n,
                    centre,
                    None,
                    s0[j],
                    s1[j],
                    before=[(hz, arb(ha)) for ha, hz in hist if ha < prev[0] - 1e-9][-1:],
                )
                crossings[names[j]] = a_st
                out(f"   D''({names[j]}) changes sign at a = {a_st:.8f}")
        s = sig(o1)
        out(
            f"   a = {a1:.2f}: max|F| {nF1:.1e}, c {fl(z1[2*n]):.5e}, D''(0) {s[0]:+.3e}, D''(p1..p4) {', '.join(f'{x:+.3e}' for x in s[1:5])}"
        )
        hist = (hist + [(a1, z1)])[-3:]
        prev = (a1, z1, o1)
    return crossings


def control():
    """(C1) dF/da against central differences (h = 1e-20) in arb at a = 33.5; (C2) report rows at 33.25 and 33.5 from the
    q496 arb states agree with the q496 values (K, S to 3 digits)"""
    X = MPX(128)
    d = json.load(open(SRC))
    ok_all = True
    for r in d["rows"][-2:]:
        z = [arb(t) for t in r["p_str"]] + [arb(t) for t in r["v"]] + ([arb(r["c"])] if r["centre"] else [])
        n, centre, a = r["n"], r["centre"], r["a"]
        z, A, nF, o = X.solve(z, n, centre, arb(a), maxit=40)
        row, _ = report(X, z, n, centre, o, a, nF)
        ok = row["K"] == r["K"] and abs(row["S"] - r["S"]) < 1e-3 * r["S"] and row["kkt_ok"] == r["ok"]
        out(
            f"(C2) a = {a}: K {row['K']} vs q496 {r['K']}, S {row['S']:.4e} vs q496 {r['S']:.4e}: {'PASS' if ok else 'FAIL'}"
        )
        ok_all &= ok
        if a == 33.5:
            col = X.dFda(o, n, centre)
            h = arb("1e-20")
            Fp = X.evalF(z, n, centre, arb(a) + h, jac=False)["F"]
            Fm = X.evalF(z, n, centre, arb(a) - h, jac=False)["F"]
            worst = max(abs(fl((Fp[i] - Fm[i]) / (2 * h) - col[i])) for i in range(len(col)))
            big = max(abs(fl(t)) for t in col)
            ok = worst / big < 1e-12
            out(
                f"(C1) a = 33.5: dF/da vs central differences: max |diff|/max |col| = {worst/big:.1e}: {'PASS' if ok else 'FAIL'}"
            )
            ok_all &= ok
    out(f"CONTROLS: {'PASS' if ok_all else 'FAIL'}")


def summary():
    """table of the change points (q496 arb events from a* >= 29.5, then q498) and of the report rows"""
    d = json.load(open(OUT))
    dq = json.load(open(SRC))
    out("=== summary: change points (q496 mpsweep up to 33.5, q498 beyond) ===")
    out(
        "   source   a*              type           K before -> after   spacing   path steps   event time [s]   prec"
    )
    prev = None
    evs = [
        ("q496", e["a_star"], e["type"], e["K_before"], e["K_after"], e.get("path_len"), None, dq.get("prec"))
        for e in dq["events"]
    ]
    evs += [
        (
            "q498",
            e["a_star"],
            e["type"],
            e["K_before"],
            e["K_after"],
            e.get("path_steps"),
            e.get("runtime_s"),
            e.get("prec"),
        )
        for e in d["events"]
    ]
    for src, a_, t_, kb, ka, ps, rt, pr in evs:
        out(
            f"   {src}   {a_:.8f}   {t_:13s}   {kb:3d} -> {ka if ka is not None else 'FAILED'}   "
            f"{'' if prev is None else f'{a_ - prev:.5f}':>9s}   {ps if ps is not None else '':>10}   {'' if rt is None else f'{rt:.0f}':>14s}   {pr}"
        )
        prev = a_
    rows = [r for r in d["rows"] if not r.get("skipped")]
    nres = sum(1 for r in rows if r["resolved"])
    out(
        f"   report rows: {len(rows)} (every 0.25 from {rows[0]['a']} to {rows[-1]['a']}), resolved {nres}; "
        f"skipped grid points {[r['a'] for r in d['rows'] if r.get('skipped')]}"
    )
    out(
        f"   min S/nu over the rows {min(r['ratio'] for r in rows):.1e}; max max|F| {max(r['maxF'] for r in rows):.1e}; "
        f"min S {min(r['S'] for r in rows):.2e} at a = {min(rows, key=lambda r: r['S'])['a']}"
    )


def compare():
    """AFTER the run: the frozen predictions (npmle_forward_predictions.json, read only) against the located changes;
    rule of PREREG_Q32 amendment 642(5): LAW3 PREFERRED if its max |error| is below half of each alternative's; LAW3
    FAILS if its max |error| exceeds 0.1. Error = predicted minus located a*."""
    pf = os.path.join(HERE, "npmle_forward_predictions.json")
    P_ = json.load(open(pf))
    d = json.load(open(OUT))
    loc = {e["K_after"]: e["a_star"] for e in d["events"] if e.get("K_after") is not None}
    models = ("LAW3", "ALOG", "A2")
    out(
        f"=== compare: frozen predictions (written {P_['written']}; file mtime {time.ctime(os.path.getmtime(pf))}) vs the q498 change points ==="
    )
    out("   K_after   located a*      LAW3 (err)            ALOG (err)            A2 (err)")
    errs = {m: [] for m in models}
    rowsc = []
    for pr in P_["predictions"]:
        K = pr["K_after"]
        if K not in loc:
            out(
                f"   {K:4d}      not located (beyond the run)   predictions {pr['LAW3']:.4f} / {pr['ALOG']:.4f} / {pr['A2']:.4f}"
            )
            continue
        e = {m: pr[m] - loc[K] for m in models}
        for m in models:
            errs[m].append(e[m])
        rowsc.append(
            dict(K_after=K, a_star=loc[K], **{m: pr[m] for m in models}, **{m + "_err": e[m] for m in models})
        )
        out(f"   {K:4d}      {loc[K]:.8f}     " + "   ".join(f"{pr[m]:.4f} ({e[m]:+.4f})" for m in models))
    mx = {m: max(abs(x) for x in errs[m]) for m in models}
    n = len(errs["LAW3"])
    pref = all(mx["LAW3"] < 0.5 * mx[m] for m in ("ALOG", "A2"))
    fails = mx["LAW3"] > 0.1
    out(
        f"   {n} changes with K_after in 47..70 located (K_after {min(r['K_after'] for r in rowsc)}..{max(r['K_after'] for r in rowsc)}); "
        f"max |error|: LAW3 {mx['LAW3']:.4f}, ALOG {mx['ALOG']:.4f}, A2 {mx['A2']:.4f}"
    )
    out(
        f"   rule 642(5): LAW3 max |error| {mx['LAW3']:.4f} < half of ALOG ({0.5*mx['ALOG']:.4f}) and of A2 ({0.5*mx['A2']:.4f}): "
        f"{'yes' if pref else 'no'}; LAW3 max |error| > 0.1: {'yes' if fails else 'no'}  ->  "
        f"{'LAW3 FAILS' if fails else ('LAW3 PREFERRED' if pref else 'neither PREFERRED nor FAILS')}"
    )
    d["compare"] = dict(
        rows=rowsc,
        max_abs_err=mx,
        n_located=n,
        preferred=bool(pref),
        fails=bool(fails),
        predictions_written=P_["written"],
    )
    json.dump(d, open(OUT, "w"), indent=0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "summary":  # prints only (the log gets the summary at the end of the session)
        summary()
        sys.exit(0)
    with open(LOGF, "a", encoding="utf-8") as log:
        _log[0] = log
        if mode == "oldfamily":
            oldfamily()
        elif mode == "control":
            control()
        elif mode == "compare":
            compare()
        elif mode == "summarylog":
            summary()
        elif mode == "run":
            run(
                float(sys.argv[2]) if len(sys.argv) > 2 else 45.0,
                resume=(len(sys.argv) > 3 and sys.argv[3] == "resume"),
            )
