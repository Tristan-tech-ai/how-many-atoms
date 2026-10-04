"""D_solver (02 Oct 2026): a solver for the least-favourable-prior ring on the ellipse past the flat-KKT wall.
The KKT system is q786's (copied as text in D_core.LFP: unknowns v = [G angles] + [orbit masses] + [C]; equations r - C at one atom per
orbit, tangential derivative at each G atom, masses sum to 1), at PREC bits with NGH Gauss-Hermite nodes.
Why q786 stalls (D_diag.py): the Jacobian is graded, one singular value per ring harmonic (twelve-ring at eps 0.1723: 6.6 ... 1.3e-6,
8.7e-10), and the softest directions are the ring's high harmonics. In angle/mass coordinates the residual has O(1) curvature along
those directions (the harmonics are nonlinear in the angles), so a Newton step of size d along them is accepted only if d < ~sigma/kappa.
Fix (option ii of the task): Newton in ring-harmonic coordinates. The linear Newton step dv is computed as usual; it is then carried by
the moment map b_j = 2 sum_k m_k cos(j t_k) (j = 2, 4, ..., 2(n_shape)), total mass 1: the new ring is the one whose harmonics are
b(v) + db (db = (db/dv) dv), found by an inner Newton solve of the moment equations (well conditioned), and C moves linearly. The
curvature that blocked the plain step lives in the moment map, so the outer problem is close to linear in these coordinates.
Continuation in rm with a predictor in the same coordinates (quadratic extrapolation of b and C), and the tracker's KKT test: max of
D - C on 91 points of [0, 90] deg above TOLV 1e-22 = KKT lost, bisect in rm to BTOL.
Usage:
  py D_solver.py newton STATE_A STATE_B RM [MODE plain|harm] [MAXIT]   (predictor: tangent at A, or secant A-B if STATE_B given)
  py D_solver.py track OUTDIR RM_END STEP STATE_1,STATE_2[,STATE_3]       (states oldest first; continues from the last)
Env: PREC (160), NGH (200), RP (1.0), FDH (1e-12), TOLV (1e-22), TOLR (1e-27), BTOL (1e-6), NSCAN (90), FUNC (lfp | cap | rd),
LAM (0 = ellipse; mixed curve of q805 otherwise), SRC_S (rd source width). KKT functions from D_core2 (text copies of q786, q719/q781, q789).
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
from flint import arb, arb_mat, ctx
from D_core2 import Ring  # 02 Oct 18:55: general KKT functions (FUNC lfp | cap | rd, env LAM)

PREC = int(os.environ.get("PREC", "160"))
NGH = int(os.environ.get("NGH", "200"))
RP = os.environ.get("RP", "1.0")
FDH = os.environ.get("FDH", "1e-12")
TOLV = float(os.environ.get("TOLV", "1e-22"))
TOLR = float(os.environ.get("TOLR", "1e-27"))
BTOL = float(os.environ.get("BTOL", "1e-6"))
NSCAN = int(os.environ.get("NSCAN", "90"))
FUNC = os.environ.get("FUNC", "lfp")
LAM = os.environ.get("LAM", "0")
SRC_S = os.environ.get("SRC_S", "1")
REUSEJ = os.environ.get("REUSEJ", "0") == "1"
CHORD = os.environ.get("CHORD", "0") == "1"
CHORDF = float(os.environ.get("CHORDF", "0.01"))
NPROC = int(os.environ.get("NPROC", "1"))
DIG = int(PREC * 0.30103) + 5  # digits kept in json and between processes
T0 = time.time()
say = lambda *a: print(*a, f"[{time.time() - T0:.0f}s]", flush=True)
f2 = lambda x: float(x.mid()) if isinstance(x, arb) else float(x)


class Solver:
    def __init__(self, X, Y, nG):
        self.P = Ring(FUNC, RP, X, Y, nG, NGH=NGH, prec=PREC, lam=LAM, src_s=SRC_S)
        P = self.P
        self.nG = nG
        self.nO = P.nO
        self.n = nG + P.nO + 1
        self.ns = nG + P.nO  # shape unknowns (angles + masses)
        self.js = [2 * k for k in range(1, self.ns)]  # harmonics fixed by the shape unknowns
        self.h = arb(FDH)

    # ---- moment map: harmonics b_j (j in js) and total mass, as functions of the shape unknowns
    def orbit_terms(self, v):
        P = self.P
        G = v[: self.nG]
        m = v[self.nG : self.ns]
        k = 0
        terms = []  # (mass, multiplicity, angle) per orbit
        if P.X:
            terms.append((m[k], 2, None, "X"))
            k += 1
        if P.Y:
            terms.append((m[k], 2, None, "Y"))
            k += 1
        for i, a in enumerate(G):
            terms.append((m[k + i], 4, a, "G"))
        return terms

    def moments(self, v):
        out = []
        for j in self.js:
            s = arb(0)
            for mm, mult, a, kind in self.orbit_terms(v):
                if kind == "X":
                    s += 2 * mult * mm
                elif kind == "Y":
                    s += 2 * mult * mm * (1 if (j // 2) % 2 == 0 else -1)
                else:
                    s += 2 * mult * mm * (j * a).cos()
            out.append(s)
        out.append(sum((mult * mm for (mm, mult, a, kind) in self.orbit_terms(v)), arb(0)))
        return out

    def moments_jac(self, v):
        ns = self.ns
        nG = self.nG
        Mj = arb_mat(ns, ns)
        G = v[:nG]
        m = v[nG:ns]
        P = self.P
        kX = 0 if P.X else None
        kY = (1 if P.X else 0) if P.Y else None
        for r, j in enumerate(self.js + [0]):
            for i in range(nG):  # angle columns
                mi = m[(int(P.X) + int(P.Y)) + i]
                Mj[r, i] = (-8 * mi * j * (j * G[i]).sin()) if j else arb(0)
            for k in range(self.nO):
                col = nG + k
                if j == 0:
                    Mj[r, col] = arb(2) if (k == kX or k == kY) else arb(4)
                elif k == kX:
                    Mj[r, col] = arb(4)
                elif k == kY:
                    Mj[r, col] = arb(4 * (1 if (j // 2) % 2 == 0 else -1))
                else:
                    Mj[r, col] = 8 * (j * G[k - int(P.X) - int(P.Y)]).cos()
        return Mj

    def moment_solve(self, target, v0, tol=None):
        """shape unknowns whose moments equal target (list of ns arbs), Newton from v0 (n-vector; C copied)."""
        tol = tol or 2.0 ** (-PREC + 12)
        v = [x for x in v0]
        for it in range(40):
            mo = self.moments(v)
            r = [a - b for a, b in zip(mo, target)]
            nr = max(abs(f2(x)) for x in r)
            if nr < tol:
                return v, nr
            try:
                d = self.moments_jac(v).solve(arb_mat(self.ns, 1, [-(x.mid()) for x in r]))
            except ZeroDivisionError:
                return v, float("inf")  # 21:18: degenerate ring; the caller damps the step
            v = [(v[i] + d[i, 0]).mid() for i in range(self.ns)] + v[self.ns :]
        return v, nr

    # ---- KKT residual and Jacobian
    def F(self, v, rm):
        return self.P.resid(v, rm)

    def jac(self, v, rm, with_rm=False):
        n = self.n
        h = self.h
        J = arb_mat(n, n)
        for i in range(n):
            J[i, n - 1] = arb(-1) if i < self.nO else arb(0)  # dF/dC exactly (F = D - C)
        if NPROC > 1:
            pool = get_pool(self)
            vs = [x.mid().str(DIG, radius=False) for x in v]
            rs = rm.mid().str(DIG, radius=False) if isinstance(rm, arb) else str(rm)
            cols = pool.map(_wcol, [(j, vs, rs) for j in range(n - 1)] + ([(-1, vs, rs)] if with_rm else []))
            for j in range(n - 1):
                for i in range(n):
                    J[i, j] = arb(cols[j][i])
            if with_rm:
                return J, arb_mat(n, 1, [arb(x) for x in cols[-1]])
            return J
        for j in range(n - 1):
            vp = list(v)
            vp[j] = vp[j] + h
            vm = list(v)
            vm[j] = vm[j] - h
            a = self.F(vp, rm)
            b = self.F(vm, rm)
            for i in range(n):
                J[i, j] = ((a[i] - b[i]) / (2 * h)).mid()
        if not with_rm:
            return J
        a = self.F(v, rm + h)
        b = self.F(v, rm - h)
        return J, arb_mat(n, 1, [((x - y) / (2 * h)).mid() for x, y in zip(a, b)])

    def step_harm(self, v, dv, lam):
        """carry the linear step lam*dv through the moment map: new ring with moments b(v) + lam*(db/dv) dv, C linear."""
        ns = self.ns
        Mj = self.moments_jac(v)
        mo = self.moments(v)
        db = Mj * arb_mat(ns, 1, [dv[i] for i in range(ns)])
        target = [mo[i] + lam * db[i, 0] for i in range(ns - 1)] + [arb(1)]  # total mass target exactly 1
        guess = [(v[i] + lam * dv[i]).mid() for i in range(self.n)]
        vn, nr = self.moment_solve(target, guess)
        vn = vn[:ns] + [(v[ns] + lam * dv[ns]).mid()]
        return vn, nr

    def newton(self, v, rm, mode="harm", maxit=30, tol=None, verbose=True, J=None, diag=False):
        if CHORD:
            return self.newton_chord(v, rm, maxit=maxit, tol=tol, verbose=verbose, J=J)
        tol = tol or TOLR
        n = self.n
        hist = []
        for it in range(maxit):
            r = self.F(v, rm)
            nr = max(abs(f2(x)) for x in r)
            n2 = sum(f2(x) ** 2 for x in r)
            hist.append(nr)
            if verbose:
                say(f"  it {it}: residual {nr:.3e}")
            if nr < tol:
                return v, nr, True, hist
            if J is None or it > 0:
                J = self.jac(v, rm)
            dvm = J.solve(arb_mat(n, 1, [-(x.mid()) for x in r]))
            dv = [dvm[i, 0] for i in range(n)]
            if diag:
                mp.mp.dps = int(PREC * 0.30103) - 2
                Mm = mp.matrix([[mp.mpf(J[i, j].str(55, radius=False)) for j in range(n)] for i in range(n)])
                U, S, V = mp.svd_r(Mm)
                rv = mp.matrix([mp.mpf(x.mid().str(55, radius=False)) for x in r])
                dm = mp.matrix([mp.mpf(x.str(55, radius=False)) for x in dv])
                say("    sigma:", [mp.nstr(s, 3) for s in S])
                say("    u_k^T F:", [mp.nstr((U[:, k].T * rv)[0], 3) for k in range(n)])
                say("    step in v_k:", [mp.nstr((V[k, :] * dm)[0], 3) for k in range(n)])
            lam = arb(1)
            best = None
            for _ in range(20):
                if mode == "harm":
                    vn, mr = self.step_harm(v, dv, lam)
                else:
                    vn = [(v[j] + lam * dv[j]).mid() for j in range(n)]
                rn = self.F(vn, rm)
                nn = sum(f2(x) ** 2 for x in rn)
                if diag:
                    say(f"    lam {f2(lam):.3e}: residual after step {max(abs(f2(x)) for x in rn):.3e}")
                if nn < n2:
                    best = vn
                    break
                lam = lam / 2
            if best is None:
                if verbose:
                    say("  no decrease along the step; stop")
                    return v, nr, False, hist
            v = best
        r = self.F(v, rm)
        nr = max(abs(f2(x)) for x in r)
        return v, nr, nr < tol, hist + [nr]

    def newton_chord(self, v, rm, maxit=30, tol=None, verbose=True, J=None):
        """harmonic Newton that keeps J while each step lowers the sum of squares by more than CHORDF (env), and recomputes it
        otherwise; the residual of the accepted trial is reused. Writes the iterate to env CKPT (json) after each accepted step.
        """
        tol = tol or TOLR
        n = self.n
        hist = []
        r = self.F(v, rm)
        fresh = False
        last_fresh = False
        if J is None and REUSEJ:
            J = getattr(self, "_lastJ", None)  # 20:2x: the J of the last solve (continuation)
        for it in range(maxit):
            nr = max(abs(f2(x)) for x in r)
            n2 = sum(f2(x) ** 2 for x in r)
            hist.append(nr)
            if verbose:
                say(
                    f"  it {it}: residual {nr:.3e}"
                    + (
                        ""
                        if it == 0
                        else (" (after a step with a fresh J)" if last_fresh else " (chord step)")
                    )
                )
            if nr < tol:
                if J is not None:
                    self._lastJ = J
                return v, nr, True, hist
            if J is None:
                J = self.jac(v, rm)
                fresh = True
            dvm = J.solve(arb_mat(n, 1, [-(x.mid()) for x in r]))
            dv = [dvm[i, 0] for i in range(n)]
            lam = arb(1)
            best = None
            for _ in range(12):
                vn, mr = self.step_harm(v, dv, lam)
                if not mr < 1e-20:
                    lam = lam / 2
                    continue  # moment solve failed: damp
                rn = self.F(vn, rm)
                nn = sum(f2(x) ** 2 for x in rn)
                if nn < n2:
                    best = (vn, rn, nn)
                    break
                lam = lam / 2
                if not fresh:
                    break  # an old J: recompute before damping
            if best is None:
                if fresh:
                    if verbose:
                        say("  no decrease along the step; stop")
                    return v, nr, False, hist
                J = None
                fresh = False
                continue
            v, r, nn = best
            if os.environ.get("CKPT"):
                save(
                    self, v, rm, os.environ["CKPT"], {"residual": max(abs(f2(x)) for x in r), "iteration": it}
                )
            if nn > CHORDF * n2:
                J = None  # weak reduction: fresh J next time
            last_fresh = fresh
            fresh = False
        nr = max(abs(f2(x)) for x in r)
        return v, nr, nr < tol, hist + [nr]

    def scan(self, v, rm, nscan=None):
        marg, best = self.P.scan(v, rm, nscan or NSCAN)
        return marg, best


_POOL = []
_W = {}


def get_pool(S):
    if not _POOL:
        import multiprocessing as mpr

        _POOL.append(
            mpr.get_context("spawn").Pool(NPROC, initializer=_winit, initargs=(int(S.P.X), int(S.P.Y), S.nG))
        )
    return _POOL[0]


def _winit(X, Y, nG):
    _W["S"] = Solver(X, Y, nG)


def _wcol(args):
    """one finite-difference column (j = -1: the rm column) in a worker; values as strings at DIG digits."""
    j, vs, rs = args
    S = _W["S"]
    ctx.prec = PREC
    v = [arb(x) for x in vs]
    rm = arb(rs)
    h = S.h
    if j < 0:
        a = S.F(v, rm + h)
        b = S.F(v, rm - h)
    else:
        vp = list(v)
        vp[j] = vp[j] + h
        vm = list(v)
        vm[j] = vm[j] - h
        a = S.F(vp, rm)
        b = S.F(vm, rm)
    return [((x - y) / (2 * h)).mid().str(DIG, radius=False) for x, y in zip(a, b)]


def load(f):
    d = json.load(open(f))
    ctx.prec = PREC
    return (
        d,
        [arb(g) for g in d["G"]] + [arb(x) for x in d["m"]] + [arb(d["C"]) if "C" in d else arb(0)],
        d["rm"],
    )


def c_from_atoms(S, v, rm):
    """q786's start value of C: the mass-weighted mean of D over the atoms (masses normalised to sum 1)."""
    P = S.P
    G = v[: S.nG]
    m = v[S.nG : S.ns]
    th, w = P.atoms(G, m)
    s = sum(w, arb(0))
    m = [x / s for x in m]
    th, w = P.atoms(G, m)
    xs = [P.pos(t, rm) for t in th]
    mult = ([2] if P.X else []) + ([2] if P.Y else []) + [4] * S.nG  # orbit representatives (D2)
    C = sum(
        (mu * mm * P.D_grad(P.pos(t, rm), xs, w, grad=False) for mu, mm, t in zip(mult, m, P.reps(G))), arb(0)
    )
    return list(G) + m + [C]


def save(S, v, rm, path, extra=None):
    nG, ns = S.nG, S.ns
    P = S.P
    out = {
        "rm": rm if isinstance(rm, str) else rm.str(40, radius=False),
        "X": int(P.X),
        "Y": int(P.Y),
        "G": [x.str(max(45, DIG), radius=False) for x in v[:nG]],
        "m": [x.str(max(45, DIG), radius=False) for x in v[nG:ns]],
        "C": v[ns].str(max(45, DIG), radius=False),
        "prec": PREC,
        "ngh": NGH,
        "rp": RP,
        "func": FUNC,
        "lam": LAM,
        "src_s": SRC_S,
    }
    if extra:
        out.update(extra)
    json.dump(out, open(path, "w"), indent=1)


def cmd_newton(argv):
    dA, vA, rmA = load(argv[0])
    S = Solver(dA["X"], dA["Y"], len(dA["G"]))
    rm = arb(argv[2])
    mode = argv[3] if len(argv) > 3 else "harm"
    maxit = int(argv[4]) if len(argv) > 4 else 20
    if argv[1] != "-":  # secant predictor in harmonic coordinates
        dB, vB, rmB = load(argv[1])
        lam = (rm - arb(rmA)) / (arb(rmA) - arb(rmB))
        mA = S.moments(vA)
        mB = S.moments(vB)
        tgt = [a + (a - b) * lam for a, b in zip(mA[:-1], mB[:-1])] + [arb(1)]
        guess = [vA[i] + (vA[i] - vB[i]) * lam for i in range(S.n)]
        v0, mr = S.moment_solve(tgt, guess)
        v0 = v0[: S.ns] + [vA[S.ns] + (vA[S.ns] - vB[S.ns]) * lam]
        say(f"secant predictor (harmonic coordinates) at rm {argv[2]}; moment residual {mr:.1e}")
    elif "C" in dA and abs(float(argv[2]) - float(rmA)) < 1e-15:  # same rm: start from the state as it is
        v0 = list(vA)
        say(f"start state {argv[0]} at its own rm {argv[2]}")
    elif "C" not in dA:  # a start ring without C (q796 copying ring)
        v0 = c_from_atoms(S, vA, rm)
        say(f"start ring {argv[0]} at rm {argv[2]}, C from the atoms")
    else:  # tangent predictor
        J, Fr = S.jac(vA, arb(rmA), with_rm=True)
        t = J.solve(-Fr)
        d = rm - arb(rmA)
        v0 = [(vA[i] + d * t[i, 0]).mid() for i in range(S.n)]
        say(f"tangent predictor at rm {argv[2]}")
    v, nr, ok, hist = S.newton(v0, rm, mode=mode, maxit=maxit, diag=os.environ.get("DIAG", "0") == "1")
    say(f"mode {mode}: final residual {nr:.3e}, converged {ok}")
    marg, best = S.scan(v, rm)
    say(
        "margins midway: "
        + "; ".join(f"{a:.4f} {f2(x):+.6e}" for a, x in marg)
        + f"; scan max {f2(best[1]):+.6e} at {best[0]:.3f} deg"
    )
    save(
        S,
        v,
        argv[2],
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "D_work",
            f"D_newton_{mode}_rm{argv[2]}"
            + (f"_ngh{NGH}" if NGH != 200 else "")
            + (f"_p{PREC}" if PREC != 160 else "")
            + os.environ.get("TAG", "")
            + ".json",
        ),
        {"residual": nr, "margins": [[a, f2(x)] for a, x in marg], "scanmax": [best[0], f2(best[1])]},
    )


if __name__ == "__main__":
    if sys.argv[1] == "newton":
        cmd_newton(sys.argv[2:])
    elif sys.argv[1] == "track":
        from D_track import cmd_track

        cmd_track(sys.argv[2:])
