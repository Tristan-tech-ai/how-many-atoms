"""D_track: continuation of one ring structure in rm with D_solver's harmonic-coordinate Newton, and the tracker's event
test copied from q788 (max of D - C on NSCAN + 1 points of [0, 90] deg above TOLV = KKT lost; bisect rm between the last valid and the
first invalid state to BTOL; rm values rounded to 12 decimals as in q788). Predictor: quadratic (or linear) extrapolation in rm of the
ring's moments b_2..b_(2 ns - 2) and of C through the last converged states, mapped back by the moment solve.
Extra readings at each accepted state (not used for decisions): a fine scan of D - C on [85, 90) deg at 0.05 deg, and the curvature of
D - C at the minor-vertex atom, d2 = 2 (r(90 - 1e-3 rad) - r(90))/1e-6 (symmetric about 90 deg).
Each state is written to OUTDIR/D_rm<rm>.json with full-precision strings; the log goes to stdout.
Usage (via D_solver.py): py D_solver.py track OUTDIR RM_END STEP STATE_1,STATE_2[,STATE_3]   (env STEPMAX, MAXIT)
"""

import os, json
from flint import arb, arb_mat, ctx
import D_solver as DS
from D_solver import Solver, load, save, say, f2


def fine_reads(S, v, rm):
    P = S.P
    out = {}
    dd = arb("1e-3")  # 20:2x: vertex reads for any layout
    if P.X:
        out["d2_X"] = f2(2 * (P.kkt(v, rm, dd) - P.kkt(v, rm, arb(0))) / (dd * dd))
    else:
        out["DC_t0"] = f2(P.kkt(v, rm, arb(0)))
    if not P.Y:
        out["DC_t90"] = f2(P.kkt(v, rm, P.pi / 2))
    if P.Y and os.environ.get("FINE85", "1") == "0":
        out["d2_Y"] = f2(2 * (P.kkt(v, rm, P.pi / 2 - dd) - P.kkt(v, rm, P.pi / 2)) / (dd * dd))
    elif P.Y:
        best = None
        for i in range(100):
            t = (P.pi / 2) * (arb(85) + arb(i) * arb("0.05")) / 90
            e = P.kkt(v, rm, t)
            if best is None or e > best[1]:
                best = (85 + 0.05 * i, e)
        out["fine85_90"] = [best[0], f2(best[1])]
        d = arb("1e-3")
        out["d2_Y"] = f2(2 * (P.kkt(v, rm, P.pi / 2 - d) - P.kkt(v, rm, P.pi / 2)) / (d * d))
    return out


TANS = {}  # 21:1x: tangents by rm string


def tangent(S, v, rm):
    """branch tangent at a converged state: dv/drm = -J^-1 F_rm (J of the last solve, F_rm by central differences), and the same in
    the coordinates the predictor uses (moments b_2.., then C)."""
    J, Fr = S.jac(v, rm, with_rm=True)
    S._lastJ = J  # 21:2x: a fresh J at the converged state (a J from the
    # previous iterate gave tangents of 1e28: the soft block moves)
    tv = J.solve(-Fr)
    tvl = [tv[i, 0] for i in range(S.n)]
    db = S.moments_jac(v) * arb_mat(S.ns, 1, tvl[: S.ns])
    return [db[i, 0] for i in range(S.ns - 1)] + [tvl[S.ns]], tvl


def predict(S, hist, rm):
    """hist: list of (rm arb, v) of converged states, oldest first; returns the predicted v at rm. With env PRED=tan and a stored
    tangent at the last state: Taylor step along the tangent, plus the quadratic term fixed by the state before (Hermite form).
    """
    pts = hist[-3:]
    xs = [p[0] for p in pts]
    ys = [S.moments(p[1])[:-1] + [p[1][S.ns]] for p in pts]
    key = xs[-1].mid().str(30, radius=False)
    if os.environ.get("PRED", "tan") == "tan" and key in TANS:
        tb, tv = TANS[key]
        d = rm - xs[-1]
        y1 = ys[-1]
        yp = [y + t * d for y, t in zip(y1, tb)]
        vg = [(x + t * d).mid() for x, t in zip(pts[-1][1], tv)]
        if len(pts) >= 2:
            d0 = xs[-2] - xs[-1]
            kap = [(y0 - y - t * d0) / (d0 * d0) for y0, y, t in zip(ys[-2], y1, tb)]
            yp = [y + k * d * d for y, k in zip(yp, kap)]
        v0, mr = S.moment_solve(yp[:-1] + [arb(1)], vg)
        return v0[: S.ns] + [yp[-1]]
    if len(pts) == 1:
        return list(pts[0][1])
    L = []
    for i in range(len(pts)):
        c = arb(1)
        for j in range(len(pts)):
            if j != i:
                c = c * (rm - xs[j]) / (xs[i] - xs[j])
        L.append(c)
    yp = [sum((L[i] * ys[i][k] for i in range(len(pts))), arb(0)) for k in range(len(ys[0]))]
    vg = [sum((L[i] * pts[i][1][k] for i in range(len(pts))), arb(0)) for k in range(S.n)]
    v0, mr = S.moment_solve(yp[:-1] + [arb(1)], vg)
    return v0[: S.ns] + [yp[-1]]


def cmd_track(argv):
    outdir, rmE, step = argv[0], float(argv[1]), float(argv[2])
    files = argv[3].split(",")
    os.makedirs(outdir, exist_ok=True)
    STEPMAX = float(os.environ.get("STEPMAX", "2e-3"))
    MAXIT = int(os.environ.get("MAXIT", "12"))
    d0, v0, rm0 = load(files[-1])
    S = Solver(d0["X"], d0["Y"], len(d0["G"]))
    hist = []
    for f in files:
        d, v, rmf = load(f)
        hist.append((arb(rmf), v))
    rmc = arb(rm0)
    v, nr, ok, _ = S.newton(hist[-1][1], rmc, maxit=MAXIT, verbose=False)
    marg, best = S.scan(v, rmc)
    hist[-1] = (rmc, v)
    if os.environ.get("PRED", "tan") == "tan":
        TANS[rmc.mid().str(30, radius=False)] = tangent(S, v, rmc)
    say(
        f"start rm {rm0} (eps {float(DS.RP) - float(rm0):.6f}, rp {DS.RP}): residual {nr:.2e}, scan max {f2(best[1]):+.2e} at {best[0]:.2f} deg; "
        f"PREC {DS.PREC}, NGH {DS.NGH}"
    )

    def solve(rm_s):
        rm = arb(rm_s)
        vp = predict(S, hist_valid, rm)
        v, nr, ok, h = S.newton(vp, rm, maxit=MAXIT, verbose=os.environ.get("VERB", "0") == "1")
        if not ok:
            return None, nr, len(h)
        if os.environ.get("PRED", "tan") == "tan":
            TANS[rm.mid().str(30, radius=False)] = tangent(S, v, rm)
        marg, best = S.scan(v, rm)
        fr = fine_reads(S, v, rm)
        G = [float((g * 180 / S.P.pi).mid()) for g in v[: S.nG]]
        st = dict(
            rm=rm_s,
            v=v,
            nr=nr,
            its=len(h) - 1,
            marg=marg,
            best=best,
            fr=fr,
            G=G,
            mmin=min(f2(x) for x in v[S.nG : S.ns]),
        )
        save(
            S,
            v,
            rm_s,
            os.path.join(outdir, f"D_rm{rm_s}.json"),
            {
                "residual": nr,
                "margins": [[a, f2(x)] for a, x in marg],
                "scanmax": [best[0], f2(best[1])],
                "nscan": DS.NSCAN,
                **fr,
            },
        )
        return st, nr, len(h)

    def show(st, tag=""):
        say(
            f"  rm {st['rm']} (eps {float(DS.RP) - float(st['rm']):.7f}) {tag}: res {st['nr']:.1e} ({st['its']} its); G {[round(g, 4) for g in st['G']]}; "
            f"masses {[round(f2(x), 6) for x in st['v'][S.nG:S.ns]]}; margins {[f'{f2(x):+.2e}' for a, x in st['marg']]}; "
            f"scan max {f2(st['best'][1]):+.2e} at {st['best'][0]:.2f}; vertex reads {st['fr']}"
        )

    hist_valid = hist
    cur_rm = float(rm0)
    while cur_rm < rmE:
        rmn = round(min(cur_rm + step, rmE), 12)
        rm_s = repr(rmn)
        st, nr, its = solve(rm_s)
        if st is None:
            step /= 2
            say(f"  rm {rm_s}: no convergence (residual {nr:.1e}), step -> {step:.2e}")
            if step < 1e-7:
                say("stop: step below 1e-7")
                return
            continue
        if f2(st["best"][1]) <= DS.TOLV and st["mmin"] > 0:
            show(st, "valid")
            hist_valid.append((arb(rm_s), st["v"]))
            cur_rm = rmn
            step = min(step * 1.5, STEPMAX)
            continue
        show(st, "INVALID" if f2(st["best"][1]) > DS.TOLV else "negative mass")
        lo, hi = (cur_rm, None), (rmn, st)
        while hi[0] - lo[0] > DS.BTOL:
            mid = round((lo[0] + hi[0]) / 2, 12)
            s, nr, its = solve(repr(mid))
            if s is None:
                say(f"  bisect rm {mid}: no convergence ({nr:.1e}); stop bisection")
                break
            ok = f2(s["best"][1]) <= DS.TOLV and s["mmin"] > 0
            show(s, "valid" if ok else "INVALID")
            if ok:
                lo = (mid, s)
                hist_valid.append((arb(repr(mid)), s["v"]))
            else:
                hi = (mid, s)
        hs = hi[1]
        rmh = arb(hs["rm"])
        marg, best = S.scan(hs["v"], rmh, int(os.environ.get("EVSCAN", "720")))
        say(
            f"EVENT: K {2*int(S.P.X) + 2*int(S.P.Y) + 4*S.nG} loses KKT at rm in ({lo[0]!r}, {hi[0]!r}] (eps in [{float(DS.RP) - hi[0]:.7f}, "
            f"{float(DS.RP) - lo[0]:.7f})); {os.environ.get('EVSCAN', '720')}-point scan of the first invalid state: max {f2(best[1]):+.3e} at {best[0]:.3f} deg; "
            f"margins {[(round(a, 4), f'{f2(x):+.4e}') for a, x in marg]}; vertex reads {hs['fr']}"
        )
        if lo[1] is not None:
            say(
                f"  last valid state: vertex reads {lo[1]['fr']}; margins {[(round(a, 4), f'{f2(x):+.4e}') for a, x in lo[1]['marg']]}"
            )
        return
    say(f"end at rm {cur_rm} without event")
