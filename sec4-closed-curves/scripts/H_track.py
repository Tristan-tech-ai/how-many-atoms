"""H_track (03 Oct 2026): continuation of one ring in rm on the mixed curve, adapted from the D_track.cmd_track (copied as
text; D_track's predictor and tangent are imported unchanged) with the full read set of H_arb.H_reads at EVERY state and the full KKT
test of H_arb.valid_check deciding validity (scan, fine reads at both vertices 0-5 / 85-90 deg at 0.05 deg, golden-refined gap maxima,
margins, atom curvatures, masses), so a narrow bump next to a vertex cannot be stepped over by the coarse scan. Bisection in rm on the
same test to BTOL. Every solved state is written at once to OUTDIR/H_rm<rm>.json with its reads (full-precision strings).
Usage: py H_track.py OUTDIR RM_END STEP STATE_1[,STATE_2[,STATE_3]]   (states oldest first; RM_END below the start = track downward)
Env as D_solver (FUNC, RP, LAM, PREC, FDH, TOLR, TOLV, BTOL, NSCAN) plus STEPMAX, MAXIT, NFINE (reads scan, default 360).
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb, arb_mat, ctx
import D_solver as DS
from D_solver import Solver, load, save, say, f2
import D_track as DT
from H_arb import H_reads, valid_check, short


def cmd_track(argv):
    outdir, rmE, step = argv[0], float(argv[1]), float(argv[2])
    files = argv[3].split(",")
    os.makedirs(outdir, exist_ok=True)
    STEPMAX = float(os.environ.get("STEPMAX", "2e-3"))
    MAXIT = int(os.environ.get("MAXIT", "12"))
    NFINE = int(os.environ.get("NFINE", "360"))
    TOLV = DS.TOLV
    d0, v0, rm0 = load(files[-1])
    S = Solver(d0["X"], d0["Y"], len(d0["G"]))
    hist = []
    for f in files:
        d, v, rmf = load(f)
        hist.append((arb(rmf), v))
    rmc = arb(rm0)
    v, nr, ok, _ = S.newton(hist[-1][1], rmc, maxit=MAXIT, verbose=False)
    hist[-1] = (rmc, v)
    DT.TANS[rmc.mid().str(30, radius=False)] = DT.tangent(S, v, rmc)
    rd = H_reads(S, v, rmc, NFINE)
    bad = valid_check(S, v, rd, TOLV)
    say(
        f"start rm {rm0} (eps {float(DS.RP) - float(rm0):.7f}) X{int(S.P.X)} Y{int(S.P.Y)} nG {S.nG}: residual {nr:.2e} ({'VALID' if not bad else 'INVALID ' + str(bad)}); "
        f"PREC {DS.PREC} FUNC {DS.FUNC} TOLV {TOLV:.0e} TOLR {DS.TOLR:.0e}"
    )
    say("   " + short(rd))

    def solve(rm_s):
        rm = arb(rm_s)
        vp = DT.predict(S, hist_valid, rm)
        v, nr, ok, h = S.newton(vp, rm, maxit=MAXIT, verbose=os.environ.get("VERB", "0") == "1")
        if not ok:
            return None, nr, len(h)
        DT.TANS[rm.mid().str(30, radius=False)] = DT.tangent(S, v, rm)
        rd = H_reads(S, v, rm, NFINE)
        bad = valid_check(S, v, rd, TOLV)
        G = [float((g * 180 / S.P.pi).mid()) for g in v[: S.nG]]
        st = dict(rm=rm_s, v=v, nr=nr, its=len(h) - 1, rd=rd, bad=bad, G=G)
        save(
            S,
            v,
            rm_s,
            os.path.join(outdir, f"H_rm{rm_s}.json"),
            {"residual": nr, "reads": rd, "bad": bad, "tolv": TOLV},
        )
        return st, nr, len(h)

    def show(st, tag=""):
        say(
            f"  rm {st['rm']} (eps {float(DS.RP) - float(st['rm']):.9f}) {tag}: res {st['nr']:.1e} ({st['its']} its); G {[round(g, 4) for g in st['G']]}; "
            f"masses {[round(f2(x), 6) for x in st['v'][S.nG:S.ns]]}"
            + ("" if not st["bad"] else f"; FAILS {st['bad']}")
        )
        say("     " + short(st["rd"]))

    hist_valid = hist
    cur_rm = float(rm0)
    sg = 1.0 if rmE > cur_rm else -1.0  # 05:4x: downward tracking (rm decreasing) allowed
    if bad:
        say("start state is not KKT-valid; tracking anyway is pointless: stop")
        return
    while sg * (rmE - cur_rm) > 1e-13:
        rmn = round(cur_rm + sg * min(step, abs(rmE - cur_rm)), 12)
        rm_s = repr(rmn)
        st, nr, its = solve(rm_s)
        if st is None:
            step /= 2
            say(f"  rm {rm_s}: no convergence (residual {nr:.1e}), step -> {step:.2e}")
            if step < 1e-7:
                say("stop: step below 1e-7")
                return
            continue
        if not st["bad"]:
            show(st, "valid")
            hist_valid.append((arb(rm_s), st["v"]))
            cur_rm = rmn
            step = min(step * 1.5, STEPMAX)
            continue
        show(st, "INVALID")
        lo, hi = (cur_rm, None), (rmn, st)
        while abs(hi[0] - lo[0]) > DS.BTOL:
            mid = round((lo[0] + hi[0]) / 2, 12)
            s, nr, its = solve(repr(mid))
            if s is None:
                say(f"  bisect rm {mid}: no convergence ({nr:.1e}); stop bisection")
                break
            ok = not s["bad"]
            show(s, "valid" if ok else "INVALID")
            if ok:
                lo = (mid, s)
                hist_valid.append((arb(repr(mid)), s["v"]))
            else:
                hi = (mid, s)
        say(
            f"EVENT: K {2*int(S.P.X) + 2*int(S.P.Y) + 4*S.nG} (X{int(S.P.X)} Y{int(S.P.Y)} nG {S.nG}) loses KKT between rm {lo[0]!r} (last valid) and {hi[0]!r} (first invalid) "
            f"(eps {float(DS.RP) - lo[0]:.9f} valid, {float(DS.RP) - hi[0]:.9f} invalid); first invalid fails: {hi[1]['bad']}"
        )
        return
    say(f"end at rm {cur_rm} without event")


if __name__ == "__main__":
    cmd_track(sys.argv[1:])
