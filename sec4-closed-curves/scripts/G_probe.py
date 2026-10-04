"""G_probe (03 Oct 2026): coarse probes and bisection for a tracked ring (coordinator's pace note, 06:42), with the
Solver (harmonic Newton; env as D_solver: FUNC, RP, PREC, FDH, TOLR, TOLV, CHORD, REUSEJ, NPROC, MAXIT, NSCAN) and the full reads of
G_trackE.fine_reads (D - C at t = 0 and pi/2, fine windows 0-5 and 85-90 deg at 0.05 deg, curvature at every atom) at every state.
A state is VALID when: residual < TOLR, every mass > 0, the NSCAN-point scan maximum <= TOLV, both fine-window maxima <= TOLV, D - C at
t = 0 <= TOLV, and every atom curvature < 0. States go to OUTDIR/D_rm<rm>.json (D_solver.save, full precision); one line per state is
appended to G_family.md (append-only) and printed.
Modes:
  probe OUTDIR STATE_A STATE_B "rm1,rm2,..."   secant predictor (moment coordinates, as D_solver's newton) from the two newest valid
                                                states; on a failed solve the step to that probe is halved (that probe only); stops at
                                                the first invalid state.
  bisect OUTDIR LO.json HI.json BTOL            bisection in rm between a valid and an invalid state; predictor = linear interpolation
                                                in moment coordinates between the two bracketing states."""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb
import D_solver as DS
from D_solver import Solver, load, save, f2
from G_trackE import fine_reads

T0 = time.time()
say = lambda *a: print(*a, f"[{time.time() - T0:.0f}s]", flush=True)
FAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "G_family.md")


def predict(S, A, B, rm):
    """A = (rm_a arb, v_a), B = (rm_b, v_b); linear extrapolation/interpolation in moment coordinates and C"""
    lam = (rm - B[0]) / (B[0] - A[0])
    mA = S.moments(A[1])
    mB = S.moments(B[1])
    tgt = [b + (b - a) * lam for a, b in zip(mA[:-1], mB[:-1])] + [arb(1)]
    guess = [B[1][i] + (B[1][i] - A[1][i]) * lam for i in range(S.n)]
    v0, mr = S.moment_solve(tgt, guess)
    return v0[: S.ns] + [B[1][S.ns] + (B[1][S.ns] - A[1][S.ns]) * lam], mr


def evaluate(S, v, rm_s, outdir, nr):
    rm = arb(rm_s)
    marg, best = S.scan(v, rm, DS.NSCAN)
    fr = fine_reads(S, v, rm)
    mmin = min(f2(x) for x in v[S.nG : S.ns])
    tol = DS.TOLV
    viol = []
    if f2(best[1]) > tol:
        viol.append(f"scan max {f2(best[1]):+.3e} at {best[0]:.2f}")
    if fr["fine0_5_max"][1] > tol:
        viol.append(f"fine 0-5 max {fr['fine0_5_max'][1]:+.3e} at {fr['fine0_5_max'][0]}")
    if fr["fine85_90_max"][1] > tol:
        viol.append(f"fine 85-90 max {fr['fine85_90_max'][1]:+.3e} at {fr['fine85_90_max'][0]}")
    if fr["DC_t0"] > tol:
        viol.append(f"D - C(0) {fr['DC_t0']:+.3e}")
    pos = [i for i, x in enumerate(fr["d2_atoms"]) if x >= 0]
    if pos:
        viol.append(f"curvature >= 0 at atom index {pos}")
    if mmin <= 0:
        viol.append("mass <= 0")
    ok = not viol
    save(
        S,
        v,
        rm_s,
        os.path.join(outdir, f"D_rm{rm_s}.json"),
        {
            "residual": nr,
            "margins": [[a, f2(x)] for a, x in marg],
            "scanmax": [best[0], f2(best[1])],
            "valid": ok,
            "violations": viol,
            **fr,
        },
    )
    G = [round(float((g * 180 / S.P.pi).mid()), 5) for g in v[: S.nG]]
    m = [round(f2(x), 7) for x in v[S.nG : S.ns]]
    line = (
        f"  probe rm {rm_s} (eps {1 - float(rm_s) if DS.RP in ('1', '1.0') else float(DS.RP) - float(rm_s):.9f}) {'VALID' if ok else 'INVALID'}"
        f"{'' if ok else ' (' + '; '.join(viol) + ')'}: residual {nr:.1e}; G deg {G}; masses {m}; margins "
        f"{[f'{a:.2f}:{f2(x):+.3e}' for a, x in marg]}; D - C(0) {fr['DC_t0']:+.3e}; D - C(90) {fr['DC_t90']:+.3e}; fine 0-5 max "
        f"{fr['fine0_5_max'][1]:+.3e} at {fr['fine0_5_max'][0]}; fine 85-90 max {fr['fine85_90_max'][1]:+.3e} at {fr['fine85_90_max'][0]}; "
        f"curvatures {[f'{x:+.3e}' for x in fr['d2_atoms']]}"
    )
    say(line)
    with open(FAM, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    return ok


def solve(S, pred, rm_s):
    v, nr, ok, hist = S.newton(
        pred,
        arb(rm_s),
        mode="harm",
        maxit=DS.MAXIT if hasattr(DS, "MAXIT") else int(os.environ.get("MAXIT", "30")),
        verbose=False,
    )
    say(f"solve rm {rm_s}: converged {ok}, {len(hist) - 1} steps, residual {nr:.1e}")
    return v, nr, ok


def cmd_probe(argv):
    out = argv[0]
    os.makedirs(out, exist_ok=True)
    dA, vA, rA = load(argv[1])
    dB, vB, rB = load(argv[2])
    S = Solver(dB["X"], dB["Y"], len(dB["G"]))
    A = (arb(str(rA)), vA)
    B = (arb(str(rB)), vB)
    targets = [x for x in argv[3].split(",")]
    with open(FAM, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            f"- {time.strftime('%Y-%m-%d %H:%M')} G_probe probe from {os.path.basename(argv[1])}, {os.path.basename(argv[2])}; targets {targets}\n"
        )
    for t in targets:
        goal = float(t)
        cur = float(B[0].mid().str(20, radius=False))
        step = goal - cur
        while True:
            rm_s = repr(round(cur + step, 12))
            pred, mr = predict(S, A, B, arb(rm_s))
            v, nr, ok = solve(S, pred, rm_s)
            if ok:
                break
            step /= 2
            say(f"  halving the step to {step:.3e}")
            if abs(step) < 1e-7:
                say("stop: step below 1e-7")
                return
        valid = evaluate(S, v, rm_s, out, nr)
        if not valid:
            say(f"first invalid probe at rm {rm_s}; last valid rm {B[0].mid().str(15, radius=False)}")
            return
        A, B = B, (arb(rm_s), v)
        if abs(float(rm_s) - goal) > 1e-12:  # a halved step: continue to the same goal
            targets_left = True
            while abs(float(B[0].mid().str(20, radius=False)) - goal) > 1e-12:
                cur = float(B[0].mid().str(20, radius=False))
                step = goal - cur
                while True:
                    rm_s = repr(round(cur + step, 12))
                    pred, mr = predict(S, A, B, arb(rm_s))
                    v, nr, ok = solve(S, pred, rm_s)
                    if ok:
                        break
                    step /= 2
                    say(f"  halving the step to {step:.3e}")
                    if abs(step) < 1e-7:
                        say("stop: step below 1e-7")
                        return
                if not evaluate(S, v, rm_s, out, nr):
                    say(f"first invalid probe at rm {rm_s}")
                    return
                A, B = B, (arb(rm_s), v)
    say("all probes valid")


def cmd_bisect(argv):
    out = argv[0]
    os.makedirs(out, exist_ok=True)
    btol = float(argv[3])
    dL, vL, rL = load(argv[1])
    dH, vH, rH = load(argv[2])
    S = Solver(dL["X"], dL["Y"], len(dL["G"]))
    L = (arb(str(rL)), vL)
    H = (arb(str(rH)), vH)
    with open(FAM, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            f"- {time.strftime('%Y-%m-%d %H:%M')} G_probe bisect between {os.path.basename(argv[1])} and {os.path.basename(argv[2])}, BTOL {btol}\n"
        )
    while float(H[0].mid().str(20, radius=False)) - float(L[0].mid().str(20, radius=False)) > btol:
        mid = repr(
            round((float(L[0].mid().str(20, radius=False)) + float(H[0].mid().str(20, radius=False))) / 2, 13)
        )
        pred, mr = predict(S, L, H, arb(mid))
        v, nr, ok = solve(S, pred, mid)
        if not ok:
            say(f"bisect: no convergence at {mid}; stop")
            return
        if evaluate(S, v, mid, out, nr):
            L = (arb(mid), v)
        else:
            H = (arb(mid), v)
    say(f"BRACKET rm ({L[0].mid().str(15, radius=False)}, {H[0].mid().str(15, radius=False)}]")


if __name__ == "__main__":
    {"probe": cmd_probe, "bisect": cmd_bisect}[sys.argv[1]](sys.argv[2:])
