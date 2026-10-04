"""q656 (copy of q650 as text, 3D library d70_lib3d; Amendment 976): continuation of the 3D ball channel (shells + origin). From q650: Continuation in A of the 2D disc channel's optimal input (rings + origin point), with event handling, using
d50_lib (definitions only). Events: birth at the origin (no 'o' and i(0) - C > 0), lift-off ('o' present and i''(0) > 0), interior birth
(a local maximum of i - C > 0 at r > 0.05 away from the support). Each event is bisected in A on converged states to 1e-7.
Writes q656_shells3d.json (states at A = 3, 4, ..., 12 and every event) and prints a log. Usage: py q649_rings2d_cont.py [AMAX]
"""

import sys, json, time
import numpy as np
import d70_lib3d as L

AMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
T0 = time.time()
say = lambda *a: print(*a, flush=True)


def solve(A, types, r, w, C):
    """Newton at A from a seed (r, w, C); returns (r, w, C, ok)."""
    g = L.Grid(A)
    r = np.array(r, float)
    r[[j for j, t in enumerate(types) if t == "w"]] = A
    x0 = L.pack(r, w, C, types)
    x, nrm, it, ok = L.newton(g, x0, types)
    rr, ww, CC = L.unpack(x, types, A)
    ok = (
        ok and np.all(ww > 0) and np.all(np.diff(rr[[j for j, t in enumerate(types) if t != "o"]]) > 0)
        if len(rr) > 1
        else ok
    )
    return rr, ww, CC, bool(ok), g


def signals(g, types, r, w, C):
    vmax, rmax, i0mC, i2_0 = L.global_check(g, r, w, C)
    if rmax is not None and np.isfinite(
        vmax
    ):  # confirm an interior candidate directly, away from the support
        rq = np.linspace(max(rmax - 0.05, 0.0), min(rmax + 0.05, g.A), 201)
        v = g.kkt(rq, r, w) - C
        k = int(np.argmax(v))
        rc = rq[k]
        if np.min(np.abs(np.asarray(r) - rc)) < 0.1 or v[k] <= 1e-10:
            vmax = -np.inf
        else:
            vmax, rmax = float(v[k]), float(rc)
    return vmax, rmax, i0mC, i2_0


def ordered(types, r, w):
    idx = np.argsort(r)
    return [types[i] for i in idx], r[idx], w[idx]


def event_of(types, sig):
    vmax, rmax, i0mC, i2_0 = sig
    TH = 1e-10  # signals below this are round-off (a lift-off leaves i(0) - C at 1e-14)
    if "o" not in types and i0mC > TH:
        return "birth", i0mC
    if "o" in types and i2_0 > TH:
        return "liftoff", i2_0
    if vmax > TH and rmax is not None and rmax > 0.05:
        return "interior", vmax
    return None, None


def evsignal(kind, sig):
    vmax, rmax, i0mC, i2_0 = sig
    return {"birth": i0mC, "liftoff": i2_0, "interior": vmax}[kind]


# start just above A1 with the origin point and the wall ring
A = 3.08
types = ["o", "w"]
r = np.array([0.0, A])
w = np.array([0.05, 0.95])
g = L.Grid(A)
C = g.mutual_info(r, w)
r, w, C, ok, g = solve(A, types, r, w, C)
say(f"start A = {A}: ok {ok}, r {np.round(r, 6)}, w {np.round(w, 6)}, C {C:.10f}")
out = {"states": [], "events": []}
dA = 0.02
marks = [float(k) for k in range(3, int(AMAX) + 1)]
while A < AMAX - 1e-12:
    A1 = min(A + dA, AMAX)
    for nxt in marks:
        if A < nxt < A1:
            A1 = nxt
    rs = r.copy()
    rs[[j for j, t in enumerate(types) if t == "r"]] *= A1 / A
    r1, w1, C1, ok1, g1 = solve(A1, types, rs, w, C)
    if not ok1:
        dA /= 2
        say(f"  step to {A1:.6f} failed; dA -> {dA:.2e}")
        if dA < 1e-6:
            say("STOP: continuation failed")
            break
        continue
    sig = signals(g1, types, r1, w1, C1)
    kind, val = event_of(types, sig)
    if kind == "interior":  # a young ring near the origin may sit on the wrong branch: re-seed it
        jr = [j for j, t in enumerate(types) if t == "r" and r1[j] < 0.5]
        for j in jr:
            for r0 in (0.1, 0.25, 0.5, 0.8):
                rs2 = r1.copy()
                rs2[j] = min(r0, 0.9 * A1)
                r2, w2, C2, ok2, g2 = solve(A1, types, rs2, w1, C1)
                if ok2:
                    sig2 = signals(g2, types, r2, w2, C2)
                    k2, v2 = event_of(types, sig2)
                    if k2 is None:
                        r1, w1, C1, g1, sig, kind, val = r2, w2, C2, g2, sig2, None, None
                        break
            if kind is None:
                break
    if kind is None:
        A, r, w, C, g = A1, r1, w1, C1, g1
        dA = min(dA * 1.5, 0.05)
        if any(abs(A - mk) < 1e-12 for mk in marks):
            out["states"].append({"A": A, "types": types, "r": r.tolist(), "w": w.tolist(), "C": C})
            json.dump(out, open("q656_shells3d.json", "w"), indent=1)  # checkpoint at every mark
            say(
                f"A = {A:.4f}: {len(types)} support points {''.join(types)}; outer depths A - r: {np.round(A - np.sort(r)[::-1][:5], 6)}  [{time.time() - T0:.0f}s]"
            )
        continue
    # bisect the event between A (no event) and A1 (event), on converged states
    lo, hi = A, A1
    rl, wl, Cl = r.copy(), w.copy(), C
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        rm = rl.copy()
        rm[[j for j, t in enumerate(types) if t == "r"]] *= mid / lo
        rmid, wmid, Cmid, okm, gm = solve(mid, types, rm, wl, Cl)
        if not okm:
            break
        k2, v2 = event_of(types, signals(gm, types, rmid, wmid, Cmid))
        if k2 == kind:
            hi = mid
        else:
            lo, rl, wl, Cl = mid, rmid, wmid, Cmid
    At = 0.5 * (lo + hi)
    sg = signals(g1, types, r1, w1, C1)
    say(
        f"EVENT {kind} at A in ({lo:.7f}, {hi:.7f}]  (signal past it {val:+.3e}; interior max at r = {sg[1]})"
    )
    out["events"].append({"kind": kind, "A_lo": lo, "A_hi": hi, "rmax": sg[1], "n_before": len(types)})
    # switch the support just past the event
    A = hi
    done = False
    if kind == "birth":
        for off in (
            0.0,
            1e-3,
            3e-3,
            1e-2,
        ):  # a tiny origin mass right at the event: step past it, try light seeds too
            A = hi + off
            for m0 in (1e-3, 0.03, 0.1, 1e-4, 1e-5, 0.01):
                t2 = ["o"] + types
                rr = np.concatenate([[0.0], rl * (A / lo)])
                ww = np.concatenate([[m0], wl * (1 - m0)])
                r2, w2, C2, ok2, g2 = solve(A, t2, rr, ww, Cl)
                if ok2 and event_of(t2, signals(g2, t2, r2, w2, C2))[0] is None:
                    types, r, w, C, g, done = t2, r2, w2, C2, g2, True
                    break
            if done:
                break
    elif kind == "liftoff":
        j0 = types.index("o")
        for off in (2e-4, 1e-3, 3e-3):  # pitchfork: the young ring grows like 2 sqrt(A - A_t)
            A = hi + off
            for r0 in (2 * np.sqrt(off), 1.5 * np.sqrt(off), 3 * np.sqrt(off), 0.05, 0.1, 0.2, 0.3):
                t2 = ["r" if t == "o" else t for t in types]
                rr = rl * (A / lo)
                rr[j0] = r0
                r2, w2, C2, ok2, g2 = solve(A, t2, rr, wl, Cl)
                if ok2 and event_of(t2, signals(g2, t2, r2, w2, C2))[0] is None:
                    t2s, r2s, w2s = ordered(t2, r2, w2)
                    types, r, w, C, g, done = t2s, r2s, w2s, C2, g2, True
                    break
            if done:
                break
    else:
        rb = sg[1]
        t2 = types + ["r"]
        rr = np.concatenate([rl * (A / lo), [rb]])
        ww = np.concatenate([wl * 0.99, [0.01]])
        r2, w2, C2, ok2, g2 = solve(A, t2, rr, ww, Cl)
        if ok2:
            t2s, r2s, w2s = ordered(t2, r2, w2)
            types, r, w, C, g, done = t2s, r2s, w2s, C2, g2, True
    if not done:
        say("STOP: could not continue past the event")
        break
    say(
        f"   now {len(types)} support points {''.join(types)} at A = {A:.7f}: r {np.round(r, 5)} w {np.round(w, 5)}"
    )
    dA = 0.01  # next step starts from hi; the threshold keeps round-off from re-triggering
json.dump(out, open("q656_shells3d.json", "w"), indent=1)
say(
    f"done [{time.time() - T0:.0f}s]; events: {len(out['events'])}; kinds: {[e['kind'] for e in out['events']]}"
)
