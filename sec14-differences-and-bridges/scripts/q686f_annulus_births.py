"""q686f (copy of q686e as text: the split seeds reach distance 0.7 on both sides at W_hi + 1e-3, 3e-3, 1e-2, and the fast solve is two
stage (30 Newton iterations, then 120 more only if the residual already fell 100-fold); Amendment 1036). q686e (copy of q686c as text: split insertion uses a fast solve (Newton, 40 iterations, no fallback) on seeds tried in the
order the solutions take, a light new ring at distance 0.05-0.3 from the splitting ring at W_hi + 1e-3, then the pitchfork seeds;
the full solve only as the last resort; Amendment 1033). q686c (copy of q686b as text: SPLIT events, i''(r_k) > 1e-9 at an interior ring, bisected and replaced by a pair; birth
exclusion 0.005 instead of 0.1; wall check i'(a) <= 0, i'(A) >= 0; Amendment 1032). q686b (copy of q686 as text: tsvd fallback on a failed Newton, step counters in the log without widths, RESUME=1 continues
from the last state the script finds in its own sealed file). q686 (Amendment 1027): births of the optimal input on the shell a <= |x| <= a + W at fixed inner radius a, by continuation in the
width W from WSTART (two walls only) to WMAX. Interior births are bisected in W to 1e-7 on converged states; the new ring is inserted
at the maximum of i - C with light seed masses. BLIND: the event widths and the states go only to q686_annulus_a{a}_d{DIM}_SEALED.json;
the log prints the number of support points and nothing else, so the outcomes stay unread until a prediction is registered.
Env DIM (default 2), AIN (default 20). Usage: py q686_annulus_births.py WMAX"""

import sys, os, json, time
import numpy as np
import d85_annulus_lib as L

DIM = float(os.environ.get("DIM", "2"))
a = float(os.environ.get("AIN", "20"))
WMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
T0 = time.time()
say = lambda *x: print(*x, flush=True)
OUT = f"q686_annulus_a{a:g}_d{DIM:g}_SEALED.json"  # same file as q686


def solve(W, types, r, w, C):
    g = L.Grid(a + W, ain=a, dim=DIM)
    L.AIN[0] = a
    r = np.array(r, float)
    r[[j for j, t in enumerate(types) if t == "w"]] = a + W
    r[[j for j, t in enumerate(types) if t == "v"]] = a
    x0 = L.pack(r, w, C, types)
    x, nrm, it, ok = L.newton(g, x0, types, tol=1e-13, maxit=100)
    if not ok:  # two nearly coincident rings after an asymmetric split: min-norm step
        from d74_newton_scaled import newton_tsvd

        x, nrm, it, ok = newton_tsvd(L, g, x0, types, tol=1e-11, rcond=1e-10)
        if not ok:
            x2, n2, it2, ok2 = L.newton(g, x, types, tol=1e-11, maxit=100)
            if ok2 or n2 < nrm:
                x, nrm, ok = x2, n2, ok2
    rr, ww, CC = L.unpack(x, types, a + W, a)
    ok = ok and np.all(ww > 0) and np.all(np.diff(rr) > 0)
    return rr, ww, CC, bool(ok), g


def solve_fast(W, types, r, w, C):
    g = L.Grid(a + W, ain=a, dim=DIM)
    L.AIN[0] = a
    r = np.array(r, float)
    r[[j for j, t in enumerate(types) if t == "w"]] = a + W
    r[[j for j, t in enumerate(types) if t == "v"]] = a
    x0 = L.pack(r, w, C, types)
    n0 = np.max(np.abs(L.residual_jac(g, x0, types)[0]))
    x, nrm, it, ok = L.newton(g, x0, types, tol=1e-13, maxit=30)
    if not ok and nrm < 1e-2 * n0:  # making progress: give it more iterations
        x, nrm, it, ok = L.newton(g, x, types, tol=1e-13, maxit=120)
    rr, ww, CC = L.unpack(x, types, a + W, a)
    ok = ok and np.all(ww > 0) and np.all(np.diff(rr) > 0)
    return rr, ww, CC, bool(ok), g


def event(g, types, r, w, C):
    """('split', i'', r_k, k) if an interior ring has i''(r_k) > 1e-9; ('birth', v, r_c, None) for a new maximum of i - C > 1e-10 at
    least 0.005 from the support; ('wall', ...) if a wall's one-sided condition fails; else (None, ...)."""
    ridx = [j for j, t in enumerate(types) if t == "r"]
    if ridx:
        i2 = g.kkt(r[ridx], r, w, deriv=2)[2]
        k = int(np.argmax(i2))
        if i2[k] > 1e-9:
            return "split", float(i2[k]), float(r[ridx[k]]), ridx[k]
    d1 = g.kkt([g.ain, g.A], r, w, deriv=1)[1]
    if d1[0] > 1e-9 or d1[1] < -1e-9:
        return "wall", float(max(d1[0], -d1[1])), None, None
    vmax, rmax, _, _ = L.global_check(g, r, w, C)
    if rmax is not None and np.isfinite(vmax):
        rq = np.linspace(max(rmax - 0.05, g.ain), min(rmax + 0.05, g.A), 401)
        v = g.kkt(rq, r, w) - C
        k = int(np.argmax(v))
        rc = rq[k]
        if np.min(np.abs(np.asarray(r) - rc)) < 0.005 or v[k] <= 1e-10:
            return None, None, None, None
        return "birth", float(v[k]), float(rc), None
    return None, None, None, None


def scaled(r, W0, W1):
    """map a state from width W0 to W1 by stretching the depth from the inner wall."""
    return a + (np.asarray(r) - a) * W1 / W0


W = float(os.environ.get("WSTART", "2.0"))
types = ["v", "w"]
r = np.array([a, a + W])
w = np.array([0.5, 0.5])
g = L.Grid(a + W, ain=a, dim=DIM)
L.AIN[0] = a
C = g.mutual_info(r, w)
r, w, C, ok, g = solve(W, types, r, w, C)
say(f"start: ok {ok}, {len(types)} points")
out = {"DIM": DIM, "a": a, "states": [], "events": []}
if os.environ.get("RESUME") == "1":  # continue from the last saved state (read by the script, not printed)
    out = json.load(open(OUT))
    last = max(out["states"], key=lambda s_: s_["W"])
    W = last["W"]
    types = list(last["types"])
    r = np.array(last["r"])
    w = np.array(last["w"])
    C = last["C"]
    out["events"] = [e for e in out["events"] if e["W_lo"] < W]
    r, w, C, ok, g = solve(W, types, r, w, C)
    say(f"resumed: ok {ok}, {len(types)} points")
marks = [float(k) for k in range(int(W) + 1, int(WMAX) + 1)]
dW = 0.05
nacc = nfail = 0
while W < WMAX - 1e-12:
    W1 = min(W + dW, WMAX)
    for mk in marks:
        if W < mk < W1:
            W1 = mk
    r1, w1, C1, ok1, g1 = solve(W1, types, scaled(r, W, W1), w, C)
    if not ok1:
        nfail += 1
        if nfail % 20 == 0:
            say(
                f"  steps: {nacc} accepted, {nfail} failed, dW {dW:.1e}, {len(types)} points  [{time.time() - T0:.0f}s]"
            )
        dW /= 2
        if dW < 1e-7:
            say("STOP: continuation failed")
            break
        continue
    kind, v, rc, kr = event(g1, types, r1, w1, C1)
    if kind is None:
        W, r, w, C, g = W1, r1, w1, C1, g1
        dW = min(dW * 1.5, 0.05)
        nacc += 1
        if any(abs(W - mk) < 1e-12 for mk in marks):
            out["states"].append({"W": W, "types": types, "r": r.tolist(), "w": w.tolist(), "C": C})
            json.dump(out, open(OUT, "w"), indent=1)
            say(f"mark {len(out['states'])}: {len(types)} points  [{time.time() - T0:.0f}s]")
        continue
    if kind == "wall":
        out["events"].append({"kind": "wall", "W_lo": W, "W_hi": W1})
        json.dump(out, open(OUT, "w"), indent=1)
        say("STOP: a wall condition failed (recorded)")
        break
    lo, hi = W, W1
    rl, wl, Cl = r.copy(), w.copy(), C
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        rm, wm, Cm, okm, gm = solve(mid, types, scaled(rl, lo, mid), wl, Cl)
        if not okm:
            break
        k2, _, _, kr2 = event(gm, types, rm, wm, Cm)
        if k2 == kind and (kind != "split" or kr2 == kr):
            hi = mid
        else:
            lo, rl, wl, Cl = mid, rm, wm, Cm
    rh, wh, Ch, okh, gh = solve(hi, types, scaled(rl, lo, hi), wl, Cl)
    _, _, rb, krh = event(gh, types, rh, wh, Ch)
    rb = rc if rb is None else rb
    out["events"].append(
        {
            "kind": kind,
            "W_lo": lo,
            "W_hi": hi,
            "r_event": rb,
            "ring": kr,
            "n_before": len(types),
            "types": list(types),
            "r": rl.tolist(),
            "w": wl.tolist(),
            "C": float(Cl),
        }
    )
    json.dump(out, open(OUT, "w"), indent=1)
    say(f"event {len(out['events'])} ({kind}) bisected  [{time.time() - T0:.0f}s]")
    done = False
    if kind == "birth":
        for off in (1e-6, 1e-4, 1e-3, 1e-2):
            W2 = hi + off
            for m0 in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-3, 1e-2):
                t2 = types + ["r"]
                rr = np.concatenate([scaled(rl, lo, W2), [a + (rb - a) * W2 / hi]])
                ww = np.concatenate([wl * (1 - m0), [m0]])
                o = np.argsort(rr)
                t2 = [t2[i] for i in o]
                rr = rr[o]
                ww = ww[o]
                r2, w2, C2, ok2, g2 = solve(W2, t2, rr, ww, Cl)
                if ok2 and event(g2, t2, r2, w2, C2)[0] is None:
                    types, r, w, C, g, W, done = t2, r2, w2, C2, g2, W2, True
                    break
            if done:
                break
    else:  # split: a light new ring next to ring kr first (fast solves)
        for off in (1e-3, 3e-3, 1e-2, 2e-4):
            W2 = hi + off
            base = scaled(rl, lo, W2)
            for dist in (0.1, 0.3, 0.05, 0.2, 0.5, 0.03, 0.7):
                for sgn in (+1, -1):
                    rr = np.concatenate([base, [base[kr] + sgn * dist]])
                    ww = np.concatenate([wl * (1 - 1e-4), [1e-4]])
                    t2 = list(types) + ["r"]
                    o = np.argsort(rr)
                    t2 = [t2[i] for i in o]
                    rr = rr[o]
                    ww = ww[o]
                    r2, w2, C2, ok2, g2 = solve_fast(W2, t2, rr, ww, Cl)
                    if ok2 and event(g2, t2, r2, w2, C2)[0] is None:
                        types, r, w, C, g, W, done = t2, r2, w2, C2, g2, W2, True
                        break
                if done:
                    break
            if done:
                break
        for off in ((2e-4, 1e-3, 3e-3, 1e-2) if not done else ()):
            W2 = hi + off
            base = scaled(rl, lo, W2)
            rk = base[kr]
            mk = wl[kr]
            for sz in (
                2 * np.sqrt(off),
                3 * np.sqrt(off),
                1.5 * np.sqrt(off),
                4 * np.sqrt(off),
                0.05,
                0.1,
                0.2,
            ):
                for fl, fr, ml in ((1, 1, 0.5), (0.5, 1, 0.5), (1, 0.5, 0.5), (1, 1, 0.3), (1, 1, 0.7)):
                    rr = np.concatenate([np.delete(base, kr), [rk - fl * sz, rk + fr * sz]])
                    ww = np.concatenate([np.delete(wl, kr), [ml * mk, (1 - ml) * mk]])
                    t2 = list(np.delete(np.array(types), kr)) + ["r", "r"]
                    o = np.argsort(rr)
                    t2 = [t2[i] for i in o]
                    rr = rr[o]
                    ww = ww[o]
                    r2, w2, C2, ok2, g2 = solve(W2, t2, rr, ww, Cl)
                    if ok2 and event(g2, t2, r2, w2, C2)[0] is None:
                        types, r, w, C, g, W, done = t2, r2, w2, C2, g2, W2, True
                        break
                if done:
                    break
            if done:
                break
    if not done:
        say("STOP: could not continue past the event")
        break
    say(f"   now {len(types)} points")
    dW = 0.01
json.dump(out, open(OUT, "w"), indent=1)
say(f"done [{time.time() - T0:.0f}s]; events {len(out['events'])}")
