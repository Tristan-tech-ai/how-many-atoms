"""q686 (Amendment 1027): births of the optimal input on the shell a <= |x| <= a + W at fixed inner radius a, by continuation in the
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
OUT = f"q686_annulus_a{a:g}_d{DIM:g}_SEALED.json"


def solve(W, types, r, w, C):
    g = L.Grid(a + W, ain=a, dim=DIM)
    L.AIN[0] = a
    r = np.array(r, float)
    r[[j for j, t in enumerate(types) if t == "w"]] = a + W
    r[[j for j, t in enumerate(types) if t == "v"]] = a
    x, nrm, it, ok = L.newton(g, L.pack(r, w, C, types), types, tol=1e-13, maxit=100)
    rr, ww, CC = L.unpack(x, types, a + W, a)
    ok = ok and np.all(ww > 0) and np.all(np.diff(rr) > 0)
    return rr, ww, CC, bool(ok), g


def event(g, r, w, C):
    vmax, rmax, _, _ = L.global_check(g, r, w, C)
    if rmax is not None and np.isfinite(vmax):
        rq = np.linspace(max(rmax - 0.05, g.ain), min(rmax + 0.05, g.A), 201)
        v = g.kkt(rq, r, w) - C
        k = int(np.argmax(v))
        rc = rq[k]
        if np.min(np.abs(np.asarray(r) - rc)) < 0.1 or v[k] <= 1e-10:
            return None, None
        return float(v[k]), float(rc)
    return None, None


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
marks = [float(k) for k in range(int(W) + 1, int(WMAX) + 1)]
dW = 0.05
while W < WMAX - 1e-12:
    W1 = min(W + dW, WMAX)
    for mk in marks:
        if W < mk < W1:
            W1 = mk
    r1, w1, C1, ok1, g1 = solve(W1, types, scaled(r, W, W1), w, C)
    if not ok1:
        dW /= 2
        if dW < 1e-7:
            say("STOP: continuation failed")
            break
        continue
    v, rc = event(g1, r1, w1, C1)
    if v is None:
        W, r, w, C, g = W1, r1, w1, C1, g1
        dW = min(dW * 1.5, 0.05)
        if any(abs(W - mk) < 1e-12 for mk in marks):
            out["states"].append({"W": W, "types": types, "r": r.tolist(), "w": w.tolist(), "C": C})
            json.dump(out, open(OUT, "w"), indent=1)
            say(f"mark {len(out['states'])}: {len(types)} points  [{time.time() - T0:.0f}s]")
        continue
    lo, hi = W, W1
    rl, wl, Cl = r.copy(), w.copy(), C
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        rm, wm, Cm, okm, gm = solve(mid, types, scaled(rl, lo, mid), wl, Cl)
        if not okm:
            break
        vm, _ = event(gm, rm, wm, Cm)
        if vm is not None:
            hi = mid
        else:
            lo, rl, wl, Cl = mid, rm, wm, Cm
    gl = L.Grid(a + lo, ain=a, dim=DIM)
    L.AIN[0] = a
    _, rb = event(L.Grid(a + hi, ain=a, dim=DIM), *solve(hi, types, scaled(rl, lo, hi), wl, Cl)[:3])
    out["events"].append(
        {
            "kind": "interior",
            "W_lo": lo,
            "W_hi": hi,
            "r_birth": rb,
            "n_before": len(types),
            "types": list(types),
            "r": rl.tolist(),
            "w": wl.tolist(),
            "C": float(Cl),
        }
    )
    json.dump(out, open(OUT, "w"), indent=1)
    say(f"event {len(out['events'])} bisected  [{time.time() - T0:.0f}s]")
    done = False
    rb = rb if rb is not None else rc
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
            if ok2 and event(g2, r2, w2, C2)[0] is None:
                types, r, w, C, g, W, done = t2, r2, w2, C2, g2, W2, True
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
