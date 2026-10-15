"""q942 (7 Oct 2026): text copy of q941 with a cubic (four-state Lagrange) predictor when four states exist (amendment 1721). q941: text copy of q932 with a quadratic (three-state Lagrange) predictor instead of the tangent (amendment 1718). Text of q932: q932 (7 Oct 2026): text copy of q923 for both parities (an even state has no centre atom). Trace a capacity family in A with the predictor-corrector of q921 (instrument for T61, recorded in an amendment).
From a saved odd state (results_q220_gauss_K61_A<A>.json: fields A, u, w, we, wc) the amplitude is raised by dA; at each step the predictor du/dA = -J^-1 b and the damped Newton corrector of q921.newton_full give the new state in 200-bit arithmetic,
its centre and off-atom signals come from Cap.signals, and the verdict follows the rule of 1697 (OPTIMAL: full residual below 1e-20, ordered atoms, positive masses, centre and off-atom signals not positive).
A failed corrector halves dA (minimum 1e-4); a converged state that is not OPTIMAL is saved and ends the trace (the family is over: the split of the centre atom).
Usage: py q942_trace_cubic.py BASE.json OUT_DIR dA A_stop"""

import os, sys, time, copy, glob
from flint import arb
import q920_cap_flint as F
import q921_cap_tools as T

t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
f, out_dir, dA, A_stop = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
A_str, u, w, we, wc, d = F.load_state(f)
odd = wc is not None
m = len(u)
K = 2 * m + 2 + (1 if odd else 0)
GRID_A = d.get("grid_A", None) or None
if len(sys.argv) > 5:
    GRID_A = sys.argv[5]  # optional override: a longer frozen grid for an extension of a trace
if GRID_A is None:
    GRID_A = f"{A_stop}"  # frozen grid of the trace: the grid of q206 for A_stop, valid for every A up to A_stop (the grid length of q206 changes with A, which broke the correctors of the first trace)
cap = F.Cap(A_str, m, odd, grid_A=GRID_A)
v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
A = float(A_str)
os.makedirs(out_dir, exist_ok=True)
v, res0, ok0 = T.newton_full(cap, v, say, tol=arb("1e-45"))
say(f"base polished on the frozen grid {GRID_A} ({cap.N} points): {ok0}, residual {F.maxabs(res0)}")
if not ok0:
    sys.exit("base does not polish on the frozen grid")
cen0, off0 = cap.signals(v)
ord0 = all(v[i] < v[i + 1] for i in range(m - 1))
wmin0 = min(v[m : 2 * m] + [v[2 * m]] + ([v[2 * m + 1]] if odd else []))
verd0 = (
    "OPTIMAL"
    if (ord0 and wmin0 > 0 and cen0 <= 0 and off0 <= 0 and F.maxabs(res0) < arb("1e-20"))
    else "other"
)
F.dump_state(
    os.path.join(out_dir, f"results_q220_gauss_K{K}_A{A:.4f}.json"),
    A_str,
    cap,
    v,
    res0,
    verd0,
    {
        "centre_signal": float(cen0.mid()),
        "off_atom_max": float(off0.mid()),
        "w_min": float(wmin0.mid()),
        "ordered": ord0,
        "grid_A": GRID_A,
    },
)  # the polished base is the first state of the family (amendment 1712)
say(
    f"base saved as the first state: verdict {verd0}, centre signal {cen0.str(5, radius=False)}, off-atom {off0.str(5, radius=False)}"
)
say(f"trace from {f}: K {K}, A {A_str}, dA {dA}, stop {A_stop}")
import json


def prime(out_dir, K, A_base):
    files = sorted(glob.glob(os.path.join(out_dir, f"results_q220_gauss_K{K}_A*.json")))
    out = []
    for fp in files:
        A_s, u_, w_, we_, wc_, d_ = F.load_state(fp)
        if float(A_s) < float(A_base) - 1e-9:
            out.append((arb(A_s), [t.mid() for t in (u_ + w_ + [we_] + ([wc_] if wc_ is not None else []))]))
    return out[-3:]


hist = prime(out_dir, K, A) + [(arb(A_str), [t for t in v])]  # the last state is the polished base


def step_quad(cap_old, v_, A_new_str):
    if len(hist) < 3:
        return T.step_in_A(cap_old, v_, A_new_str, say)
    npt = 4 if len(hist) >= 4 else 3
    cap_new = copy.copy(cap_old)
    cap_new.A = arb(A_new_str)
    xs = [h[0] for h in hist[-npt:]]
    ys = [h[1] for h in hist[-npt:]]
    x = cap_new.A
    vp = []
    for c in range(len(v_)):
        tot = arb(0)
        for i in range(npt):
            li = arb(1)
            for j in range(npt):
                if j != i:
                    li = li * (x - xs[j]) / (xs[i] - xs[j])
            tot = tot + li * ys[i][c]
        vp.append(tot.mid())
    vn_, rn_, ok_ = T.newton_full(cap_new, vp, say)
    return cap_new, vn_, rn_, ok_


last_ok = dA  # instrument repair of 7 Oct (amendment 1715): a new step starts at 1.6 times the last accepted step, at most dA, instead of dA
while A < A_stop - 1e-9:
    step = min(dA, last_ok * 1.6)
    while True:
        A_new = round(A + step, 6)
        A_new_str = f"{A_new:.6f}"
        cap_new, vn, res, ok = step_quad(cap, v, A_new_str)
        if ok or step <= 1.5e-4:
            break
        step /= 2
        say(f"   corrector failed at A = {A_new_str}; dA -> {step}")
    cen, off = cap_new.signals(vn)
    ordered = all(vn[i] < vn[i + 1] for i in range(m - 1))
    wmin = min(vn[m : 2 * m] + [vn[2 * m]] + ([vn[2 * m + 1]] if odd else []))
    rmax = F.maxabs(res)
    optimal = ok and ordered and wmin > 0 and cen <= 0 and off <= 0 and rmax < arb("1e-20")
    verdict = (
        "OPTIMAL"
        if optimal
        else (
            "not converged"
            if not ok
            else (
                "centre positive" if not (cen <= 0) else ("off-atom positive" if not (off <= 0) else "other")
            )
        )
    )
    say(
        f"A = {A_new_str}: ok {ok}, residual {rmax.str(3, radius=False)}, wc {(vn[2 * m + 1].str(8, radius=False) if odd else "none")}, w1 {vn[m].str(8, radius=False)}, centre signal {cen.str(5, radius=False)}, off-atom {off.str(5, radius=False)}, ordered {ordered} -> {verdict}"
    )
    if ok:
        F.dump_state(
            os.path.join(out_dir, f"results_q220_gauss_K{K}_A{A_new:.4f}.json"),
            A_new_str,
            cap_new,
            vn,
            res,
            verdict,
            {
                "centre_signal": float(cen.mid()),
                "off_atom_max": float(off.mid()),
                "w_min": float(wmin.mid()),
                "ordered": ordered,
                "grid_A": GRID_A,
            },
        )
    if not optimal:
        print(f"RESULT q941: trace ends at A = {A_new_str} with verdict {verdict}", flush=True)
        sys.exit(0 if ok else 1)
    cap, v, A = cap_new, vn, A_new
    last_ok = step
    hist.append((arb(A_new_str), [t for t in vn]))
print(f"RESULT q941: reached A = {A:.4f} (stop {A_stop}) with OPTIMAL states", flush=True)
