"""Q220: continuation in A of a certified Gaussian state (Amendment 308 NEXT candidate (b) for the soft wall): from
the state file, step A by dA (uniform compression x -> x A_new/A_old of the interior atoms as predictor, weights
kept), Newton with the SOLVER of the environment (svdls recommended), certify, save, repeat. Each step is small
enough that the soft modes stay in their linear regime, which the law seeds do not guarantee.
Usage: py q220_walk_A.py <state.json> <A_to> <dA> [dps]"""

import os, sys, json, time
from mpmath import mp, mpf, log, pi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q206_gauss_mp as G

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
t0 = time.time()
f, A_to, dA = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
mp.dps = int(sys.argv[4]) if len(sys.argv) > 4 else 40
G.L2P = log(2 * pi) / 2
clean = lambda v: mpf(str(v).replace("np.float64(", "").replace(")", "").strip())
d = json.load(open(os.path.join(HERE, f)))
d = d[-1] if isinstance(d, list) else d
A = clean(d["A"])
u = [clean(v) for v in d["u"]]
w = [clean(v) for v in d["w"]]
we = clean(d["we"])
wc = None if d.get("wc") in (None, "None") else clean(d["wc"])
K = 2 * len(u) + 2 + (1 if wc is not None else 0)
say(
    f"walk K = {K} from A = {mp.nstr(A, 8)} (verdict {d.get('verdict')}) to {A_to} by {dA}; SOLVER={G.SOLVER}"
)
step = mpf(repr(-abs(dA))) if A_to < float(A) else mpf(repr(abs(dA)))
while (step < 0 and float(A) + float(step) >= A_to - 1e-9) or (
    step > 0 and float(A) + float(step) <= A_to + 1e-9
):
    A_new = A + step
    u = [x * A_new / A for x in u]
    A = A_new
    y = G.grid(A)
    uu, ww, wwe, wwc, res, k = G.newton(A, y, u, w, we, wc)
    ok = res < mpf("1e-20")
    cen, off, where, C = G.signals(A, y, uu, ww, wwe, wwc)
    ordered = all(uu[i] < uu[i + 1] for i in range(len(uu) - 1))
    wmin = min(list(ww) + [wwe] + ([wwc] if wwc is not None else []))
    optimal = ok and ordered and wmin > 0 and cen <= 0 and off <= 0
    verdict = (
        "OPTIMAL"
        if optimal
        else ("not converged" if not ok else ("centre positive" if cen > 0 else "off-atom positive"))
    )
    say(
        f"A = {mp.nstr(A, 8)}, K = {K}: residual {mp.nstr(res, 3)} in {k} steps, min w {mp.nstr(wmin, 4)}, ordered {ordered}, centre signal {mp.nstr(cen, 4)}, off-atom max {mp.nstr(off, 4)} -> {verdict}   [{time.time()-t0:.0f}s]"
    )
    out = {
        "A": mp.nstr(A, 40),
        "K": K,
        "residual": float(res),
        "verdict": verdict,
        "centre_signal": float(cen),
        "off_atom_max": float(off),
        "w_min": float(wmin),
        "ordered": ordered,
        "u": [mp.nstr(v, 40) for v in uu],
        "w": [mp.nstr(v, 40) for v in ww],
        "we": mp.nstr(wwe, 40),
        "wc": (None if wwc is None else mp.nstr(wwc, 40)),
    }
    json.dump(out, open(os.path.join(HERE, f"results_q220_gauss_K{K}_A{float(A):.4f}.json"), "w"), indent=1)
    if not ok:
        say("   stop")
        break
    u, w, we, wc = list(uu), list(ww), wwe, wwc
