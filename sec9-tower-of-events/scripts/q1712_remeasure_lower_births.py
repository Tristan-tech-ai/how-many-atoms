"""q1712 (10 Oct 2026, READING, no verdict word): independent re-measurement of the lower births K_b 61..82 with the fast arb walker at 192 bits, the way q1711 did K_b 73 (reading 171, +4.497e-12 from the stored tower value). Stored amplitudes: tower_K{K+1}_As2.json (field A_s) for odd K_b and tower_K{K+1}_birth_out.json (field A_cross) for even K_b, as in q1671. Start state: the stored state of the family K with the largest A not above the stored amplitude minus 0.003, from tower/K{K}, K{K}_stage, K{K}_split or K{K}_stage2. The walk goes to the stored amplitude plus 0.01 and stops at the located event (walk_family and locate only; after_split and after_birth are not run).
Predictions registered in the registration text of 12:14 (before any event other than K_b 73 was re-measured):
 P1 every re-measured amplitude differs from the stored one by less than 1e-11 in absolute value;
 P2 the signed shifts (new minus stored) are positive for most events and grow toward lower K;
 P3 after replacing the stored amplitudes by the re-measured ones the largest |a_pred - A| over the re-measured events falls below 1.055e-11 (q1671), evaluated by q1713.
Control per event (otherwise that event is VOID and listed): the start state re-solved with tolerance 1e-54 has a residual below 4.5e-55 (three evaluation floors) and, for an odd family, a positive centre mass.
Output: q1712_res.json (per K_b: stored, new, shift, bracket, control residual, time) and one line per event. The events are walked one after the other, K_b from 82 down to 61. Usage: py q1712_remeasure_lower_births.py (about 10 to 20 minutes in the background, one process).
"""

import os, sys, json, glob, time

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from flint import arb, ctx
import q1411_fast_walk as QW

ctx.prec = 192
A_DIR = os.path.join(here, "..", "scratch")
T = os.path.join(here, "..", "tower")


def stored(K):
    fn = (
        os.path.join(A_DIR, "tower_K%d_birth_out.json" % (K + 1))
        if K % 2 == 0
        else os.path.join(A_DIR, "tower_K%d_As2.json" % (K + 1))
    )
    if not os.path.exists(fn):
        return None
    d = json.load(open(fn))
    return d["A_cross"] if K % 2 == 0 else d["A_s"]


def start_state(K, a_stored):
    c = []
    for sub in ("K%d" % K, "K%d_stage" % K, "K%d_split" % K, "K%d_stage2" % K):
        for f in glob.glob(os.path.join(T, sub, "results_q220_gauss_K%d_A*.json" % K)):
            try:
                a = float(os.path.basename(f).split("_A")[1][:-5])
            except Exception:
                continue
            if a <= a_stored - 0.003:
                c.append((a, f))
    return max(c)[1] if c else None


res = {}
t00 = time.time()
for K in range(82, 60, -1):
    t0 = time.time()
    a_st = stored(K)
    if a_st is None:
        print("K_b %d: no stored amplitude" % K, flush=True)
        continue
    sf = start_state(K, a_st)
    if sf is None:
        print("K_b %d: no start state below the stored amplitude (stored %.12f)" % (K, a_st), flush=True)
        res[K] = dict(status="no start state")
        continue
    try:
        out = os.path.join(here, "fw_lower_check", "K%d" % K)
        W = QW.Walker(192, out, H=float(os.environ.get("Q1411_H", "0.05")))
        A, u, w, we, wc = QW.load(sf, W.F)
        u, w, we, wc, rs, k = W.solve_at(A, u, w, we, wc, tol="1e-54")
        ctl = bool(rs < arb("4.5e-55")) and (wc is None or bool(wc > 0))
        if not ctl:
            print(
                "K_b %d: VOID control (residual %.2e, wc %s)"
                % (K, float(rs), wc is not None and float(wc.mid())),
                flush=True,
            )
            res[K] = dict(status="void control", residual=float(rs))
            continue
        r = W.walk_family(A, u, w, we, wc, a_st + 0.01, hmax=0.01)
        if r[0] != "event":
            print("K_b %d: walk returned %s" % (K, r[0]), flush=True)
            res[K] = dict(status="no event: " + str(r[0]))
            continue
        _, A_lo, st_lo, s_lo, A_hi, st_hi, s_hi = r
        A_lo2, st_lo2, s_lo2, A_hi2, st_hi2, s_hi2, A_ev = W.locate((A_lo, st_lo, s_lo), (A_hi, st_hi, s_hi))
        shift = float((A_ev - arb(repr(a_st))).mid())
        res[K] = dict(
            status="ok",
            stored=a_st,
            new=A_ev.str(20, radius=False),
            shift=shift,
            bracket=float((A_hi2 - A_lo2).mid()),
            residual=float(rs),
            start=os.path.basename(sf),
            seconds=time.time() - t0,
        )
        print(
            "K_b %d: stored %.12f, new minus stored %+.3e, bracket %.2e, control residual %.1e, start %s [%.0fs]"
            % (K, a_st, shift, res[K]["bracket"], float(rs), os.path.basename(sf), time.time() - t0),
            flush=True,
        )
    except Exception as e:
        print("K_b %d: exception %s: %s" % (K, type(e).__name__, str(e)[:120]), flush=True)
        res[K] = dict(status="exception " + type(e).__name__)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(here, "q1712_res.json"), "w"), indent=1)
ok = {K: v for K, v in res.items() if v.get("status") == "ok"}
lab = lambda x: "holds" if x else "does not hold"
sh = {K: v["shift"] for K, v in ok.items()}
p1 = bool(sh) and all(abs(s) < 1e-11 for s in sh.values())
pos = sum(1 for s in sh.values() if s > 0)
lowK = [sh[K] for K in sorted(sh) if K <= 70]
hiK = [sh[K] for K in sorted(sh) if K > 70]
p2 = (
    bool(sh)
    and pos > len(sh) / 2
    and bool(lowK)
    and bool(hiK)
    and (sum(lowK) / len(lowK)) > (sum(hiK) / len(hiK))
)
print(
    "re-measured %d of 22 events; shifts positive %d; mean shift K_b <= 70: %s, K_b > 70: %s"
    % (
        len(ok),
        pos,
        ("%+.2e" % (sum(lowK) / len(lowK))) if lowK else "n/a",
        ("%+.2e" % (sum(hiK) / len(hiK))) if hiK else "n/a",
    )
)
print("P1 %s; P2 %s; P3 is evaluated by q1713 [%.0fs]" % (lab(p1), lab(p2), time.time() - t00))
print("READING q1712 done (labels by the script; no verdict word)")
