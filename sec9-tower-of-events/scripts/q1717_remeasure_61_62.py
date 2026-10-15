"""q1717 (10 Oct 2026, READING, no verdict word): re-measurement of the births K_b 62 and K_b 61 with the fast arb walker, the method of q1712 (copy with three changes: the events K_b 62 and 61 only, the start states searched also in ../capacity_even62_family and ../capacity_odd, the result file q1717_res.json). Stored amplitudes as in q1671: tower_K63_birth_out.json (A_cross) for K_b 62 and tower_K62_As2.json (A_s) for K_b 61.
Predictions registered in the registration text of 12:28 (before any re-measurement of these two events):
 P1 the shifts (new minus stored) of K_b 62 and 61 are below 5e-13 in absolute value, as for the other 19 events of q1712 (largest 4.986e-13);
 P2 the largest |a_pred - A| over K_b 61..82 then stays 1.055e-11 (K_b 61), evaluated by the same table as q1713 with the new amplitudes;
 Control per event: residual of the re-solved start state below 4.5e-55 and, for an odd family, a positive centre mass.
Usage: py q1717_remeasure_61_62.py (about 1 to 3 minutes, one process)."""

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
    for sub in (
        "K%d" % K,
        "K%d_stage" % K,
        "K%d_split" % K,
        "K%d_stage2" % K,
        os.path.join("..", "capacity_even62_family"),
        os.path.join("..", "capacity_odd"),
    ):
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
for K in (62, 61):
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
        out = os.path.join(here, "fw_lower_check2", "K%d" % K)
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
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(here, "q1717_res.json"), "w"), indent=1)
ok = {K: v for K, v in res.items() if v.get("status") == "ok"}
lab = lambda x: "holds" if x else "does not hold"
print(
    "re-measured %d of 2 events: %s"
    % (len(ok), ", ".join("K_b %d %+.3e" % (K, v["shift"]) for K, v in sorted(ok.items())))
)
print(
    "P1 %s; P2 is evaluated by the table of q1713 with the new amplitudes [%.0fs]"
    % (lab(len(ok) == 2 and all(abs(v["shift"]) < 5e-13 for v in ok.values())), time.time() - t00)
)
print("READING q1717 done (labels by the script; no verdict word)")
