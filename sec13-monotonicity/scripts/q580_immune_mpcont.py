"""Q580: 200-bit continuation in a of the symmetric uniform immune optimum (q579's arb polish and signals), written 30 Sep 17:18:36
(date stamp), from PLAN_192BIT_IMMUNE.md. Start: the certified a = 23.5 state (q556 v3, K = 29, centre atom). Step DA = 0.05:
polish the current structure at a + DA from the previous state; transition signal = s0 = g(0)/C - 1 without a centre atom (APPEAR
when positive) or s2 = g''(0)/C with one (SPLIT when positive); the off-support maximum is checked at every accepted row (a positive
value off the centre would contradict the centre-only birth rule and stops the run). At a sign change the OLD family's signal is
bisected to 1e-5 in a; the new family is seeded at a_t + 0.01 (APPEAR: centre masses 1e-4, 1e-3, 1e-2; SPLIT: centre replaced by a pair
at +-0.05, +-0.1, +-0.2) and the first candidate that polishes (max|F| < 1e-35, weights > 0) with negative signals is taken.
Row acceptance: max|F| < 1e-35, all masses > 0, transition signal < 0, off-support maximum < 0.
PRE-REGISTERED USE: the rows at a = 24, 25, ..., 30 score q577 (offset K - 0.4002 a^(4/3) in [1.3, 2.9], mean within 0.4 of 2.03).
Output: results_q580.json (rows: a, K, centre, positions, masses, s0/s2, off, max|F|; transitions), log q580.log.
v5 (18:03): every step takes a double-precision predictor (q508c spolish, descent if needed) before the arb corrector
(v4 spent minutes per failed arb polish from the previous state). v4 (17:57): on a failed polish, short steps 0.01, 0.005, 0.0025, 0.001 before stopping (v3 stopped at 23.75, 2.5e-6).
v3 (17:53): split seeds +-0.7, +-0.5, +-1.0, +-0.3 and appear masses up to 0.1 (the post-split pair at a_t + 0.02 converged only from +-0.7:
arb 3.3e-51, signals -2.2e-12). v2: a new family is first brought within ~1e-8 by the double pipeline (q556's L-BFGS descent, then q508c's symmetric polish)
and only then polished in arb; split seeds at +-0.3, +-0.5, +-0.15 and appear seeds at 1e-3, 1e-2, 3e-2, at a_t + 0.02 (v1: raw seeds
at a_t + 0.01 stalled in arb Newton at 2e-8 for +-0.05, 0.1, 0.2). RESUME=1 restarts from the last row of results_q580.json.
Usage: py q580_immune_mpcont.py [AMAX]"""

import sys, json, time
import numpy as np

AMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
sys.argv = [sys.argv[0]]
import q579_immune_mp as M
import os

sys.argv = [sys.argv[0], "24.0", "unif"]
import q556_symfixed as Q6

sys.argv = [sys.argv[0]]
S = Q6.S
from flint import arb

say = lambda *a: print(*a, flush=True)
t0 = time.time()
DA = 0.05


def run_polish(a, p, v, centre):
    x6, q6 = Q6.setup(a)
    pd2, vd2, rd = S.spolish(np.array(p, float), np.array(v, float), centre, x6, q6)  # v5: double predictor
    if not (rd < 1e-6 and np.all(np.diff(np.concatenate([[0.0], pd2])) > 0) and np.all(vd2 > 0)):
        yf, wf = S.full(np.array(p, float), np.array(v, float), centre)
        yf, wf = Q6.descend(yf, wf, x6, q6, iters=20000)
        pdd, vdd, cdd = Q6.symmetrise(yf, wf)
        if cdd == centre and len(pdd) == len(p):
            pd2, vd2, rd = S.spolish(pdd, vdd, centre, x6, q6)
    if rd < 1e-6 and len(pd2) == len(p):
        p, v = list(pd2), list(vd2)
    I = M.IMP(a, 9)
    pp, vv, r = I.polish(list(p), list(v), centre)
    pf = [float(t.mid()) for t in pp]
    vf = [float(t.mid()) for t in vv]
    ok = float(r) < 1e-35 and min(vf) > 0 and (not centre or 1 - sum(vf) > 0) and all(np.diff([0.0] + pf) > 0)
    return I, pp, vv, pf, vf, float(r), ok


def sig(I, pp, vv, centre):
    s0, s2, off = I.signals(pp, vv, centre)
    return float(s0.mid()), float(s2.mid()), float(off)


def tsig(centre, s0, s2):
    return s2 if centre else s0


if __name__ == "__main__":
    if os.environ.get("RESUME"):
        last = json.load(open("results_q580.json"))["rows"][-1]
        p0, v0, centre, a = np.array(last["p"]), np.array(last["v"]), bool(last["centre"]), float(last["a"])
        say(f"resumed at a = {a} from results_q580.json")
    else:
        p0, v0, centre = M.seed_from("results_q556_unif0.0_a23.5.json")
        a = 23.5
    I, pp, vv, pf, vf, r, ok = run_polish(a, p0, v0, centre)
    s0, s2, off = sig(I, pp, vv, centre)
    say(
        f"start a = {a}: K = {2*len(pf) + int(centre)}, centre {centre}, max|F| {r:.1e}, s0 {s0:+.2e}, s2 {s2:+.2e}, off {off:+.2e}, ok {ok}   [{time.time() - t0:.0f}s]"
    )
    rows = []
    trans = []
    if os.environ.get("RESUME"):
        old = json.load(open("results_q580.json"))
        rows, trans = old["rows"][:-1], old["transitions"]

    def save():
        json.dump({"rows": rows, "transitions": trans}, open("results_q580.json", "w"), indent=1)

    def add_row(a, pf, vf, centre, s0, s2, off, r):
        K = 2 * len(pf) + int(centre)
        rows.append(
            {"a": a, "K": K, "centre": centre, "p": pf, "v": vf, "s0": s0, "s2": s2, "off": off, "res": r}
        )
        save()
        if abs(a - round(a)) < 1e-9:
            say(f"   SCORE ROW a = {a}: K = {K}, offset {K - 0.4002*a**(4/3):.3f}")

    add_row(a, pf, vf, centre, s0, s2, off, r)
    while a < AMAX - 1e-9:
        an = round((np.floor(a / DA + 1e-9) + 1) * DA, 6)
        In, ppn, vvn, pfn, vfn, rn, okn = run_polish(an, pf, vf, centre)  # v3: snap to the DA grid
        if not okn:  # v4: shorter steps when the polish fails
            for sub in (0.01, 0.005, 0.0025, 0.001):
                an = round(a + sub, 6)
                In, ppn, vvn, pfn, vfn, rn, okn = run_polish(an, pf, vf, centre)
                if okn:
                    say(f"   short step {sub} to a = {an}")
                    break
        if not okn:
            say(f"STOP at a = {an}: polish failed (max|F| {rn:.1e})")
            break
        s0n, s2n, offn = sig(In, ppn, vvn, centre)
        if tsig(centre, s0n, s2n) <= 0:
            if offn > 0:
                say(
                    f"STOP at a = {an}: off-support maximum {offn:+.2e} with the centre signal negative (contradicts the birth rule)"
                )
                break
            a, pf, vf = an, pfn, vfn
            add_row(a, pf, vf, centre, s0n, s2n, offn, rn)
            say(
                f"a = {a}: K = {2*len(pf) + int(centre)}, sig {tsig(centre, s0n, s2n):+.2e}, off {offn:+.2e}, |F| {rn:.0e}   [{time.time() - t0:.0f}s]"
            )
            continue
        lo, hi, plo, vlo = a, an, pf, vf
        while hi - lo > 1e-5:
            mid = round(0.5 * (lo + hi), 8)
            Im, ppm, vvm, pfm, vfm, rm, okm = run_polish(mid, plo, vlo, centre)
            if not okm:
                say(f"   bisection polish failed at {mid}")
                break
            s0m, s2m, _ = sig(Im, ppm, vvm, centre)
            if tsig(centre, s0m, s2m) > 0:
                hi = mid
            else:
                lo, plo, vlo = mid, pfm, vfm
        kind = "split" if centre else "appear"
        at = 0.5 * (lo + hi)
        trans.append({"a": at, "lo": lo, "hi": hi, "kind": kind, "K_before": 2 * len(pf) + int(centre)})
        say(f"TRANSITION {kind} at a in ({lo:.6f}, {hi:.6f}]   [{time.time() - t0:.0f}s]")
        a1 = round(hi + 0.02, 6)
        got = False
        cands = (
            [("c", m) for m in (1e-3, 1e-2, 3e-2, 0.1)]
            if not centre
            else [("d", d) for d in (0.7, 0.5, 1.0, 0.3)]
        )
        x6, q6 = Q6.setup(a1)
        for typ, par in cands:
            if typ == "c":
                p1 = np.array(plo)
                v1 = np.array(vlo) * (1 - par)
                c1 = True
            else:
                vc = 1 - sum(vlo)
                p1 = np.concatenate([[par], plo])
                v1 = np.concatenate([[vc], vlo])
                c1 = False
            yf, wf = S.full(p1, v1, c1)
            yf, wf = Q6.descend(yf, wf, x6, q6, iters=20000)
            pd_, vd_, cd_ = Q6.symmetrise(yf, wf)
            if cd_ != c1 or len(pd_) != len(p1):
                say(f"   candidate {typ}={par}: the descent changed the structure; skipped")
                continue
            pd2, vd2, rd = S.spolish(pd_, vd_, cd_, x6, q6)
            say(f"   candidate {typ}={par}: double stage residual {rd:.1e}")
            I1, pp1, vv1, pf1, vf1, r1, ok1 = run_polish(a1, list(pd2), list(vd2), c1)
            if not ok1:
                say(f"   candidate {typ}={par}: arb polish failed ({r1:.1e})")
                continue
            s01, s21, off1 = sig(I1, pp1, vv1, c1)
            if tsig(c1, s01, s21) <= 0 and off1 <= 0:
                a, pf, vf, centre = a1, pf1, vf1, c1
                add_row(a, pf, vf, centre, s01, s21, off1, r1)
                got = True
                say(
                    f"   new family at a = {a}: K = {2*len(pf) + int(centre)}, sig {tsig(centre, s01, s21):+.2e}, off {off1:+.2e}, |F| {r1:.0e}"
                )
                break
            say(f"   candidate {typ}={par}: signals {tsig(c1, s01, s21):+.2e}, off {off1:+.2e}")
        if not got:
            say("STOP: no new family certified")
            break
    save()
    say(f"END at a = {a}   [{time.time() - t0:.0f}s]")
