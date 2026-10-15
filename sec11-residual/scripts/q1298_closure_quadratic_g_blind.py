"""q1298 (8 Oct 2026, registered in amendment 2076): blind test of the closure R_k = A (k + k0) Da(v^2) [1 + (a0 + a1 k + a2 k^2) / v^2] (model M2 of q1297, parameters fixed on the 243 seen events of k = 0.75 to 12) for the new exponents k = 6, 10, 16 at K_b 14 to 28 and 37 to 40 (19 events per exponent).
Da(v^2) = |Im S_+(1/v^2)| / (2 rho) in 1e-9, rho = v / (4 pi), [8/8] approximant of e_2 .. e_18, theta 0.3 (q1285). v^2 is that of the event in the quintic half-line N_half at a_med.
Items: L6, L10: at least 15 of the 19 events located and |R / R_pred - 1| <= 0.05 at every located event; L16: the same with 0.15. Class PASS if all three; INCOMPLETE if for some exponent fewer than 15 are located; otherwise FAIL.
Located: a_t not null, last residual below 1e-18, |a_t - last a| < 1e-9.
--registration-run : controls, table q1298_predictions.json (sha256), launch scripts run1298a/b/c.sh; --read : items; last line 'VERDICT q1298 quadratic-g closure class: ...'.
Controls (amendment 2076): c1 item functions; c2 Da(115) = 0.21082; c3 the law reproduces the seen events of the ten exponents (q1296 data) within 0.05 in the ratio at every event and the rms of ln R equals 0.0066 within 0.0003; c4 power: for k = 10 and 16 the law differs from the linear laws of amendment 2072 by more than the item bar at 10 or more of the 19 events; c5 (repaired in amendment 2077) quintic against septic a_med per exponent at most one fifth of the item half-width (0.010, 0.010, 0.030) with a synthetic test (twice each bar fails, 0.9 of each passes); and the items are evaluated with both interpolants; c6 D = 80 against 92 within 0.5 percent of R_pred; c7 chain solver bit-identical at k = 2; c8 the stage of each exponent converged at D = 92 with G = 1 to 5e-5.
Stage files: c33_k6c.json, c33_k10c.json by c36; for k = 16 the derived-law run of c36 did not converge at D = 20 (amendment 2077), the stage c33_k16d.json is the run of c37_halfline_seed.py (the c34 text with a seeded first stage).
Usage: py q1298_closure_quadratic_g_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import hashlib, io, contextlib, json, os, re, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import mpmath as mp

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1250_quintic_residual as Q
    import q1271_linear_in_k_blind as Z
    import q1273_k4_curvature_blind as W
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
Bf = T.machine(T.coef_series(), 8, 8)[0]
A_, K0_, a0_, a1_, a2_ = 2.60436356, 0.17144331, -11.10486513, -2.71708651, 0.11885015
TARGET = list(range(14, 29)) + list(range(37, 41))
NEED = 15
TOL = {6.0: 0.05, 10.0: 0.05, 16.0: 0.15}
C5BAR = {6.0: 0.010, 10.0: 0.010, 16.0: 0.030}
STAGE = {6.0: ("c33_k6c.json", "k6"), 10.0: ("c33_k10c.json", "k10"), 16.0: ("c33_k16d.json", "k16")}
ITEM = {6.0: "L6", 10.0: "L10", 16.0: "L16"}


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


def sig(k):
    return A_ * (k + K0_)


def gk(k):
    return a0_ + a1_ * k + a2_ * k**2


def rpred(k, v2):
    return sig(k) * Da(v2) * (1 + gk(k) / v2)


def rlin(k, v2):
    return 2.616046 * (k + 0.140737) * Da(v2) * (1 + (-10.1349 - 2.4524 * k) / v2)


def locr(f):
    try:
        j = json.load(open(f + ".json"))
        hl = [x for x in j["hist"] if "res" in x and "a" in x]
        a = j.get("a_t")
        return a if (a is not None and hl and hl[-1]["res"] < 1e-18 and abs(a - hl[-1]["a"]) < 1e-9) else None
    except Exception:
        return None


def item(k, r):
    return abs(r - 1) <= TOL[k]


def c5ok(k, w):
    return w <= C5BAR[k]


def lawfrom(tag):
    t = open(f"c36_{tag}.out", encoding="utf-8").read()
    m = re.findall(r"fitted law numbers \[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]", t)
    if not m:
        raise SystemExit(f"READING q1298: VOID (no fitted law numbers in c36_{tag}.out)")
    return tuple(float(x) for x in m[-1])


def nh(k, deg, law, D_=92.0):
    return W.nhalf(STAGE[k][0], D_, deg, law)


def stage_ok(k):
    st = [s for s in json.load(open(STAGE[k][0]))["stages"] if s["conv"] and s["D"] == 92.0]
    return bool(st) and abs(st[-1]["G"] - 1.0) < 5e-5


def regrun():
    c1 = (
        item(6.0, 1.0)
        and item(6.0, 0.96)
        and not item(6.0, 1.07)
        and not item(6.0, 0.93)
        and item(10.0, 1.0)
        and item(10.0, 0.96)
        and not item(10.0, 1.07)
        and not item(10.0, 0.93)
        and item(16.0, 1.12)
        and not item(16.0, 1.20)
    )
    c2 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
    with contextlib.redirect_stdout(buf):
        import q1296_closure_forms_all_k as C
    ys = []
    dev = 0.0
    for k in sorted(C.data):
        for K, r, v in C.data[k]:
            rp = rpred(k, v)
            dev = max(dev, abs(r / rp - 1))
            ys.append(np.log(r / rp))
    rms = float(np.sqrt(np.mean(np.array(ys) ** 2)))
    c3 = dev <= 0.05 and abs(rms - 0.0066) <= 0.0003
    c5neg = all(not c5ok(k, 2 * C5BAR[k]) and c5ok(k, 0.9 * C5BAR[k]) for k in STAGE)
    c7 = (
        abs(json.load(open("q607_ctl_k2_K30.json"))["a_t"] - json.load(open("c19_u1231_K30.json"))["a_t"])
        < 1e-12
    )
    c8 = all(stage_ok(k) for k in STAGE)
    out = {}
    w5 = {}
    w6 = {}
    npow = {}
    for k in STAGE:
        law = lawfrom(STAGE[k][1])
        N5 = nh(k, 5, law)
        N7 = nh(k, 7, law)
        N80 = nh(k, 5, law, 80.0)
        out[str(k)] = {"law": list(law)}
        w5[k] = w6[k] = 0.0
        npow[k] = 0
        for K in TARGET:
            am = Q.root(N5, K)
            v2 = Z.v2f(N5, am)
            rp = rpred(k, v2)
            w5[k] = max(w5[k], abs(am - Q.root(N7, K)) * 1e9 / rp)
            w6[k] = max(w6[k], abs(am - Q.root(N80, K)) * 1e9 / rp)
            if abs(rp / rlin(k, v2) - 1) > TOL[k]:
                npow[k] += 1
            out[str(k)][str(K)] = dict(a_med=am, v2=float(v2), R_pred=rp, a0=float(am + rp * 1e-9 - 0.0045))
        print(
            f"k {k}: law {law}; v^2 {out[str(k)]['14']['v2']:.2f} to {out[str(k)]['40']['v2']:.2f}; predicted R (1e-9): K_b 14 {out[str(k)]['14']['R_pred']:.2f}, 20 {out[str(k)]['20']['R_pred']:.3f}, 28 {out[str(k)]['28']['R_pred']:.4f}, 40 {out[str(k)]['40']['R_pred']:.5f}; sigma {sig(k):.3f}, g {gk(k):.2f}"
        )
    c4 = npow[10.0] >= 10 and npow[16.0] >= 10
    c5 = c5neg and all(c5ok(k, w5[k]) for k in STAGE)
    c6 = max(w6.values()) <= 0.005
    print(
        f"control c1 {c1}; c2 {c2}; c3 (law reproduces seen events: max |ratio - 1| {dev:.4f}, rms ln R {rms:.4f}) {c3}; c4 (events differing from the linear law by more than the bar: {npow}) {c4}; c5 (quintic vs septic, max relative per k {', '.join(f'{k}: {w5[k]:.5f}' for k in STAGE)}; synthetic test {c5neg}) {c5}; c6 (D = 80 vs 92, {max(w6.values()):.2e}) {c6}; c7 {c7}; c8 (stages converged at D = 92, G = 1) {c8}"
    )
    if not (c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8):
        print("REGISTRATION-RUN q1298: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1298_predictions.json", "w"), indent=1)
    for letter, k in zip("abc", STAGE):
        tag = STAGE[k][1]
        law = out[str(k)]["law"]
        L = [
            "#!/bin/bash",
            f"# run1298{letter}.sh (8 Oct 2026): full-chain locators c35 for k = {k} at K_b 14-28 and 37-40 (amendments 2076 and the record amendment); starts from q1298_predictions.json",
            'cd "./centre_scripts"',
            f"export KEXP={k} HL_JSON={STAGE[k][0]} HL_STAGE=92 D0={law[0]:.5f} NU0={law[1]:.5f}",
        ]
        for K in TARGET:
            L.append(
                f"py c35_chain_k.py {K} {K // 2} 160 {out[str(k)][str(K)]['a0']:.6f} 0.003 u1298_{tag}_K{K} > q607_u1298_{tag}_K{K}.log 2>&1"
            )
        L.append(f"echo done $(date +%H:%M:%S) > run1298{letter}.done")
        open(f"run1298{letter}.sh", "w", newline="\n").write("\n".join(L) + "\n")
    print("sha256", sh)
    print("REGISTRATION-RUN q1298: controls True, table and run1298a/b/c.sh written")


def read():
    js = json.load(open("q1298_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1298: VOID (table hash)")
        sys.exit(2)
    res = {}
    incomplete = False
    for k in STAGE:
        law = tuple(js[str(k)]["law"])
        N5 = nh(k, 5, law)
        N7 = nh(k, 7, law)
        rows = []
        rows7 = []
        chg = 0.0
        for K in TARGET:
            a = locr(f"q607_u1298_{STAGE[k][1]}_K{K}")
            if a is None:
                print(f"k {k} K_b {K}: NOT LOCATED")
                continue
            v2 = Z.v2f(N5, a)
            r = (a - js[str(k)][str(K)]["a_med"]) * 1e9
            rp = rpred(k, v2)
            rows.append((K, r / rp))
            r7 = (a - Q.root(N7, K)) * 1e9 / rp
            rows7.append((K, r7))
            chg = max(chg, abs(r7 - r / rp))
            print(f"k {k} K_b {K}: R {r:+.5f}e-9, R_pred {rp:.5f}, ratio {r / rp:.4f}")
        print(f"k {k}: Reading, largest change of the ratio with the septic a_med: {chg:.5f}")
        if len(rows) < NEED:
            incomplete = True
            print(f"k {k}: located {len(rows)} of {len(TARGET)}")
            continue
        ok5 = all(item(k, x) for _, x in rows)
        ok7 = all(item(k, x) for _, x in rows7)
        ok = ok5 and ok7
        print(f"k {k}: item with the quintic a_med {ok5}; with the septic a_med {ok7}")
        res[k] = (ok, min(x for _, x in rows), max(x for _, x in rows))
        print(
            f"k {k}: located {len(rows)}; ratio range {res[k][1]:.4f} to {res[k][2]:.4f}; item {ITEM[k]} (|ratio - 1| <= {TOL[k]}): {ok}"
        )
    if incomplete:
        print("VERDICT q1298 quadratic-g closure class: INCOMPLETE")
        return
    allok = all(v[0] for v in res.values())
    print(
        f"VERDICT q1298 quadratic-g closure class: {'PASS' if allok else 'FAIL'}; "
        + "; ".join(f"{ITEM[k]} {res[k][0]}" for k in STAGE)
        + "; ratio ranges "
        + ", ".join(f"{k:g}: {res[k][1]:.3f}-{res[k][2]:.3f}" for k in sorted(res))
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
