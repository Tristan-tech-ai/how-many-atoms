"""q1294 (8 Oct 2026, registered in amendment 2072): blind test of the full closure R_k = sigma(k) Da(v^2) (1 + g(k) / v^2) with the laws sigma(k) = 2.616046 (k + 0.140737), g(k) = -10.1349 - 2.4524 k (fitted on k = 1 to 5, q1293) for the exponents k = 0.75, 8, 12 at K_b 14 to 28 and 37 to 40 (19 events per exponent; the events K_b 29 to 36 of these exponents were seen and are not test events).
Da(v^2) = |Im S_+(1/v^2)| / (2 rho) in 1e-9, rho = v / (4 pi), [8/8] approximant of e_2 .. e_18, theta 0.3 (q1285). v^2 is that of the event in the half-line N_half of the functional.
Items: L075, L8, L12: at least 15 of the 19 events located and |R / R_pred - 1| <= 0.06 (k = 0.75), 0.10 (k = 8), 0.15 (k = 12) at every located event. Class PASS if all three; INCOMPLETE if for some k fewer than 15 are located; otherwise FAIL.
Located: a_t not null, last residual below 1e-18, |a_t - last a| < 1e-9.
--registration-run : controls, table q1294_predictions.json (sha256), launch script run1294.sh; --read : items; last line 'VERDICT q1294 closure-in-k class: ...'.
Controls (first): c1 item functions accept 1.0 and 0.95 and reject 1.10 and 0.88 (k = 0.75), accept 1.08 and reject 1.15 (k = 8), accept 1.12 and reject 1.25 (k = 12); c2 Da(115) = 0.21082 within 1e-4 relative; c3 the laws reproduce the fitted values of q1293 for k = 1 to 5 within 1 percent (sigma) and 0.5 (g); c4 the closure with the laws reproduces the measured k = 2 events K_b 14 to 40 within 3 percent at every event; c5 (repaired in amendment 2073) quintic against septic a_med: the largest relative difference over the events of an exponent at most one tenth of its item half-width (0.006, 0.010, 0.015), with a synthetic negative and positive test; c6 D = 80 against 92 likewise; c7 chain solver bit-identical at k = 2.
Usage: py q1294_closure_linear_k_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import hashlib, io, contextlib, json, os, sys

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
    import q1282_extended_decay_reading as D
mp.mp.dps = 30
Bf = T.machine(T.coef_series(), 8, 8)[0]
A_S, K0_S, G0, G1 = 2.616046, 0.140737, -10.1349, -2.4524
TARGET = list(range(14, 29)) + list(range(37, 41))
NEED = 15
TOL = {0.75: 0.06, 8.0: 0.10, 12.0: 0.15}
STAGE = {
    0.75: ("c33_k075c.json", "q1278_predictions.json", "0.75", "k075"),
    8.0: ("c33_k8c.json", "q1281_predictions.json", "8.0", "k8"),
    12.0: ("c33_k12c.json", "q1281_predictions.json", "12.0", "k12"),
}


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


def sig(k):
    return A_S * (k + K0_S)


def gk(k):
    return G0 + G1 * k


def rpred(k, v2):
    return sig(k) * Da(v2) * (1 + gk(k) / v2)


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


def nh(k, deg, D_=92.0):
    sj, pj, key, tag = STAGE[k]
    law = tuple(json.load(open(pj))[key]["law"])
    return W.nhalf(sj, D_, deg, law), law


def c5ok(k, w):
    return w <= 0.1 * TOL[k]


def regrun():
    c1 = (
        item(0.75, 1.0)
        and item(0.75, 0.95)
        and not item(0.75, 1.10)
        and not item(0.75, 0.88)
        and item(8.0, 1.08)
        and not item(8.0, 1.15)
        and item(12.0, 1.12)
        and not item(12.0, 1.25)
    )
    c2 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
    fitted = {
        1.0: (3.0087, -12.325),
        1.5: (4.2959, -13.515),
        2.0: (5.5832, -15.019),
        2.5: (6.9008, -16.690),
        3.0: (8.2010, -17.941),
        4.0: (10.8222, -20.141),
        5.0: (13.4703, -21.908),
    }
    c3 = all(abs(sig(k) / s - 1) < 0.01 and abs(gk(k) - g) < 0.5 for k, (s, g) in fitted.items())
    d2 = [(K, r, v) for K, r, v in D.data.get(2, []) if 14 <= K <= 40 and K != 38]
    dev = max(abs(r / rpred(2.0, v) - 1) for K, r, v in d2)
    c4 = dev < 0.03
    c7 = (
        abs(json.load(open("q607_ctl_k2_K30.json"))["a_t"] - json.load(open("c19_u1231_K30.json"))["a_t"])
        < 1e-12
    )
    out = {}
    w5 = {}
    w6 = {}
    for k in STAGE:
        N5, law = nh(k, 5)
        N7, _ = nh(k, 7)
        N80, _ = nh(k, 5, 80.0)
        out[str(k)] = {"law": list(law)}
        w5[k] = w6[k] = 0.0
        for K in TARGET:
            am = Q.root(N5, K)
            v2 = Z.v2f(N5, am)
            rp = rpred(k, v2)
            w5[k] = max(w5[k], abs(am - Q.root(N7, K)) * 1e9 / rp)
            w6[k] = max(w6[k], abs(am - Q.root(N80, K)) * 1e9 / rp)
            out[str(k)][str(K)] = dict(a_med=am, v2=float(v2), R_pred=rp, a0=float(am + rp * 1e-9 - 0.0045))
        print(
            f"k {k}: v^2 {out[str(k)]['14']['v2']:.2f} to {out[str(k)]['40']['v2']:.2f}; predicted R (1e-9): K_b 14 {out[str(k)]['14']['R_pred']:.2f}, 20 {out[str(k)]['20']['R_pred']:.3f}, 28 {out[str(k)]['28']['R_pred']:.4f}, 40 {out[str(k)]['40']['R_pred']:.5f}; sigma {sig(k):.3f}, g {gk(k):.2f}"
        )
    c5neg = (
        not c5ok(0.75, 0.02)
        and not c5ok(8.0, 0.02)
        and not c5ok(12.0, 0.02)
        and c5ok(0.75, 0.9 * 0.006)
        and c5ok(8.0, 0.9 * 0.010)
        and c5ok(12.0, 0.9 * 0.015)
    )
    c5 = c5neg and all(c5ok(k, w5[k]) for k in STAGE)
    c6 = max(w6.values()) <= 0.005
    print(
        f"control c1 {c1}; c2 {c2}; c3 (laws vs fitted values) {c3}; c4 (closure vs measured k = 2, max deviation {dev:.4f}) {c4}; c5 (quintic vs septic, max relative per k {', '.join(f'{k}: {w5[k]:.5f}' for k in STAGE)}; synthetic test {c5neg}) {c5}; c6 (D = 80 vs 92, {max(w6.values()):.2e}) {c6}; c7 {c7}"
    )
    if not (c1 and c2 and c3 and c4 and c5 and c6 and c7):
        print("REGISTRATION-RUN q1294: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1294_predictions.json", "w"), indent=1)
    L = [
        "#!/bin/bash",
        "# run1294.sh (8 Oct 2026): full-chain locators c35 for k = 0.75, 8, 12 at K_b 14-28 and 37-40 (amendments 2072 and 2073); starts from q1294_predictions.json",
        'cd "./centre_scripts"',
    ]
    for k, (sj, pj, key, tag) in STAGE.items():
        law = out[str(k)]["law"]
        L.append(f"export KEXP={k} HL_JSON={sj} HL_STAGE=92 D0={law[0]:.5f} NU0={law[1]:.5f}")
        for K in TARGET:
            L.append(
                f"py c35_chain_k.py {K} {K // 2} 160 {out[str(k)][str(K)]['a0']:.6f} 0.003 u1294_{tag}_K{K} > q607_u1294_{tag}_K{K}.log 2>&1"
            )
    L.append("echo done $(date +%H:%M:%S) > run1294.done")
    open("run1294.sh", "w", newline="\n").write("\n".join(L) + "\n")
    print("sha256", sh)
    print("REGISTRATION-RUN q1294: controls True, table and run1294.sh written")


def read():
    js = json.load(open("q1294_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1294: VOID (table hash)")
        sys.exit(2)
    res = {}
    incomplete = False
    for k, (sj, pj, key, tag) in STAGE.items():
        N5, _ = nh(k, 5)
        N7, _ = nh(k, 7)
        rows = []
        chg = 0.0
        for K in TARGET:
            a = locr(f"q607_u1294_{tag}_K{K}")
            if a is None:
                print(f"k {k} K_b {K}: NOT LOCATED")
                continue
            v2 = Z.v2f(N5, a)
            r = (a - js[str(k)][str(K)]["a_med"]) * 1e9
            rp = rpred(k, v2)
            rows.append((K, r / rp))
            chg = max(chg, abs((a - Q.root(N7, K)) * 1e9 / rp - r / rp))
            print(f"k {k} K_b {K}: R {r:+.5f}e-9, R_pred {rp:.5f}, ratio {r / rp:.4f}")
        print(f"k {k}: Reading, largest change of the ratio with the septic a_med: {chg:.5f}")
        if len(rows) < NEED:
            incomplete = True
            print(f"k {k}: located {len(rows)} of {len(TARGET)}")
            continue
        ok = all(item(k, x) for _, x in rows)
        res[k] = (ok, min(x for _, x in rows), max(x for _, x in rows))
        print(
            f"k {k}: located {len(rows)}; ratio range {res[k][1]:.4f} to {res[k][2]:.4f}; item {'L075' if k == 0.75 else 'L8' if k == 8.0 else 'L12'} (|ratio - 1| <= {TOL[k]}): {ok}"
        )
    if incomplete:
        print("VERDICT q1294 closure-in-k class: INCOMPLETE")
        return
    allok = all(v[0] for v in res.values())
    print(
        f"VERDICT q1294 closure-in-k class: {'PASS' if allok else 'FAIL'}; L075 {res[0.75][0]}; L8 {res[8.0][0]}; L12 {res[12.0][0]}; ratio ranges "
        + ", ".join(f"{k}: {res[k][1]:.3f}-{res[k][2]:.3f}" for k in sorted(res))
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
