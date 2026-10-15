"""q1276 (8 Oct 2026, registered in amendment 2055): blind test that the amplitude c(k) of the median-sum residual is a smooth function of the exponent k: the held-out values k = 1.5 and 2.5 (interpolation) and k = 5 (extrapolation), events K_b 29 to 36 for each.
Measured values behind the prediction (q1274, means over K_b 29 to 36, NPMLE residual = 1): c(1) = 1, c(2) = 1.8007, c(3) = 2.5571, c(4) = 3.2896 (increments 0.8007, 0.7564, 0.7325; second differences -0.0443, -0.0239; third difference +0.0204).
Predictions from these four numbers only (no event of k = 1.5, 2.5, 5 exists): the cubic through k = 1, 2, 3, 4 gives c(1.5) = 1.4072 and c(2.5) = 2.1832; for k = 5 the continuations of the increments agree: cubic 4.019, geometric decay of the second difference 4.009, constant second difference 3.998, power law c ~ k^0.874 3.998; the registered band covers them.
Items (each on the mean over the located events of c = R_k / R_NPMLE at equal v^2, R_NPMLE interpolated in ln R against v^2 from the refreshed NPMLE events K_b 20 to 44 except 38): P15 |m(1.5) - 1.407| <= 0.012; P25 |m(2.5) - 2.183| <= 0.020; P5 3.96 <= m(5) <= 4.06. Bar derivation: c(3) and c(4) drift by 0.7 percent over K_b 29 to 36, so 0.012 (0.85 percent) and 0.020 (0.9 percent) hold a drift; the k = 5 band (+-1.2 percent) covers the four extrapolations and excludes the straight line 0.2 + 0.8 k (4.2). Class: PASS if P15, P25 and P5; INCOMPLETE if for some k fewer than 7 of the 8 events are located and in range; otherwise FAIL. Priors: P15 0.7, P25 0.7, P5 0.5.
Inputs: half-line stages c33_k15c.json, c33_k25c.json, c33_k5c.json (driver c36, derived bulk law, D = 92), base law numbers of each spline fitted in this script to the atoms of depth 14 to 30 (stored in the table), median-sum table q1220, the refreshed NPMLE residuals. No amplitude of these functionals is computed or read.
--registration-run : controls, the table q1276_predictions.json (sha256) with, per k and event, a_med, v^2, R_NPMLE, the predicted residual and the start a0 = a_med + c_pred R_NPMLE - 0.004; and the launch script run1276.sh (written with real newlines).
--read             : items on the located events (files q607_u1276_k<tag>_K<K>.json, key a_t; solver c35 with KEXP = k).
Located: a_t not null, the last residual below 1e-20 and |a_t - last a| < 1e-9.
Controls (first): c1 the item functions accept 1.407, 2.183 and 4.01 for P15, P25, P5 and reject 1.43, 2.22, 3.90 and 4.12; c2 the quintic and septic splines give a_med within 0.5 percent of the predicted residual at each event and each k; c3 the stages D = 80 and D = 92 give a_med within 0.5 percent of the predicted residual; c4 the chain solver is bit-identical at k = 2 and k = 1 (amendment 2046) and the reference NPMLE interpolation is exact at the v^2 of the NPMLE event K_b 35 within 1e-9 relative (K_b 33 is not stored); c5 all eight v^2 of each k lie inside the range of the NPMLE reference; c6 the offsets of the stages vary smoothly with k (OFF1 and OFF2 of k = 1.5, 2, 2.5, 3, 4, 5 have no sign change of the second difference).
Usage: py q1276_ck_smooth_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)"""

import hashlib, json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import q1250_quintic_residual as Q
import q1271_linear_in_k_blind as Z
import q1273_k4_curvature_blind as W

TARGET = list(range(29, 37))
TAGS = {1.5: "k15", 2.5: "k25", 5.0: "k5"}
CPRED = {1.5: 1.4072, 2.5: 2.1832, 5.0: 4.010}
OFFS = {
    1.5: (-3.41721, 1.34932),
    2.0: (-3.22067, 1.31065),
    2.5: (-3.06448, 1.27786),
    3.0: (-2.93445, 1.24916),
    4.0: (-2.72486, 1.20012),
    5.0: (-2.55856, 1.15875),
}


def items(k, m):
    if k == 1.5:
        return abs(m - 1.407) <= 0.012
    if k == 2.5:
        return abs(m - 2.183) <= 0.020
    return 3.96 <= m <= 4.06


def regrun():
    rn = Z.curve(Z.refs()[1])
    x33 = Z.refs()[1][35][1]
    dn = Z.refs()[1]
    c1 = (
        items(1.5, 1.407)
        and items(2.5, 2.183)
        and items(5.0, 4.01)
        and not (items(1.5, 1.43) or items(2.5, 2.22) or items(5.0, 3.90) or items(5.0, 4.12))
    )
    c4 = (
        abs(rn(x33) / dn[35][0] - 1) < 1e-9
        and abs(json.load(open("q607_ctl_k2_K30.json"))["a_t"] - json.load(open("c19_u1231_K30.json"))["a_t"])
        < 1e-12
    )
    ks = sorted(OFFS)
    d2 = [[OFFS[ks[i + 1]][j] - OFFS[ks[i]][j] for i in range(len(ks) - 1)] for j in range(2)]
    c6 = all(abs(x) > 0 for x in d2[0]) and all(
        np.sign(np.diff(np.array(row) / np.diff(ks))).std() == 0 for row in d2
    )
    out = {}
    worst4 = worst5 = 0.0
    ok5 = True
    for k, tag in TAGS.items():
        sj = f"c33_{tag}c.json"
        law = W.fitlaw(sj, 92.0)
        N5 = W.nhalf(sj, 92.0, 5, law)
        N7 = W.nhalf(sj, 92.0, 7, law)
        N80 = W.nhalf(sj, 80.0, 5, law)
        out[str(k)] = {"law": list(law)}
        for K in TARGET:
            am = Q.root(N5, K)
            x = W.v2f(N5, am)
            r = rn(x)
            if r is None:
                ok5 = False
                print(f"k {k} K_b {K}: v^2 {x:.2f} outside the NPMLE reference range")
                continue
            rp = CPRED[k] * r
            worst4 = max(worst4, abs(am - Q.root(N7, K)) * 1e9 / rp)
            worst5 = max(worst5, abs(am - Q.root(N80, K)) * 1e9 / rp)
            out[str(k)][str(K)] = dict(
                a_med=am, v2=float(x), R_NPMLE=r, R_pred=rp, a0=float(am + rp * 1e-9 - 0.004)
            )
    c2, c3 = worst4 <= 0.005, worst5 <= 0.005
    print(
        f"control c1 (item functions): {c1}; c2 (quintic vs septic, max relative {worst4:.5f}): {c2}; c3 (D = 80 vs 92, max relative {worst5:.2e}): {c3}; c4 (chain bit-identical, NPMLE reference exact): {c4}; c5 (all v^2 in range): {ok5}; c6 (offsets smooth in k): {c6}"
    )
    if not (c1 and c2 and c3 and c4 and ok5 and c6):
        print("REGISTRATION-RUN q1276: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1276_predictions.json", "w"), indent=1)
    L = [
        "#!/bin/bash",
        "# run1276.sh (8 Oct 2026): full-chain locators c35 for k = 1.5, 2.5, 5 and K_b 29 to 36 of amendments 2055 and 2056; starts a0 from q1276_predictions.json",
        'cd "./centre_scripts"',
    ]
    for k, tag in TAGS.items():
        law = out[str(k)]["law"]
        L.append(f"export KEXP={k} HL_JSON=c33_{tag}c.json HL_STAGE=92 D0={law[0]:.5f} NU0={law[1]:.5f}")
        for K in TARGET:
            L.append(
                f"py c35_chain_k.py {K} {K // 2} 160 {out[str(k)][str(K)]['a0']:.6f} 0.003 u1276_{tag}_K{K} > q607_u1276_{tag}_K{K}.log 2>&1"
            )
    L.append("echo done $(date +%H:%M:%S) > run1276.done")
    open("run1276.sh", "w", newline="\n").write("\n".join(L) + "\n")
    for k in TAGS:
        print(
            f"k {k}: base law {out[str(k)]['law'][0]:.5f}, {out[str(k)]['law'][1]:.5f}; v^2 {out[str(k)]['29']['v2']:.2f} to {out[str(k)]['36']['v2']:.2f}; predicted R "
            + ", ".join(f"{out[str(k)][str(K)]['R_pred']:.4f}" for K in TARGET)
        )
    print("sha256", sh)
    print("REGISTRATION-RUN q1276: controls True, table and run1276.sh written")


def read():
    js = json.load(open("q1276_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1276: VOID (table hash)")
        sys.exit(2)
    rn = Z.curve(Z.refs()[1])
    res = {}
    allok = True
    incomplete = False
    for k, tag in TAGS.items():
        N5 = W.nhalf(f"c33_{tag}c.json", 92.0, 5, tuple(js[str(k)]["law"]))
        cs = []
        for K in TARGET:
            a = W.loc(f"q607_u1276_{tag}_K{K}")
            if a is None:
                print(f"k {k} K_b {K}: NOT LOCATED")
                continue
            x = W.v2f(N5, a)
            r = rn(x)
            if r is None:
                print(f"k {k} K_b {K}: v^2 {x:.2f} outside the reference")
                continue
            c = (a - js[str(k)][str(K)]["a_med"]) * 1e9 / r
            cs.append(c)
            print(
                f"k {k} K_b {K}: R {(a - js[str(k)][str(K)]['a_med']) * 1e9:+.4f}e-9 (predicted {js[str(k)][str(K)]['R_pred']:.4f}), c = R / R_NPMLE {c:.4f}"
            )
        if len(cs) < 7:
            incomplete = True
            print(f"k {k}: events {len(cs)} of {len(TARGET)}")
            continue
        m = float(np.mean(cs))
        res[k] = (m, items(k, m))
        allok = allok and items(k, m)
        print(
            f"k {k}: mean c {m:.4f} (predicted {CPRED[k]}); item {'P15' if k == 1.5 else 'P25' if k == 2.5 else 'P5'}: {items(k, m)}"
        )
    if incomplete:
        print("VERDICT q1276 smooth-c(k) class: INCOMPLETE")
        return
    print(
        f"VERDICT q1276 smooth-c(k) class: {'PASS' if allok else 'FAIL'}; P15 {res[1.5][1]}; P25 {res[2.5][1]}; P5 {res[5.0][1]}; means "
        + ", ".join(f"{k}: {res[k][0]:.4f}" for k in (1.5, 2.5, 5.0))
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
