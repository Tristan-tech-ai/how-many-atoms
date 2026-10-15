"""q1278 (8 Oct 2026, registered in amendment 2058): blind test of the smooth continuation of the amplitude c(k) of the median-sum residual to SMALL exponents: k = 1.25 (interpolation) and k = 0.75, 0.5 (extrapolation toward the capacity), events K_b 29 to 36 each.
Measured values behind the prediction (q1274, q1276; means over K_b 29 to 36, NPMLE residual = 1): c(1) = 1, c(1.5) = 1.4080, c(2) = 1.8007, c(2.5) = 2.1825, c(3) = 2.5571 (and c(4) = 3.2896, c(5) = 4.0106). The quartic through k = 1, 1.5, 2, 2.5, 3 gives c(1.25) = 1.2062, c(0.75) = 0.7887, c(0.5) = 0.5716, c(0.25) = 0.348, c(0) = 0.117; polynomials of degree 3 to 6 through more points give c(1.25) 1.206 to 1.2064, c(0.75) 0.788 to 0.7895, c(0.5) 0.569 to 0.575, c(0) 0.10 to 0.14. The capacity amplitude measured at K_b 29 to 36 (interpolated, least accurate) is 0.14 to 0.20, so the continuation to k = 0 is consistent with it. The competing description, the quadratic fit of c^2 on k = 2 to 5 (exploratory, amendment 2057), gives 1.189, 0.710, 0.374 at 1.25, 0.75, 0.5.
Items (each on the mean m over the located and in-range events of c = R_k / R_NPMLE at equal v^2, R_NPMLE interpolated in ln R against v^2 from the refreshed NPMLE events): Q125 |m(1.25) - 1.206| <= 0.020; Q075 |m(0.75) - 0.789| <= 0.050; Q05 |m(0.5) - 0.572| <= 0.080. Bar derivation: c(k) drifts with K_b by 0.1 to 1.5 percent for k = 1.5 to 5 but the capacity ratio drifts by 40 percent over K_b 29 to 36, so the drift at small k is not known; the bands are 1.7 percent (k = 1.25), 6.3 percent (k = 0.75) and 14 percent (k = 0.5), which excludes the competing description (0.710 and 0.374) and the straight line through k = 1, 2 (0.6 at k = 0.5 is inside Q05 and is not separated by it). Class: PASS if Q125, Q075 and Q05; INCOMPLETE if for some k fewer than 7 of 8 events are located and in range; otherwise FAIL. Priors: Q125 0.8, Q075 0.55, Q05 0.4.
Reading without item: the trend of c with K_b at each small k, c(k) ordering (c(0.5) < c(0.75) < c(1.25) < 1), and the capacity amplitude.
Inputs: half-line stages c33_k125c.json, c33_k075c.json, c33_k05bc.json (driver c36, derived bulk law, D = 92; k = 0.5 and 0.75 from run b converged to D = 68, offsets fitted on its atoms), base law numbers of each spline fitted in this script to the atoms of depth 14 to 30, median-sum table q1220, the refreshed NPMLE residuals. No amplitude of these functionals is computed or read.
--registration-run : controls, the table q1278_predictions.json (sha256) and the launch script run1278.sh.
--read             : items on the located events (files q607_u1278_k<tag>_K<K>.json; solver c35 with KEXP = k).
Located: a_t not null, the last residual below 1e-20 and |a_t - last a| < 1e-9.
Controls (first): c1 item functions accept 1.206, 0.789, 0.572 and reject 1.23, 0.84, 0.66; c2 quintic against septic a_med within 0.5 percent of the predicted residual at each event and k; c3 D = 80 against 92 likewise; c4 chain solver bit-identical at k = 2 and the NPMLE reference exact at the v^2 of K_b 35; c5 all v^2 inside the NPMLE reference range; c6 the offsets vary smoothly with k.
Usage: py q1278_ck_small_k_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)"""

import hashlib, json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import q1250_quintic_residual as Q
import q1271_linear_in_k_blind as Z
import q1273_k4_curvature_blind as W

TARGET = list(range(29, 37))
TAGS = {1.25: "k125", 0.75: "k075", 0.5: "k05b"}
CPRED = {1.25: 1.2062, 0.75: 0.7887, 0.5: 0.5716}
OFFS = {
    0.5: (-4.11796, 1.46511),
    0.75: (-3.86853, 1.42768),
    1.25: (-3.53897, 1.37187),
    1.5: (-3.41721, 1.34932),
    2.0: (-3.22067, 1.31065),
    2.5: (-3.06448, 1.27786),
    3.0: (-2.93445, 1.24916),
    4.0: (-2.72486, 1.20012),
    5.0: (-2.55856, 1.15875),
}


def items(k, m):
    if k == 1.25:
        return abs(m - 1.206) <= 0.020
    if k == 0.75:
        return abs(m - 0.789) <= 0.050
    return abs(m - 0.572) <= 0.080


def regrun():
    rn = Z.curve(Z.refs()[1])
    x33 = Z.refs()[1][35][1]
    dn = Z.refs()[1]
    c1 = (
        items(1.25, 1.206)
        and items(0.75, 0.789)
        and items(0.5, 0.572)
        and not (items(1.25, 1.23) or items(0.75, 0.84) or items(0.5, 0.66))
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
        print("REGISTRATION-RUN q1278: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1278_predictions.json", "w"), indent=1)
    L = [
        "#!/bin/bash",
        "# run1278.sh (8 Oct 2026): full-chain locators c35 for k = 1.25, 0.75, 0.5 and K_b 29 to 36 of amendments 2058 and 2059; starts a0 from q1278_predictions.json",
        'cd "./centre_scripts"',
    ]
    for k, tag in TAGS.items():
        law = out[str(k)]["law"]
        L.append(f"export KEXP={k} HL_JSON=c33_{tag}c.json HL_STAGE=92 D0={law[0]:.5f} NU0={law[1]:.5f}")
        for K in TARGET:
            L.append(
                f"py c35_chain_k.py {K} {K // 2} 160 {out[str(k)][str(K)]['a0']:.6f} 0.003 u1278_{tag}_K{K} > q607_u1278_{tag}_K{K}.log 2>&1"
            )
    L.append("echo done $(date +%H:%M:%S) > run1278.done")
    open("run1278.sh", "w", newline="\n").write("\n".join(L) + "\n")
    for k in TAGS:
        print(
            f"k {k}: base law {out[str(k)]['law'][0]:.5f}, {out[str(k)]['law'][1]:.5f}; v^2 {out[str(k)]['29']['v2']:.2f} to {out[str(k)]['36']['v2']:.2f}; predicted R "
            + ", ".join(f"{out[str(k)][str(K)]['R_pred']:.4f}" for K in TARGET)
        )
    print("sha256", sh)
    print("REGISTRATION-RUN q1278: controls True, table and run1278.sh written")


def read():
    js = json.load(open("q1278_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1278: VOID (table hash)")
        sys.exit(2)
    rn = Z.curve(Z.refs()[1])
    res = {}
    allok = True
    incomplete = False
    for k, tag in TAGS.items():
        N5 = W.nhalf(f"c33_{tag}c.json", 92.0, 5, tuple(js[str(k)]["law"]))
        cs = []
        for K in TARGET:
            a = W.loc(f"q607_u1278_{tag}_K{K}")
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
            f"k {k}: mean c {m:.4f} (predicted {CPRED[k]}); item {'Q125' if k == 1.25 else 'Q075' if k == 0.75 else 'Q05'}: {items(k, m)}"
        )
    if incomplete:
        print("VERDICT q1278 small-k c(k) class: INCOMPLETE")
        return
    print(
        f"VERDICT q1278 small-k c(k) class: {'PASS' if allok else 'FAIL'}; Q125 {res[1.25][1]}; Q075 {res[0.75][1]}; Q05 {res[0.5][1]}; means "
        + ", ".join(f"{k}: {res[k][0]:.4f}" for k in (1.25, 0.75, 0.5))
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
