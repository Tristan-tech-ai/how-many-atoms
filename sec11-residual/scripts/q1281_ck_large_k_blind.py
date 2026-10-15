"""q1281 (8 Oct 2026, registered in amendment 2061): blind test at LARGE exponents, k = 8 and k = 12, separating two descriptions of the amplitude c(k) of the median-sum residual (NPMLE residual = 1) that agree on the ten measured values k = 0.5 to 5 (K_b 29 to 36 means: 0.5659, 0.7878, 1, 1.2064, 1.4080, 1.8007, 2.1825, 2.5571, 3.2896, 4.0106):
 POWER  c(k) = ((k + a) / (1 + a))^p with a = 0.0562, p = 0.8852 (two parameters, maximum residual 0.28 percent, rms 0.14 percent): c(8) = 6.0407, c(12) = 8.6311;
 LINEAR (asymptotically linear) c(k) = (A k + B - D / (k + s)) / (A + B - D / (1 + s)) with A = 0.1766, B = 0.1929, D = 0.3372, s = 1.976 (four parameters, maximum residual 0.09 percent): c(8) = 6.1356, c(12) = 8.9306.
They differ by 1.6 percent at k = 8 and 3.5 percent at k = 12 (and by 0.3 percent at k = 5 where c = 4.0106 was measured). Events: K_b 29 to 36 for k = 8 and k = 12 (functional W P^-k, never solved before).
Items (each on the mean m over the located and in-range events of c = R_k / R_NPMLE at equal v^2, R_NPMLE interpolated in ln R against v^2 from the refreshed NPMLE events): HP12 |m(12) - 8.6311| <= 0.012 * 8.6311; HL12 |m(12) - 8.9306| <= 0.012 * 8.9306; HP8 |m(8) - 6.0407| <= 0.007 * 6.0407; HL8 |m(8) - 6.1356| <= 0.007 * 6.1356. The bands do not overlap. Class by k = 12: POWER if HP12 and not HL12; LINEAR if HL12 and not HP12; NEITHER otherwise (both False); INCOMPLETE if fewer than 7 of 8 events are located and in range at k = 12. The k = 8 items are reported next to it. Bar derivation: at k = 1.5 to 5 the means drift by 0.1 to 1.5 percent over K_b 29 to 36; 0.7 and 1.2 percent hold a drift of that size and are smaller than half the distance between the forms (0.8 and 1.7 percent). Priors: POWER 0.3, LINEAR 0.3, NEITHER 0.4.
Reading without item: per-event ratios, the trend with K_b, the local exponent d ln c / d ln k between k = 5, 8, 12.
Inputs: half-line stages c33_k8c.json and c33_k12c.json (driver c36; k = 12 run b with law (1.42, -0.13), amendment 2061), base law numbers of each spline fitted in this script to the atoms of depth 14 to 30, median-sum table q1220, the refreshed NPMLE residuals. No amplitude of these functionals is computed or read.
--registration-run : controls, the table q1281_predictions.json (sha256) and the launch script run1281.sh.
--read             : items on the located events (files q607_u1281_k<tag>_K<K>.json; solver c35 with KEXP = k).
Located: a_t not null, the last residual below LOC_RES (default 1e-20; 1e-18 for the second reading of amendment 2063) and |a_t - last a| < 1e-9.
Controls (first): c1 the item functions accept 8.6311 for HP12 only, 8.9306 for HL12 only, and the midpoint 8.781 for neither; likewise 6.0407, 6.1356 and 6.088 at k = 8; c2 quintic against septic a_med within 0.5 percent of the predicted residual at each event and k; c3 D = 80 against 92 likewise; c4 chain solver bit-identical at k = 2 and the NPMLE reference exact at the v^2 of K_b 35; c5 all v^2 inside the NPMLE reference range; c6 the offsets vary smoothly with k.
Usage: py q1281_ck_large_k_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)"""

import hashlib, json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import q1250_quintic_residual as Q
import q1271_linear_in_k_blind as Z
import q1273_k4_curvature_blind as W

RES_MAX = float(os.environ.get("LOC_RES", "1e-20"))


def locr(f):
    try:
        j = json.load(open(f + ".json"))
        hl = [x for x in j["hist"] if "res" in x and "a" in x]
        a = j.get("a_t")
        return (
            a if (a is not None and hl and hl[-1]["res"] < RES_MAX and abs(a - hl[-1]["a"]) < 1e-9) else None
        )
    except Exception:
        return None


TARGET = list(range(29, 37))
TAGS = {8.0: "k8", 12.0: "k12"}
PF = lambda k: ((k + 0.0562) / 1.0562) ** 0.8852
LF = lambda k: (0.1766 * k + 0.1929 - 0.3372 / (k + 1.976)) / (0.1766 + 0.1929 - 0.3372 / 2.976)
BAND = {8.0: 0.007, 12.0: 0.012}
CPRED = {8.0: 0.5 * (PF(8.0) + LF(8.0)), 12.0: 0.5 * (PF(12.0) + LF(12.0))}
OFFS = {
    8.0: (-2.1977, 1.06121),
    12.0: (-1.87501, 0.96461),
    5.0: (-2.55856, 1.15875),
    4.0: (-2.72486, 1.20012),
    3.0: (-2.93445, 1.24916),
    0.5: (-4.11796, 1.46511),
    0.75: (-3.86853, 1.42768),
    1.25: (-3.53897, 1.37187),
    1.5: (-3.41721, 1.34932),
    2.0: (-3.22067, 1.31065),
    2.5: (-3.06448, 1.27786),
}


def hp(k, m):
    return abs(m - PF(k)) <= BAND[k] * PF(k)


def hl(k, m):
    return abs(m - LF(k)) <= BAND[k] * LF(k)


def items(k, m):
    return hp(k, m) or hl(k, m)


def regrun():
    rn = Z.curve(Z.refs()[1])
    x33 = Z.refs()[1][35][1]
    dn = Z.refs()[1]
    c1 = all(
        hp(k, PF(k))
        and not hl(k, PF(k))
        and hl(k, LF(k))
        and not hp(k, LF(k))
        and not hp(k, CPRED[k])
        and not hl(k, CPRED[k])
        for k in (8.0, 12.0)
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
        print("REGISTRATION-RUN q1281: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1281_predictions.json", "w"), indent=1)
    L = [
        "#!/bin/bash",
        "# run1281.sh (8 Oct 2026): full-chain locators c35 for k = 8 and 12 and K_b 29 to 36 of amendments 2061 and 2062; starts a0 from q1281_predictions.json",
        'cd "./centre_scripts"',
    ]
    for k, tag in TAGS.items():
        law = out[str(k)]["law"]
        L.append(f"export KEXP={k} HL_JSON=c33_{tag}c.json HL_STAGE=92 D0={law[0]:.5f} NU0={law[1]:.5f}")
        for K in TARGET:
            L.append(
                f"py c35_chain_k.py {K} {K // 2} 160 {out[str(k)][str(K)]['a0']:.6f} 0.003 u1281_{tag}_K{K} > q607_u1281_{tag}_K{K}.log 2>&1"
            )
    L.append("echo done $(date +%H:%M:%S) > run1281.done")
    open("run1281.sh", "w", newline="\n").write("\n".join(L) + "\n")
    for k in TAGS:
        print(
            f"k {k}: base law {out[str(k)]['law'][0]:.5f}, {out[str(k)]['law'][1]:.5f}; v^2 {out[str(k)]['29']['v2']:.2f} to {out[str(k)]['36']['v2']:.2f}; predicted R "
            + ", ".join(f"{out[str(k)][str(K)]['R_pred']:.4f}" for K in TARGET)
        )
    print("sha256", sh)
    print("REGISTRATION-RUN q1281: controls True, table and run1281.sh written")


def read():
    js = json.load(open("q1281_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1281: VOID (table hash)")
        sys.exit(2)
    rn = Z.curve(Z.refs()[1])
    res = {}
    incomplete = False
    for k, tag in TAGS.items():
        N5 = W.nhalf(f"c33_{tag}c.json", 92.0, 5, tuple(js[str(k)]["law"]))
        cs = []
        for K in TARGET:
            a = locr(f"q607_u1281_{tag}_K{K}")
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
                f"k {k} K_b {K}: R {(a - js[str(k)][str(K)]['a_med']) * 1e9:+.4f}e-9, c = R / R_NPMLE {c:.4f}"
            )
        if len(cs) < 7:
            incomplete = True
            print(f"k {k}: events {len(cs)} of {len(TARGET)}")
            continue
        m = float(np.mean(cs))
        res[k] = (m, hp(k, m), hl(k, m))
        print(
            f"k {k}: mean c {m:.4f}; power form {PF(k):.4f}: {res[k][1]}; linear form {LF(k):.4f}: {res[k][2]}"
        )
    if 12.0 not in res:
        print("VERDICT q1281 large-k class: INCOMPLETE")
        return
    hp12, hl12 = res[12.0][1], res[12.0][2]
    cl = "POWER" if hp12 and not hl12 else "LINEAR" if hl12 and not hp12 else "NEITHER"
    print(
        f"VERDICT q1281 large-k class: {cl}; HP12 {hp12}; HL12 {hl12}; HP8 {res.get(8.0, (0, None, None))[1]}; HL8 {res.get(8.0, (0, None, None))[2]}; means "
        + ", ".join(f"{k}: {res[k][0]:.4f}" for k in sorted(res))
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
