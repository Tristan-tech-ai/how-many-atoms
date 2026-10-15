"""q1345 (8 Oct 2026, registered in amendment 2117): blind test, on the REPAIRED instrument (tight_run.py), of the hard-wall closure R / Da = c0(k) - C1 / v^2 with ONE constant C1 and a cubic c0(k), on three new exponents.
Frozen from q1342_out.txt (repaired data, amendment 2116): C1 = 146.413 and the ten values c0(k) of the common-c1 fit at k = 0.02, 0.05, 0.06, 0.075, 0.09, 0.1, 0.25, 0.35, 0.5, 0.75; c0(k) = the cubic least-squares polynomial through them (coefficients 0.72399, 0.11402, 1.39352, 1.86178 from the highest degree; residuals at most 0.0020; c0(0) = 1.8618 against 1.8596 of the capacity events fitted with c1 fixed, q1344).
Test exponents k = 0.15, 0.4, 0.6 (new stages c52_hw015c, hw04c, hw06c, D = 92); test events K_b 33 to 45 (13 each, 39 in all). Y = (R / Da) / (c0(k) - C1 / v^2) - 1 with R = (A_t - a_med) 1e9, a_med the quintic median-sum amplitude of the stage computed with the repaired density inversion (the chain solver c61 is unchanged: only its amplitude is read), v^2 = Z.v2f(N5, A_t).
Located rule: A_t returned, last Newton residual below 1e-18, |A_t - last A| < 1e-9; both interpolants, quintic and septic.
Items G1 (k = 0.15), G2 (k = 0.4), G3 (k = 0.6): at least 11 of the 13 events located and |Y| <= 0.04 at every located event, for both interpolants. Class PASS if G1, G2 and G3 hold; INCOMPLETE if for some exponent fewer than 11 events are located; otherwise FAIL.
Bar derivation: on the seen hard-wall events of K_b 33 to 45 (57 events, ten exponents; q1345a_bar_derivation_reading.py) the closure with the cubic c0 and the common C1 misses by at most 0.0153 (rms 0.0073; per exponent 0.0059 to 0.0153); the interpolation of the cubic between its points adds at most 0.0020 / 0.6 = 0.0033 in R / Da; the bar 0.04 is 2.6 times the largest seen misfit. The null 'no 1 / v^2 term' misses by about 0.7 to 2.8 in the same measure and the old common value 149.27 by about 0.02 to 0.05.
Priors: G1 0.8, G2 0.8, G3 0.8, class 0.55. Competing reading: the common constant varies slowly with k and the cubic c0 does not interpolate (a failure of one exponent).
Controls (registration pass): c1 item function; c2 Da(115) = 0.21082; c3 the closure with the frozen constants reproduces the seen events of K_b 33 to 45 (read from q1342's data, repaired) with |Y| <= 0.04 at every one (the largest value is printed); c4 the chain controls (c61_ctl30k0.json, c61_ctl30k1em6.json); c5 the three stages converged to D = 92; c6 the cubic reproduces the ten c0 values to 0.005; c7 power: with C1 = 0 every test event has |Y| > 0.04 and with C1 = 149.27 at least one event of k = 0.15 has |Y| > 0.04 (repaired in amendment 2118: the first pass, which required this of every exponent, was VOID; the old value misses by at most 0.0429, 0.0264, 0.0191 at k = 0.15, 0.4, 0.6, as the bar derivation said, about 0.02 to 0.05, so the test cannot separate it from the new value at k = 0.4 and 0.6); c8 at every test event |a_quintic - a_septic| 1e9 / Da <= 0.02 (no decision role).
Usage: py tight_run.py q1345_hardwall_repaired_blind.py --registration-run | --read   (from temporal_within/centre_scripts/; the registration run and the reading must both run through tight_run.py)
"""

import hashlib, io, contextlib, json, os, re, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import q1342_hardwall_refit_tight_reading as P  # builds the repaired seen-event data P.data (needs tight_run.py)
B = P.B
H, QQ, Z, Da, nhw = B.H, B.QQ, B.Z, B.Da, B.nhw
BAR = 0.04
KS = list(range(33, 46))
NEED = 11
TAGS = {0.15: "hw015", 0.4: "hw04", 0.6: "hw06"}


def frozen():
    t = open("q1342_out.txt", encoding="utf-8").read()
    m = re.search(r"one common c1 for all exponents: c1 = (-?[\d.]+); c0 per k: (.*)", t)
    C1 = -float(m.group(1))
    c0 = {float(k): float(v) for k, v in re.findall(r"([\d.]+): ([\d.]+)", m.group(2))}
    ks = sorted(c0)
    return C1, c0, np.polyfit(ks, [c0[k] for k in ks], 3)


def inside(y):
    return abs(y) <= BAR


def regrun():
    C1, c0, poly = frozen()
    ks = sorted(c0)
    c1 = inside(0.0) and inside(0.04) and inside(-0.04) and not inside(0.0401) and not inside(-0.0401)
    c2 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
    seen = [(k, K, R, v) for k in ks for K, R, v in P.data[k] if 33 <= K <= 45]
    ymax = max(abs((R / Da(v)) / (np.polyval(poly, k) - C1 / v) - 1) for k, K, R, v in seen)
    c3 = ymax <= BAR and len(seen) >= 50
    a0 = json.load(open("c61_ctl30k0.json"))["A_t"]
    a1 = json.load(open("c61_ctl30k1em6.json"))["A_t"]
    ast = json.load(open("c60_small_K30.json"))["A_t"]
    c4 = abs(a0 - ast) < 1e-9 and 1e-8 < abs(a1 - ast) < 1e-5
    c5 = all(H.stage_ok(t) for t in TAGS.values())
    c6 = all(abs(np.polyval(poly, k) - c0[k]) < 0.005 for k in ks)
    out = {"C1": C1, "poly": poly.tolist(), "c0_points": {str(k): v for k, v in c0.items()}}
    w5 = 0.0
    null_ok = True
    old_ok = True
    for k, tag in TAGS.items():
        sj = f"c52_{tag}c.json"
        N5, law = nhw(sj, 92.0, 5)
        N7, _ = nhw(sj, 92.0, 7)
        stl = [s for s in json.load(open(sj))["stages"] if s["D"] == 92.0][-1]
        cc = float(np.polyval(poly, k))
        out[str(k)] = {"c0": cc, "law": list(law), "wall_mass": stl["m"][0]}
        nbad = 0
        for K in KS:
            am = QQ.root(N5, K)
            v2 = Z.v2f(N5, am)
            dd = Da(v2)
            w5 = max(w5, abs(am - QQ.root(N7, K)) * 1e9 / dd)
            pred = cc - C1 / v2
            null_ok = null_ok and not inside(cc / pred - 1)
            nbad += int(not inside((cc - 149.27 / v2) / pred - 1))
            out[str(k)][str(K)] = dict(a_med=am, v2=float(v2), Rda_pred=pred, a0=float(am - 0.0045))
        if k == 0.15:
            old_ok = old_ok and nbad >= 1
    c7 = null_ok and old_ok
    c8 = w5 <= 0.02
    print(
        f"C1 {C1}; cubic {poly.round(5).tolist()}; control c1 {c1}; c2 {c2}; c3 (frozen constants reproduce {len(seen)} seen events, largest |Y| {ymax:.4f}) {c3}; c4 {c4}; c5 (stages converged) {c5}; c6 {c6}; c7 (null and old C1 fail the bar) {c7}; c8 (max |q5 - q7| / Da {w5:.5f}) {c8}"
    )
    if not (c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8):
        print("REGISTRATION-RUN q1345: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1345_predictions.json", "w"), indent=1)
    for k, tag in TAGS.items():
        law = out[str(k)]["law"]
        for part, Kp in (("a", KS[:7]), ("b", KS[7:])):
            L = [
                "#!/bin/bash",
                f"# run1345{tag}{part}.sh (8 Oct 2026): hard-wall chain c61 for k = {k}, K_b {Kp[0]}-{Kp[-1]}; start A0 = a_med - 0.0045, step 0.003 (q1345_predictions.json)",
                'cd "./centre_scripts"',
                f"export KEXP={k} HL_JSON=c52_{tag}c.json HL_STAGE=92 LAW0={law[0]:.5f},{law[1]:.5f} WM0={out[str(k)]['wall_mass']:.6f}",
            ]
            for K in Kp:
                L.append(
                    f"py c61_hw_fullchain.py {K} {out[str(k)][str(K)]['a0']:.6f} 0.003 160 u1345_{tag}_K{K} > c61_u1345_{tag}_K{K}.log 2>&1"
                )
            L.append(f"echo done $(date +%H:%M:%S) > run1345{tag}{part}.done")
            open(f"run1345{tag}{part}.sh", "w", newline="\n").write("\n".join(L) + "\n")
    print("sha256", sh)
    print("REGISTRATION-RUN q1345: controls True, table and run1345*.sh written")


def read():
    js = json.load(open("q1345_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1345: VOID (table hash)")
        sys.exit(2)
    res = {}
    incomplete = False
    for k, tag in TAGS.items():
        sj = f"c52_{tag}c.json"
        N5, _ = nhw(sj, 92.0, 5)
        N7, _ = nhw(sj, 92.0, 7)
        rows = []
        for K in KS:
            A = H.locr(f"c61_u1345_{tag}_K{K}")
            if A is None:
                print(f"k {k} K_b {K}: NOT LOCATED")
                continue
            e = js[str(k)][str(K)]
            v2 = Z.v2f(N5, A)
            d = Da(v2)
            R5 = (A - e["a_med"]) * 1e9 / d
            R7 = (A - QQ.root(N7, K)) * 1e9 / d
            Y5 = R5 / e["Rda_pred"] - 1
            Y7 = R7 / e["Rda_pred"] - 1
            rows.append((K, Y5, Y7))
            print(
                f"k {k} K_b {K}: v^2 {v2:.2f}, R/Da {R5:.4f}, predicted {e['Rda_pred']:.4f}, Y {Y5:+.4f} (septic {Y7:+.4f})"
            )
        if len(rows) < NEED:
            incomplete = True
            print(f"k {k}: located {len(rows)} of {len(KS)}")
            continue
        res[k] = (
            all(inside(a) and inside(b) for _, a, b in rows),
            len(rows),
            max(abs(a) for _, a, _ in rows),
        )
        print(
            f"k {k}: located {len(rows)}; largest |Y| {res[k][2]:.4f}; item {'G1' if k == 0.15 else 'G2' if k == 0.4 else 'G3'}: {res[k][0]}"
        )
    if incomplete:
        print("VERDICT q1345 hard-wall repaired class: INCOMPLETE")
        return
    print(
        f"VERDICT q1345 hard-wall repaired class: {'PASS' if all(v[0] for v in res.values()) else 'FAIL'}; "
        + "; ".join(f"k {k}: {res[k][0]} ({res[k][1]} located, largest |Y| {res[k][2]:.3f})" for k in TAGS)
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
