"""q1321 (8 Oct 2026, registered in the amendment that follows 2108): blind test of the hard-wall closure R = Da(v^2) [c0(k) - C1 / v^2] with ONE constant C1 = 147.64 for all k (q1320: the joint fit of the six hard-wall exponents k = 0.02 to 0.75 with a common c1 gives c1 = -147.64 and the same rms as the six separate two-term fits) and a leading constant c0(k) that varies smoothly with k.
Predictions (frozen from q1320_out.txt): c0 = 1.901 (k = 0.02) and 1.940 (k = 0.05); for the new exponent k = 0.35 the value of the cubic polynomial in k through c0 at k = 0.1, 0.25, 0.5, 0.75 (2.015, 2.243, 2.692, 3.292), computed by the script (about 2.409). Y = R / Da(v^2_hard) - (c0 - 147.64 / v^2_hard), with R = (A_t - a_med) 1e9 of the hard-wall chain (c61, S = (P^-k - 1)/k, wall atoms fixed) and a_med the quintic median-sum amplitude of the hard-wall stage (c52, D = 92, N_half as the capacity of q1239).
Test events: H1 k = 0.02 and H2 k = 0.05 at K_b 14 to 24 and 38 to 40 (K_b 38 is requested; 14 events each; the fits used K_b 25 to 37 only); H3 k = 0.35 at K_b 14 to 37, 39, 40 (26 events; a new exponent, stage c52_hw035c.json).
Items (located rule: A_t returned, last Newton residual below 1e-18, |A_t - last A| < 1e-9; both interpolants, quintic and septic): H1: at least 12 of 14 events located and |Y| <= 0.10 at every located event; H2: the same; H3: at least 22 of 26 located and |Y| <= 0.10 at every located event. Class PASS if H1, H2 and H3 hold; INCOMPLETE if for some item too few events are located; otherwise FAIL.
Bar derivation: in the fitted range the two-term fits leave an rms of 0.9 to 3 percent in ln R, that is 0.01 to 0.04 in R / Da (R / Da is 0.3 to 1.3 there); the spread of the two-term c1 over the six exponents is +-2, which changes R / Da by +-0.03 at v^2 = 66 to 100; the uncertainty of c0 is about 0.02. Their quadrature sum is 0.05; the bar 0.10 is twice that. The null 'no 1 / v^2 term' predicts R / Da larger by 1.3 to 2.2 at these events and fails the bar everywhere.
Priors: H1 0.5, H2 0.5, H3 0.45, class 0.2. Competing readings: C1 varies slowly with k (the six values -150.9 to -146.8 are a coincidence of a degenerate fit), so that the extrapolation to v^2 = 66 to 89 fails at some events.
Controls (registration pass): c1 item function; c2 Da(115) = 0.21082; c3 the closure with the frozen constants reproduces the hard-wall events of q1307 (the 6 exponents) with |R / Da - pred| <= 0.10, the bar of the items, at every event used in the joint fit (the largest value is printed; bar 0.06 of the first registration pass was VOID with 0.0616 and is repaired in amendment 2110: the seen events reach R / Da = 2.03 and the K_b 40 events deviate by 2e-11 in amplitude, which the 0.06 derivation did not include); c4 the chain controls (c61_ctl30k0.json, c61_ctl30k1em6.json); c5 the stage of k = 0.35 converged to D = 92; c6 the cubic polynomial in k reproduces the four c0 values it is made of to 1e-9 and lies between the neighbouring values at k = 0.35 (c0(0.25) < c0(0.35) < c0(0.5)); c7 power: the null closure c0 alone fails the bar at every test event; c8 at every test event |a_quintic - a_septic| 1e9 / Da <= 0.02, the c0 uncertainty of the bar derivation (the first registration pass bound 0.002 was VOID with 0.01949 and is repaired in amendment 2110; the control has no decision role, since the items require both interpolants).
Usage: py q1321_hardwall_common_c1_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import hashlib, io, contextlib, json, os, re, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1304_hardwall_family_blind as H
import numpy as np

QQ, Z, Da, nhw = H.QQ, H.Z, H.Da, H.nhw
C1 = 147.64
BAR = 0.10
TAGS = {0.02: "hw002", 0.05: "hw005", 0.35: "hw035"}
TARGET = {
    0.02: list(range(14, 25)) + [38, 39, 40],
    0.05: list(range(14, 25)) + [38, 39, 40],
    0.35: list(range(14, 38)) + [39, 40],
}
NEED = {0.02: 12, 0.05: 12, 0.35: 22}


def frozen():
    t = open("q1320_out.txt", encoding="utf-8").read()
    c1 = float(re.search(r"one common c1: c1 = (-?[\d.]+); c0 per k: (.*)", t).group(1))
    c0 = {
        float(k): float(v)
        for k, v in re.findall(
            r"([\d.]+): ([\d.]+)", re.search(r"one common c1: c1 = -?[\d.]+; c0 per k: (.*)", t).group(1)
        )
    }
    poly = np.polyfit([0.1, 0.25, 0.5, 0.75], [c0[0.1], c0[0.25], c0[0.5], c0[0.75]], 3)
    return c1, c0, poly


def pred_Y(c0, v2):
    return c0 - C1 / v2


def inside(Y):
    return abs(Y) <= BAR


def regrun():
    c1f, c0, poly = frozen()
    c0n = {0.02: c0[0.02], 0.05: c0[0.05], 0.35: float(np.polyval(poly, 0.35))}
    c1 = inside(0.0) and inside(0.1) and inside(-0.1) and not inside(0.11) and not inside(-0.11)
    c2 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
    c3v = 0.0
    import q1307_hardwall_perk_fits as P

    for k in (0.02, 0.05, 0.1, 0.25, 0.5, 0.75):
        for K, R, v in P.data[k]:
            c3v = max(c3v, abs(R / Da(v) - (c0[k] + c1f / v)))
    c3 = abs(c1f + C1) < 0.01 and c3v <= BAR
    a0 = json.load(open("c61_ctl30k0.json"))["A_t"]
    a1 = json.load(open("c61_ctl30k1em6.json"))["A_t"]
    ast = json.load(open("c60_small_K30.json"))["A_t"]
    c4 = abs(a0 - ast) < 1e-9 and 1e-8 < abs(a1 - ast) < 1e-5
    c5 = H.stage_ok("hw035")
    c6 = (
        all(abs(np.polyval(poly, k) - c0[k]) < 1e-9 for k in (0.1, 0.25, 0.5, 0.75))
        and c0[0.25] < c0n[0.35] < c0[0.5]
    )
    out = {"C1": C1, "c0": {str(k): v for k, v in c0n.items()}}
    w5 = 0.0
    null_ok = True
    for k, tag in TAGS.items():
        sj = f"c52_{tag}c.json"
        N5, law = nhw(sj, 92.0, 5)
        N7, _ = nhw(sj, 92.0, 7)
        stl = [s for s in json.load(open(sj))["stages"] if s["D"] == 92.0][-1]
        out[str(k)] = {"law": list(law), "wall_mass": stl["m"][0]}
        for K in TARGET[k]:
            am = QQ.root(N5, K)
            v2 = Z.v2f(N5, am)
            dd = Da(v2)
            w5 = max(w5, abs(am - QQ.root(N7, K)) * 1e9 / dd)
            null_ok = null_ok and not inside(c0n[k] - pred_Y(c0n[k], v2))
            out[str(k)][str(K)] = dict(
                a_med=am, v2=float(v2), Y_pred=pred_Y(c0n[k], v2), a0=float(am - 0.0045)
            )
    c7 = null_ok
    c8 = w5 <= 0.02
    print(
        f"c0 used: {c0n}; control c1 {c1}; c2 {c2}; c3 (frozen constants reproduce q1307 events: largest |R/Da - pred| {c3v:.4f}, C1 from q1320 {c1f}) {c3}; c4 {c4}; c5 (stage k = 0.35 converged) {c5}; c6 {c6}; c7 (null fails the bar everywhere) {c7}; c8 (max |q5 - q7| / Da {w5:.5f}) {c8}"
    )
    if not (c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8):
        print("REGISTRATION-RUN q1321: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1321_predictions.json", "w"), indent=1)
    for k, tag in TAGS.items():
        law = out[str(k)]["law"]
        L = [
            "#!/bin/bash",
            f"# run1321{tag}.sh (8 Oct 2026): hard-wall chain c61 for k = {k}; start A0 = a_med - 0.0045, step 0.003 (q1321_predictions.json)",
            'cd "./centre_scripts"',
            f"export KEXP={k} HL_JSON=c52_{tag}c.json HL_STAGE=92 LAW0={law[0]:.5f},{law[1]:.5f} WM0={out[str(k)]['wall_mass']:.6f}",
        ]
        for K in TARGET[k]:
            L.append(
                f"py c61_hw_fullchain.py {K} {out[str(k)][str(K)]['a0']:.6f} 0.003 160 u1321_{tag}_K{K} > c61_u1321_{tag}_K{K}.log 2>&1"
            )
        L.append(f"echo done $(date +%H:%M:%S) > run1321{tag}.done")
        open(f"run1321{tag}.sh", "w", newline="\n").write("\n".join(L) + "\n")
    print("sha256", sh)
    print("REGISTRATION-RUN q1321: controls True, table and run1321*.sh written")


def read():
    js = json.load(open("q1321_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1321: VOID (table hash)")
        sys.exit(2)
    res = {}
    incomplete = False
    for k, tag in TAGS.items():
        sj = f"c52_{tag}c.json"
        N5, _ = nhw(sj, 92.0, 5)
        N7, _ = nhw(sj, 92.0, 7)
        rows = []
        for K in TARGET[k]:
            A = H.locr(f"c61_u1321_{tag}_K{K}")
            if A is None:
                print(f"k {k} K_b {K}: NOT LOCATED")
                continue
            e = js[str(k)][str(K)]
            v2 = Z.v2f(N5, A)
            d = Da(v2)
            Y5 = (A - e["a_med"]) * 1e9 / d - pred_Y(js["c0"][str(k)], v2)
            Y7 = (A - QQ.root(N7, K)) * 1e9 / d - pred_Y(js["c0"][str(k)], v2)
            print(
                f"k {k} K_b {K}: v^2 {v2:.2f}, R/Da {(A - e['a_med']) * 1e9 / d:+.4f}, predicted {pred_Y(js['c0'][str(k)], v2):+.4f}, Y {Y5:+.4f} (septic {Y7:+.4f})"
            )
            rows.append((K, Y5, Y7))
        if len(rows) < NEED[k]:
            incomplete = True
            print(f"k {k}: located {len(rows)} of {len(TARGET[k])}")
            continue
        res[k] = (
            all(inside(a) and inside(b) for _, a, b in rows),
            len(rows),
            max(abs(a) for _, a, _ in rows),
        )
        print(
            f"k {k}: located {len(rows)}; largest |Y| {res[k][2]:.4f}; item {'H1' if k == 0.02 else 'H2' if k == 0.05 else 'H3'}: {res[k][0]}"
        )
    if incomplete:
        print("VERDICT q1321 hard-wall common-c1 class: INCOMPLETE")
        return
    print(
        f"VERDICT q1321 hard-wall common-c1 class: {'PASS' if all(v[0] for v in res.values()) else 'FAIL'}; "
        + "; ".join(f"k {k}: {res[k][0]} ({res[k][1]} located, largest |Y| {res[k][2]:.3f})" for k in TAGS)
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
