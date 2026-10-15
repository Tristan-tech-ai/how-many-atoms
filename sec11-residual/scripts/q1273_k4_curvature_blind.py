"""q1273 (8 Oct 2026, registered in amendment 2052): blind test of the curvature of the amplitude c(k) of the median-sum residual in the functional exponent k, at k = 4 (functional W P^-4, never solved before), events K_b 29 to 36.
Facts behind it (amendments 2045, 2051): at equal v^2, c(1) = 1, c(2) = 1.800 to 1.803, c(3) = 2.549 to 2.566 (K_b 30 to 36), so the increments are 0.80 and 0.75 to 0.76 and the second difference of c over k = 1, 2, 3 is about -0.045. Two continuations to k = 4 are registered:
 H_q  (constant second difference, the quadratic through k = 1, 2, 3): c(4) - c(3) = 0.755 - 0.045 = 0.710, so R_4 / R_3 = (2.555 + 0.710) / 2.555 = 1.278;
 H_l  (constant increment, the straight line): c(4) - c(3) = 0.80 (the increment between k = 1 and k = 2), c(4) = 3.355, so R_4 / R_3 = 1.313 with c(3) = 2.555; with the line through the three-problem values (0.2 + 0.8 k) it is 3.4 / 2.6 = 1.308. The value 1.308 is registered for H_l.
Items (each on the mean m over the located events of R_4 / R_3 at equal v^2, R_3 interpolated in ln R against v^2 from the ten measured events of q1271 (q607_u1271_K28..37.json); an event enters only if its v^2 lies inside the range of the reference): H_q |m - 1.278| <= 0.012; H_l |m - 1.308| <= 0.012. The bands do not overlap (1.266 to 1.290 and 1.296 to 1.320). Class: QUADRATIC if only H_q, LINE if only H_l, NEITHER if both False, INCOMPLETE if fewer than 7 of the 8 events are located and in range. Priors: H_q 0.35, H_l 0.2, neither 0.45.
Reading without item: d4 = (R_4 - 2 R_3 + R_imm) / (R_4 - R_3) (H_q gives about -0.064, H_l gives 0), the ratios per event, the trend with K_b.
Inputs: half-line stage c33_k4c.json (solver c34 via the driver c36: derived bulk law, OFF1 -2.72486, OFF2 1.20012, D = 92), base law numbers of the spline fitted in this script to the atoms of depth 14 to 30 of that stage (stored in the table and used as D0, NU0 of the chain seed), median-sum table q1220; no k = 4 amplitude is computed or read.
--registration-run : controls, the table q1273_predictions.json (sha256) with a_med, v^2, R_3(v^2), the predicted R_4 for H_q and the starts a0 = a_med + 1.278 R_3 - 0.004.
--read             : items on the located events (files q607_u1273_K<K>.json, key a_t; solver c35 with KEXP = 4).
Located: a_t not null, the last residual below 1e-20 and |a_t - last a| < 1e-9.
Controls (first): c1 the driver c36 reproduces the k = 3 stage: the a_med of K_b 28 to 37 from c33_k3pc.json and c33_k3c.json differ by at most 0.5 percent of the k = 3 residual; c2 the chain solver at KEXP 2 and 1 is bit-identical to the stored events (amendment 2046, recorded); c3 the item functions accept 1.278 for H_q and 1.308 for H_l, reject 1.30 for H_q and 1.28 for H_l, and give NEITHER at 1.293; c4 the quintic and septic splines give a_med within 0.5 percent of the predicted R_4 at each event; c5 the stages D = 80 and D = 92 give a_med within 0.5 percent of the predicted R_4; c6 the reference interpolation returns the measured R_3 at the v^2 of the k = 3 event K_b 33 within 1e-9 relative.
Usage: py q1273_k4_curvature_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)"""

import hashlib, json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.optimize import brentq, least_squares
import q1250_quintic_residual as Q
import q1271_linear_in_k_blind as Z

TARGET = list(range(29, 37))
CQ, CL, TOL = 1.278, 1.308, 0.012
H = 1e-3


def loc(f):
    return Z.loc(f)


def fitlaw(stage_json, Dsel):
    st = [s for s in json.load(open(stage_json))["stages"] if s["D"] == Dsel][-1]
    p = np.array(st["p"][: st["nfree"]])
    j = np.arange(1, len(p) + 1) - 0.5
    m = (p >= 14) & (p <= 30)
    r = least_squares(lambda q: np.array([Q.N4(x, (q[0], q[1])) for x in p[m]]) - j[m], [0.5, 0.0])
    return (float(r.x[0]), float(r.x[1]))


def nhalf(stage_json, Dsel, deg, law):
    st = [s for s in json.load(open(stage_json))["stages"] if s["D"] == Dsel][-1]
    p = np.array(st["p"][: st["nfree"]])
    p = p[p <= Dsel - 20]
    return Q.make(lambda a: Q.N4(a, law), p, deg)


def v2f(Nf, a):
    return (4 * np.pi * (Nf(a + H) - Nf(a - H)) / (2 * H)) ** 2


def ref3():
    """measured R_3 (1e-9) and v^2 at the ten k = 3 events"""
    N3 = Z.nhalf(3, "c33_k3c.json", 92.0, 5)
    d = {}
    for K in Z.TARGET:
        a = loc(f"q607_u1271_K{K}")
        if a is None:
            print(f"reference event K_b {K} missing")
            sys.exit(2)
        d[K] = ((a - Q.root(N3, K)) * 1e9, v2f(N3, a))
    return d


def curve(d):
    ks = sorted(d)
    xs = np.array([d[K][1] for K in ks])
    ys = np.array([np.log(d[K][0]) for K in ks])
    o = np.argsort(xs)
    return lambda x: float(np.exp(np.interp(x, xs[o], ys[o]))) if xs.min() <= x <= xs.max() else None


def cls(m):
    hq = abs(m - CQ) <= TOL
    hl = abs(m - CL) <= TOL
    return hq, hl, ("QUADRATIC" if hq and not hl else "LINE" if hl and not hq else "NEITHER")


def regrun():
    law = fitlaw("c33_k4c.json", 92.0)
    N5 = nhalf("c33_k4c.json", 92.0, 5, law)
    N7 = nhalf("c33_k4c.json", 92.0, 7, law)
    N80 = nhalf("c33_k4c.json", 80.0, 5, law)
    d3 = ref3()
    R3 = curve(d3)
    K33 = d3[33]
    c6 = abs(R3(K33[1]) / K33[0] - 1) < 1e-9
    rp = {}
    for K in TARGET:
        am = Q.root(N5, K)
        x = v2f(N5, am)
        r3 = R3(x)
        if r3 is None:
            print(f"K_b {K}: v^2 {x:.2f} outside the k = 3 reference range")
            sys.exit(2)
        rp[K] = (am, x, r3)
    dd4 = max(abs(Q.root(N5, K) - Q.root(N7, K)) * 1e9 / (CQ * rp[K][2]) for K in TARGET)
    dd5 = max(abs(Q.root(N5, K) - Q.root(N80, K)) * 1e9 / (CQ * rp[K][2]) for K in TARGET)
    N3p = Z.nhalf(3, "c33_k3pc.json", 92.0, 5)
    N3c = Z.nhalf(3, "c33_k3c.json", 92.0, 5)
    dd1 = max(abs(Q.root(N3p, K) - Q.root(N3c, K)) * 1e9 / d3[K][0] for K in Z.TARGET)
    a, b, c = cls(1.278), cls(1.308), cls(1.293)
    c3 = (
        a[0]
        and not a[1]
        and b[1]
        and not b[0]
        and cls(1.30)[0] is False
        and cls(1.28)[1] is False
        and c[2] == "NEITHER"
    )
    k21 = (
        abs(json.load(open("q607_ctl_k2_K30.json"))["a_t"] - json.load(open("c19_u1231_K30.json"))["a_t"])
        < 1e-12
    )
    c1, c4, c5 = dd1 <= 0.005, dd4 <= 0.005, dd5 <= 0.005
    print(
        f"control c1 (driver reproduces k = 3, max relative a_med difference {dd1:.2e}): {c1}; c2 (chain solver bit-identical at k = 2): {k21}; c3 (item functions): {c3}; c4 (quintic vs septic, max relative to R_pred {dd4:.5f}): {c4}; c5 (D = 80 vs 92 {dd5:.2e}): {c5}; c6 (reference interpolation exact): {c6}"
    )
    if not (c1 and k21 and c3 and c4 and c5 and c6):
        print("REGISTRATION-RUN q1273: VOID (control)")
        sys.exit(2)
    out = {"law": list(law)}
    for K in TARGET:
        am, x, r3 = rp[K]
        out[str(K)] = dict(
            a_med=am, v2=float(x), R3=r3, R4_Hq=CQ * r3, R4_Hl=CL * r3, a0=float(am + CQ * r3 * 1e-9 - 0.004)
        )
        print(
            f"K_b {K}: v^2 {x:.2f}, a_med {am:.10f}, R_3 {r3:.4f}, predicted R_4 {CQ * r3:.4f} (H_q) / {CL * r3:.4f} (H_l), start a0 = {am + CQ * r3 * 1e-9 - 0.004:.6f}"
        )
    print(f"base law numbers of the k = 4 spline and chain seed: d0 {law[0]:.5f}, nu0 {law[1]:.5f}")
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1273_predictions.json", "w"), indent=1)
    print("sha256", sh)
    print("REGISTRATION-RUN q1273: controls True, table written")


def read():
    js = json.load(open("q1273_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1273: VOID (table hash)")
        sys.exit(2)
    law = tuple(js["law"])
    N5 = nhalf("c33_k4c.json", 92.0, 5, law)
    R3 = curve(ref3())
    di, dn = Z.refs()
    Ri = Z.curve(di)
    rows = {}
    for K in TARGET:
        a = loc(f"q607_u1273_K{K}")
        if a is None:
            print(f"K_b {K}: NOT LOCATED")
            continue
        r4 = (a - js[str(K)]["a_med"]) * 1e9
        x = v2f(N5, a)
        r3 = R3(x)
        ri = Ri(x)
        if r3 is None:
            print(f"K_b {K}: v^2 {x:.2f} outside the reference range")
            continue
        rows[K] = (r4, r3, ri, r4 / r3, (r4 - 2 * r3 + ri) / (r4 - r3))
        print(
            f"K_b {K}: R_4 {r4:+.4f}e-9 (H_q {js[str(K)]['R4_Hq']:.4f}, H_l {js[str(K)]['R4_Hl']:.4f}), R_3 {r3:.4f}, R_imm {ri:.4f}, R_4/R_3 {rows[K][3]:.4f}, R_4/R_imm {r4 / ri:.4f}, d4 {rows[K][4]:+.4f}"
        )
    n = len(rows)
    if n < 7:
        print(f"located and in range {n} of {len(TARGET)}")
        print(f"VERDICT q1273 curvature class: INCOMPLETE; events {n} of {len(TARGET)}")
        return
    m = float(np.mean([r[3] for r in rows.values()]))
    hq, hl, cl = cls(m)
    print(
        f"events {n} of {len(TARGET)}; mean R_4/R_3 {m:.4f}; item H_q (|m - {CQ}| <= {TOL}): {hq}; item H_l (|m - {CL}| <= {TOL}): {hl}"
    )
    print(f"VERDICT q1273 curvature class: {cl}; events {n}; mean {m:.4f}; H_q {hq}; H_l {hl}")


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
