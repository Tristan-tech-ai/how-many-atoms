"""q1271 (8 Oct 2026, registered in amendment 2046): blind test of the dependence of the median-sum residual on the functional exponent k, for the functional W P^-k with k = 3, new events K_b 28 to 37.
Reading that motivates it (q1269, refreshed 160-bit data): at equal v^2 the residual of the immune problem (k = 2) is 1.800 +- 0.003 times that of the NPMLE problem (k = 1) for K_b 30 to 36, and the capacity residual (k -> 0) is 0.19 to 0.21 times it for K_b 36 to 41. These are the ratios 9 : 5 : 1 of an amplitude c(k) proportional to 4 k + 1.
Prediction: R_3 / R_imm = c(3) / c(2) = 13 / 9 = 1.444 at equal v^2, equivalently R_3 = 2.6 R_NPMLE, and the second difference R_3 - 2 R_imm + R_NPMLE vanishes. A cubic dependence such as (k + 1)(k + 2) would give 20 / 12 = 1.667.
Inputs: the half-line stage D = 92 of the functional k = 3 on the derived bulk law (c33_k3c.json, solver c34, OFF1 = -2.93445, OFF2 = 1.24916; the first versions c33_k3a and c33_k3b on fitted law numbers were not converged, amendment 2047), base law numbers of the spline (0.50382, 0.01310), the median-sum table q1220 (theta 0.3, quintic spline of the half-line counting function through the atoms of depth <= 72), the refreshed immune residuals (K_b 24 to 48 except 38) and NPMLE residuals (K_b 20 to 44 except 38) interpolated in ln R against v^2. No k = 3 amplitude is computed or read.
--registration-run : controls, the table q1271_predictions.json (sha256) with a_med, v^2, the predicted R_3 and the starts a0 = a_med + 1.444 R_imm(v^2) - 0.004.
--read             : items on the located events (files q607_u1271_K<K>.json, key a_t; solver c35 with KEXP = 3).
Located: a_t not null, the last residual of the history below 1e-20 and |a_t - last a| < 1e-9.
Items: T1 at least 8 of the 10 events K_b 28 to 37 are located and have R_3 / R_imm in [1.35, 1.55] at equal v^2 (R_imm interpolated to the v^2 of the k = 3 event), and every located event has R_3 / R_imm < 1.60; T2 at the located events |d3| <= 0.10 with d3 = (R_3 - 2 R_imm + R_NPMLE) / (R_3 - R_imm), all at equal v^2.
Controls (first): c1 the generic chain path c35 at KEXP = 2 reproduces the immune K_b 30 event of c19_u1231 bit-identically (tolerance 1e-12; recorded in amendment 2046), and the default path reproduces the NPMLE K_b 34 event (tolerance 1e-12); c2 the half-line generic path at k = 2 reproduces the immune stages D = 20 and 28 (recorded, 0.0); c3 the item functions accept (1.444, d3 = 0) and reject ratio 1.667, ratio 1.25 and d3 = 0.3; c4 the quintic and septic splines through the k = 3 atoms give a_med within 0.5 percent of the predicted R_3 at each of the ten events (amendment 2048); c5 the stages D = 80 and D = 92 give a_med within 0.5 percent of the predicted R_3 at each event; c6 the immune interpolation at the v^2 of the immune event K_b 30 returns its own residual within 1e-9 relative (interpolation exactness).
Usage: py q1271_linear_in_k_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)"""

import hashlib, json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.interpolate import make_interp_spline
import q1250_quintic_residual as Q
import q1252_order16_quintic as R

C, N, m = Q.C, Q.N, R.m
TARGET = list(range(28, 38))
LAW3 = (0.50382, 0.01310)
CPRED = 13 / 9
LO, HI, CAP = 1.35, 1.55, 1.60
D3 = 0.10
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
H = 1e-3


def v2f(Nf, a):
    return (4 * np.pi * (Nf(a + H) - Nf(a - H)) / (2 * H)) ** 2


def loc(f):
    try:
        j = json.load(open(f + ".json"))
        hl = [x for x in j["hist"] if "res" in x and "a" in x]
        a = j.get("a_t")
        return a if (a is not None and hl and hl[-1]["res"] < 1e-20 and abs(a - hl[-1]["a"]) < 1e-9) else None
    except Exception:
        return None


def nhalf(k, stage_json, Dsel, deg):
    st = [s for s in json.load(open(stage_json))["stages"] if s["D"] == Dsel][-1]
    p = np.array(st["p"][: st["nfree"]])
    p = p[p <= Dsel - 20]
    return Q.make(lambda a: Q.N4(a, LAW3), p, deg)


def refs():
    """immune and NPMLE refreshed residuals (1e-9) and v^2 tables"""
    sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
    pi_ = np.array(sn["p"][: sn["nfree"]])
    pi_ = pi_[pi_ <= 72]
    Ni = Q.make(lambda a: Q.N4(a, (0.22573, 0.06806)), pi_, 5)
    Nn = Q.make(lambda a: Q.N4(a, N.LAW), N.pn, 5)
    ev = {int(e[1]): float(e[0]) for e in json.load(open("q624_immune_table.json")) if int(e[1]) >= 24}
    for K in range(29, 38):
        ev[K] = json.load(open(f"c19_u1231_K{K}.json"))["a_t"]
    for K in range(14, 24):
        ev[K] = json.load(open(f"c19_u1263_K{K}.json"))["a_t"]
    for K in range(36, 49):
        ev[K] = json.load(open(f"c19_u1267_K{K}.json"))["a_t"]
    npm = {K: gv(f) for K, f in m.NPM_FILES.items() if 6 <= K <= 52}
    for K in range(34, 49):
        npm[K] = json.load(open(f"q607_u1268_K{K}.json"))["a_t"]
    di = {K: ((a - Q.root(Ni, K)) * 1e9, v2f(Ni, a)) for K, a in ev.items() if 24 <= K <= 48 and K != 38}
    dn = {K: ((a - Q.root(Nn, K)) * 1e9, v2f(Nn, a)) for K, a in npm.items() if 20 <= K <= 44 and K != 38}
    return di, dn


def curve(d):
    ks = sorted(d)
    xs = np.array([d[K][1] for K in ks])
    ys = np.array([np.log(d[K][0]) for K in ks])
    o = np.argsort(xs)
    return lambda x: float(np.exp(np.interp(x, xs[o], ys[o]))) if xs.min() <= x <= xs.max() else None


def t1(r):
    return LO <= r <= HI and r < CAP


def t2(d3):
    return abs(d3) <= D3


def regrun():
    di, dn = refs()
    Ri, Rn = curve(di), curve(dn)
    x30 = di[30][1]
    c6 = abs(Ri(x30) / di[30][0] - 1) < 1e-9
    c3 = t1(CPRED) and t2(0.0) and not t1(1.667) and not t1(1.25) and not t2(0.3)
    N5 = nhalf(3, "c33_k3c.json", 92.0, 5)
    N7 = nhalf(3, "c33_k3c.json", 92.0, 7)
    N80 = nhalf(3, "c33_k3c.json", 80.0, 5)
    rp = {K: CPRED * Ri(v2f(N5, Q.root(N5, K))) for K in TARGET}
    dd4 = max(abs(Q.root(N5, K) - Q.root(N7, K)) * 1e9 / rp[K] for K in TARGET)
    dd5 = max(abs(Q.root(N5, K) - Q.root(N80, K)) * 1e9 / rp[K] for K in TARGET)
    c4, c5 = dd4 <= 0.005, dd5 <= 0.005
    k2 = (
        abs(json.load(open("q607_ctl_k2_K30.json"))["a_t"] - json.load(open("c19_u1231_K30.json"))["a_t"])
        < 1e-12
        and abs(
            json.load(open("q607_ctl_k1_K34.json"))["a_t"] - json.load(open("q607_u1268_K34.json"))["a_t"]
        )
        < 1e-12
    )
    print(
        f"control c1 (generic chain path k = 2 and default k = 1 reproduce stored events): {k2}; c3 (item functions): {c3}; c4 (quintic vs septic a_med, max relative to R_pred {dd4:.5f}, bar 0.005): {c4}; c5 (D = 80 vs 92, max relative to R_pred {dd5:.2e}, bar 0.005): {c5}; c6 (immune interpolation exact): {c6}"
    )
    if not (k2 and c3 and c4 and c5 and c6):
        print("REGISTRATION-RUN q1271: VOID (control)")
        sys.exit(2)
    out = {}
    for K in TARGET:
        am = Q.root(N5, K)
        x = v2f(N5, am)
        ri = Ri(x)
        rn = Rn(x)
        if ri is None or rn is None:
            print(f"K_b {K}: v^2 {x:.2f} outside the reference range")
            sys.exit(2)
        out[str(K)] = dict(
            a_med=am,
            v2=float(x),
            R_imm=ri,
            R_npmle=rn,
            R3_pred=CPRED * ri,
            a0=float(am + CPRED * ri * 1e-9 - 0.004),
        )
        print(
            f"K_b {K}: v^2 {x:.2f}, a_med {am:.10f}, R_imm {ri:.4f}, R_NPMLE {rn:.4f}, predicted R_3 {CPRED * ri:.4f}e-9 (bar {LO * ri:.4f} to {HI * ri:.4f}), start a0 = {am + CPRED * ri * 1e-9 - 0.004:.6f}"
        )
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1271_predictions.json", "w"), indent=1)
    print("sha256", sh)
    print("REGISTRATION-RUN q1271: controls True, table written")


def read():
    js = json.load(open("q1271_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1271: VOID (table hash)")
        sys.exit(2)
    di, dn = refs()
    Ri, Rn = curve(di), curve(dn)
    N5 = nhalf(3, "c33_k3c.json", 92.0, 5)
    rows = {}
    for K in TARGET:
        a = loc(f"q607_u1271_K{K}")
        if a is None:
            print(f"K_b {K}: NOT LOCATED")
            continue
        am = js[str(K)]["a_med"]
        r3 = (a - am) * 1e9
        x = v2f(N5, a)
        ri = Ri(x)
        rn = Rn(x)
        rows[K] = (r3, ri, rn, r3 / ri, (r3 - 2 * ri + rn) / (r3 - ri))
        print(
            f"K_b {K}: R_3 {r3:+.4f}e-9 (predicted {js[str(K)]['R3_pred']:.4f}), R_imm {ri:.4f}, R_NPMLE {rn:.4f}, R_3/R_imm {rows[K][3]:.4f}, R_3/R_NPMLE {r3 / rn:.4f}, d3 {rows[K][4]:+.4f}"
        )
    n = len(rows)
    a1 = n >= 8 and sum(t1(r[3]) for r in rows.values()) >= 8 and all(r[3] < CAP for r in rows.values())
    a2 = n >= 8 and all(t2(r[4]) for r in rows.values())
    print(f"located {n} of {len(TARGET)}; item T1: {a1}; item T2: {a2}")
    print(
        f"VERDICT q1271 linear-in-k class: {'PASS' if (a1 and a2) else ('INCOMPLETE' if n < 8 else 'FAIL')}; located {n} of {len(TARGET)}; T1 {a1}; T2 {a2}"
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
