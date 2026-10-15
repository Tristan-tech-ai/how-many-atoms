"""q1287 (8 Oct 2026, registered in amendment 2067): blind test of the closure R_k = sigma_k * Da(v^2) for the functional k = 3 at K_b 14 to 23 (v^2 about 69 to 88; no k = 3 event exists there).
Da(v^2) = |Im S_+(1/v^2)| / (2 rho) in 1e-9, rho = v / (4 pi), S_+ the lateral Borel sum (theta 0.3, diagonal Pade [8/8], e_2 .. e_18) of q1285. sigma_3 = the mean over the measured events K_b 28 to 37 (q1271) of R_3 / Da at the v^2 of the event.
Items: A1 at least 8 of the 10 events located and |R_3 / (sigma_3 Da) - 1| <= 0.12 at every located event; A2 R_3 / R_2 in [1.36, 1.44] at equal v^2 at every located event (R_2: the refreshed immune residual, events K_b 14 to 28). Class PASS if A1 and A2; INCOMPLETE if fewer than 8 located; otherwise FAIL.
Located: a_t not null, last residual below 1e-18, |a_t - last a| < 1e-9 (the bound registered in amendment 2063 for solver floors).
--registration-run : controls, the table q1287_predictions.json (sha256) and run1287.sh; --read : the items; last line 'VERDICT q1287 closure class: ...'.
Controls (first): c1 item functions accept ratios 1.0 and 0.9 and reject 0.85 and 1.15 (A1) and accept 1.40 and reject 1.30 and 1.50 (A2); c2 Da(115) = 0.21082 within 1e-4 relative; c3 quintic against septic a_med within 0.5 percent of the predicted residual at each event; c4 D = 80 against 92 likewise; c5 chain solver bit-identical at k = 2 (recorded); c6 sigma_3 from the events K_b 28 to 37 lies in 6.6 to 7.0.
Usage: py q1287_stokes_closure_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)
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
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
Bf, _p, _q = T.machine(T.coef_series(), 8, 8)
TARGET = list(range(14, 24))
LO3, HI3 = 1.36, 1.44
TOL1 = 0.12


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


def locr(f):
    try:
        j = json.load(open(f + ".json"))
        hl = [x for x in j["hist"] if "res" in x and "a" in x]
        a = j.get("a_t")
        return a if (a is not None and hl and hl[-1]["res"] < 1e-18 and abs(a - hl[-1]["a"]) < 1e-9) else None
    except Exception:
        return None


def a1(r):
    return abs(r - 1) <= TOL1


def a2(c):
    return LO3 <= c <= HI3


def sigma3():
    N3 = Z.nhalf(3, "c33_k3c.json", 92.0, 5)
    out = []
    for K in range(28, 38):
        a = locr(f"q607_u1271_K{K}")
        if a is None:
            a = json.load(open(f"q607_u1271_K{K}.json"))["a_t"]
        v2 = Z.v2f(N3, a)
        out.append(((a - Q.root(N3, K)) * 1e9) / Da(v2))
    return float(np.mean(out)), N3


def imm_ref():
    sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
    pi_ = np.array(sn["p"][: sn["nfree"]])
    pi_ = pi_[pi_ <= 72]
    Ni = Q.make(lambda a: Q.N4(a, (0.22573, 0.06806)), pi_, 5)
    d = {}
    for K in range(14, 24):
        a = json.load(open(f"c19_u1263_K{K}.json"))["a_t"]
        d[K] = ((a - Q.root(Ni, K)) * 1e9, Z.v2f(Ni, a))
    for e in json.load(open("q624_immune_table.json")):
        K = int(e[1])
        if 24 <= K <= 28:
            a = float(e[0])
            d[K] = ((a - Q.root(Ni, K)) * 1e9, Z.v2f(Ni, a))
    return Z.curve(d)


def regrun():
    s3, N3 = sigma3()
    N7 = Z.nhalf(3, "c33_k3c.json", 92.0, 7)
    N80 = Z.nhalf(3, "c33_k3c.json", 80.0, 5)
    c1 = a1(1.0) and a1(0.9) and not a1(0.85) and not a1(1.15) and a2(1.40) and not a2(1.30) and not a2(1.50)
    c2 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
    c5 = (
        abs(json.load(open("q607_ctl_k2_K30.json"))["a_t"] - json.load(open("c19_u1231_K30.json"))["a_t"])
        < 1e-12
    )
    c6 = 6.6 < s3 < 7.0
    out = {"sigma3": s3}
    w3 = w4 = 0.0
    for K in TARGET:
        am = Q.root(N3, K)
        v2 = Z.v2f(N3, am)
        da = Da(v2)
        rp = s3 * da
        w3 = max(w3, abs(am - Q.root(N7, K)) * 1e9 / rp)
        w4 = max(w4, abs(am - Q.root(N80, K)) * 1e9 / rp)
        out[str(K)] = dict(a_med=am, v2=float(v2), Da=da, R_pred=rp, a0=float(am + rp * 1e-9 - 0.004))
        print(
            f"K_b {K}: v^2 {v2:.2f}, a_med {am:.10f}, Da {da:.5f}e-9, predicted R_3 {rp:.4f}e-9, start a0 = {am + rp * 1e-9 - 0.004:.6f}"
        )
    c3, c4 = w3 <= 0.005, w4 <= 0.005
    print(
        f"control c1 {c1}; c2 (Da(115)) {c2}; c3 (quintic vs septic, max relative {w3:.5f}) {c3}; c4 (D = 80 vs 92 {w4:.2e}) {c4}; c5 (chain bit-identical at k = 2) {c5}; c6 (sigma_3 = {s3:.4f} in 6.6 to 7.0) {c6}"
    )
    if not (c1 and c2 and c3 and c4 and c5 and c6):
        print("REGISTRATION-RUN q1287: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1287_predictions.json", "w"), indent=1)
    L = [
        "#!/bin/bash",
        "# run1287.sh (8 Oct 2026): full-chain locators c35 (KEXP 3) K_b 14 to 23 of amendments 2067 and 2068; starts from q1287_predictions.json",
        'cd "./centre_scripts"',
        "export KEXP=3 HL_JSON=c33_k3c.json HL_STAGE=92 D0=0.50382 NU0=0.01310",
    ]
    for K in TARGET:
        L.append(
            f"py c35_chain_k.py {K} {K // 2} 160 {out[str(K)]['a0']:.6f} 0.003 u1287_K{K} > q607_u1287_K{K}.log 2>&1"
        )
    L.append("echo done $(date +%H:%M:%S) > run1287.done")
    open("run1287.sh", "w", newline="\n").write("\n".join(L) + "\n")
    print("sha256", sh)
    print("REGISTRATION-RUN q1287: controls True, table and run1287.sh written")


def read():
    js = json.load(open("q1287_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1287: VOID (table hash)")
        sys.exit(2)
    N3 = Z.nhalf(3, "c33_k3c.json", 92.0, 5)
    R2 = imm_ref()
    rows = {}
    for K in TARGET:
        a = locr(f"q607_u1287_K{K}")
        if a is None:
            print(f"K_b {K}: NOT LOCATED")
            continue
        v2 = Z.v2f(N3, a)
        r3 = (a - js[str(K)]["a_med"]) * 1e9
        da = Da(v2)
        r2 = R2(v2)
        if r2 is None:
            print(f"K_b {K}: v^2 {v2:.2f} outside the immune reference")
            continue
        rows[K] = (r3, r3 / (js["sigma3"] * da), r3 / r2)
        print(
            f"K_b {K}: R_3 {r3:+.4f}e-9, Da {da:.5f}, R_3 / (sigma_3 Da) {rows[K][1]:.4f}, R_3 / R_2 {rows[K][2]:.4f}"
        )
    n = len(rows)
    i1 = n >= 8 and all(a1(r[1]) for r in rows.values())
    i2 = n >= 8 and all(a2(r[2]) for r in rows.values())
    print(
        f"located and in range {n} of {len(TARGET)}; sigma_3 = {js['sigma3']:.4f}; item A1: {i1}; item A2: {i2}"
    )
    print(
        f"VERDICT q1287 closure class: {'PASS' if (i1 and i2) else ('INCOMPLETE' if n < 8 else 'FAIL')}; located {n} of {len(TARGET)}; A1 {i1}; A2 {i2}"
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
