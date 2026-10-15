"""q1304 (8 Oct 2026, registered in the amendment that follows 2086): blind test of the statement that the boundary condition at the edge (a fixed atom on the amplitude bound, a whole-line KKT integral) multiplies the Stokes residual of the family by a factor that stays near the value 2.8 found for the capacity against the extrapolated free-edge family (amendment 2084).
Problem: the hard-wall family of c52 / c61, S = (P^-k - 1)/k, atoms on [-A, A] with atoms fixed at +-A; k = 0 is the capacity. Events: K_b 25 to 37 (13 per exponent) for k = 0.02, 0.05, 0.1, 0.25.
Quantity: sigma_hard = R_hard / Da(v^2_hard), R_hard = (A_t - a_med) * 1e9, a_med the quintic median-sum amplitude of the hard-wall half-line stage (c52, D = 92; N_half = N_law(a; CAPL) + quintic spline through the atoms with depth <= 72, wall atom counted as j = 1, exactly as the capacity of q1239 / q1250). Reference: sigma_free = R / Da of the free-edge family at the same k and K_b, from q1301b_out.txt (the located events of amendments 2082 and 2084). Ratio rho = sigma_hard / sigma_free.
Items (test events K_b in {29, 31, 33, 35, 37}, located on both sides; located rule: the chain returns A_t, last Newton residual below 1e-18, |A_t - last A| < 1e-9; both interpolants, quintic and septic, must satisfy the item):
 W1: for each k in {0.02, 0.05, 0.1}: at least 4 of the 5 test events located and 2.0 <= rho <= 4.0 at every located one.
 W2: for k = 0.25: at least 4 of 5 located and 1.5 <= rho <= 4.5 at every located one.
Class PASS if W1 holds for all three exponents and W2; INCOMPLETE if for some exponent fewer than 4 test events are located; otherwise FAIL.
Controls (registration pass): c1 item functions (accept 2.8, 2.0, 4.0 and reject 1.9, 4.1 for W1; accept 1.5, 4.5 and reject 1.4, 4.6 for W2); c2 Da(115) = 0.21082; c3 the half-line N_half machinery of this script at k = 0 on the capacity stage c50_capc_D110.json reproduces the roots of the capacity N_half of q1250 at K_b 30 and 36 to 1e-12 (independent earlier form); c4 the chain solver c61 at KEXP = 0 reproduces the stored capacity transition K = 30 (c60_small_K30.json) to 1e-9 and at KEXP = 1e-6 differs from it by 1e-8 to 1e-5 (files c61_ctl30k0.json, c61_ctl30k1em6.json); c5 the free-edge reference table parsed from q1301b_out.txt contains the five test events for each of the four exponents; c6 stage health: every stage of c52_<tag>c.json converged, last D = 92; c7 interpolant and stage precision: at every test event |a_quintic - a_septic| and |a_med(D = 92) - a_med(D = 80)| are at most 5 percent of the expected residual 2.8 * R_free (the band of the items is a factor 2 wide, so a tenth of its relative half-width bounds what the instrument may move the ratio).
Usage: py q1304_hardwall_family_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import hashlib, io, contextlib, json, os, re, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import mpmath as mp

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1250_quintic_residual as QQ
    import q1271_linear_in_k_blind as Z
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
from scipy.optimize import least_squares

Bf = T.machine(T.coef_series(), 8, 8)[0]
NOF = QQ.C.Q.N_of
KS = (0.02, 0.05, 0.1, 0.25)
TAGS = {0.02: "hw002", 0.05: "hw005", 0.1: "hw01", 0.25: "hw025"}
TARGET = list(range(25, 38))
TEST = (29, 31, 33, 35, 37)
BAND = {0.02: (2.0, 4.0), 0.05: (2.0, 4.0), 0.1: (2.0, 4.0), 0.25: (1.5, 4.5)}


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


def locr(f):
    try:
        j = json.load(open(f + ".json"))
        hl = [x for x in j["hist"] if "res" in x and "s" in x and "A" in x]
        a = j.get("A_t")
        return a if (a is not None and hl and hl[-1]["res"] < 1e-18 and abs(a - hl[-1]["A"]) < 1e-9) else None
    except Exception:
        return None


def item(k, r):
    lo, hi = BAND[k]
    return lo <= r <= hi


def free_ref():
    ref = {}
    cur = None
    for l in open("q1301b_out.txt", encoding="utf-8"):
        m = re.match(r"k = ([\d.]+): \d+ events", l)
        if m:
            cur = float(m.group(1))
            ref[cur] = {}
            continue
        if l.startswith("capacity"):
            cur = None
        m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", l)
        if m and cur is not None:
            ref[cur][int(m.group(1))] = (
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
            )  # v2, R, sigma_eff
    return ref


def nhw(stage_json, Dsel, deg):
    st = [s for s in json.load(open(stage_json))["stages"] if s["D"] == Dsel and s["conv"]][-1]
    d = np.array(st["d"][: 1 + st["nfree"]])
    j = np.arange(1, len(d) + 1) - 0.5
    mm = (d >= 4) & (d <= 30)
    law = tuple(
        least_squares(lambda q: np.array([NOF(x, q[0], q[1]) for x in d[mm]]) - j[mm], [0.0558, 0.5482]).x
    )
    X = d[d <= Dsel - 20.0]
    return QQ.make(lambda a: NOF(a, *law), X, deg), law


def stage_ok(tag):
    st = json.load(open(f"c52_{tag}c.json"))["stages"]
    return all(s["conv"] for s in st) and st[-1]["D"] == 92.0


def regrun():
    c1 = (
        all(item(0.05, r) for r in (2.8, 2.0, 4.0))
        and not item(0.05, 1.9)
        and not item(0.05, 4.1)
        and item(0.25, 1.5)
        and item(0.25, 4.5)
        and not item(0.25, 1.4)
        and not item(0.25, 4.6)
    )
    c2 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082
    Nc, _ = nhw("c50_capc_D110.json", 110.0, 5)
    capN = QQ.make(lambda a: NOF(a, *QQ.C.CAPL), QQ.C.X, 5)
    c3 = all(abs(QQ.root(Nc, K) - QQ.root(capN, K)) < 1e-12 for K in (30, 36))
    a0 = json.load(open("c61_ctl30k0.json"))["A_t"]
    a1 = json.load(open("c61_ctl30k1em6.json"))["A_t"]
    ast = json.load(open("c60_small_K30.json"))["A_t"]
    c4 = abs(a0 - ast) < 1e-9 and 1e-8 < abs(a1 - ast) < 1e-5
    ref = free_ref()
    c5 = all(all(K in ref.get(k, {}) for K in TEST) for k in KS)
    c6 = all(stage_ok(TAGS[k]) for k in KS)
    out = {}
    w5 = 0.0
    w6 = 0.0
    for k in KS:
        tag = TAGS[k]
        sj = f"c52_{tag}c.json"
        N5, law = nhw(sj, 92.0, 5)
        N7, _ = nhw(sj, 92.0, 7)
        N80, _ = nhw(sj, 80.0, 5)
        stl = [s for s in json.load(open(sj))["stages"] if s["D"] == 92.0][-1]
        out[str(k)] = {"law": list(law), "wall_mass": stl["m"][0]}
        for K in TARGET:
            am = QQ.root(N5, K)
            v2 = Z.v2f(N5, am)
            rexp = 2.8 * ref[k][K][1] * 1e-9 if K in ref[k] else None
            if K in TEST:
                w5 = max(w5, abs(am - QQ.root(N7, K)) / rexp)
                w6 = max(w6, abs(am - QQ.root(N80, K)) / rexp)
            out[str(k)][str(K)] = dict(a_med=am, v2=float(v2), a0=float(am - 0.0045))
    c7 = w5 <= 0.05 and w6 <= 0.05
    print(
        f"control c1 {c1}; c2 {c2}; c3 (capacity N_half reproduced) {c3}; c4 (chain capacity recovery {a0 - ast:+.2e}, k = 1e-6 shift {a1 - ast:+.2e}) {c4}; c5 (free-edge reference complete) {c5}; c6 (stages converged to D = 92) {c6}; c7 (max |q5 - q7| / expected R {w5:.4f}, max |D92 - D80| / expected R {w6:.4f}, bar 0.05) {c7}"
    )
    if not (c1 and c2 and c3 and c4 and c5 and c6 and c7):
        print("REGISTRATION-RUN q1304: VOID (control)")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1304_predictions.json", "w"), indent=1)
    for k in KS:
        tag = TAGS[k]
        law = out[str(k)]["law"]
        L = [
            "#!/bin/bash",
            f"# run1304{tag}.sh (8 Oct 2026): hard-wall family chain c61 for k = {k}, K_b 25-37; start A0 = a_med - 0.0045, step 0.003 (q1304_predictions.json)",
            'cd "./centre_scripts"',
            f"export KEXP={k} HL_JSON=c52_{tag}c.json HL_STAGE=92 LAW0={law[0]:.5f},{law[1]:.5f} WM0={out[str(k)]['wall_mass']:.6f}",
        ]
        for K in TARGET:
            L.append(
                f"py c61_hw_fullchain.py {K} {out[str(k)][str(K)]['a0']:.6f} 0.003 160 u1304_{tag}_K{K} > c61_u1304_{tag}_K{K}.log 2>&1"
            )
        L.append(f"echo done $(date +%H:%M:%S) > run1304{tag}.done")
        open(f"run1304{tag}.sh", "w", newline="\n").write("\n".join(L) + "\n")
    print("sha256", sh)
    print("REGISTRATION-RUN q1304: controls True, table and run1304*.sh written")


def read():
    js = json.load(open("q1304_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1304: VOID (table hash)")
        sys.exit(2)
    ref = free_ref()
    res = {}
    incomplete = False
    for k in KS:
        tag = TAGS[k]
        sj = f"c52_{tag}c.json"
        N5, _ = nhw(sj, 92.0, 5)
        N7, _ = nhw(sj, 92.0, 7)
        rows = []
        for K in TARGET:
            A = locr(f"c61_u1304_{tag}_K{K}")
            if A is None:
                print(f"k {k} K_b {K}: NOT LOCATED")
                continue
            am5 = js[str(k)][str(K)]["a_med"]
            am7 = QQ.root(N7, K)
            v2 = Z.v2f(N5, A)
            R5 = (A - am5) * 1e9
            R7 = (A - am7) * 1e9
            s5 = R5 / Da(v2)
            s7 = R7 / Da(v2)
            fr = ref[k].get(K)
            rho5 = s5 / fr[2] if fr else None
            rho7 = s7 / fr[2] if fr else None
            print(
                f"k {k} K_b {K}: A_t {A:.10f}, R_hard {R5:+.5f}e-9, sigma_hard {s5:.4f}, sigma_free {fr[2] if fr else float('nan'):.4f}, ratio {rho5 if rho5 is not None else float('nan'):.3f} (septic {rho7 if rho7 is not None else float('nan'):.3f})"
            )
            if K in TEST and fr:
                rows.append((K, rho5, rho7))
        n = len(rows)
        if n < 4:
            incomplete = True
            print(f"k {k}: test events located {n} of 5")
            continue
        ok = all(item(k, a) and item(k, b) for _, a, b in rows)
        res[k] = (ok, min(a for _, a, _ in rows), max(a for _, a, _ in rows))
        print(
            f"k {k}: test events located {n}; ratio range {res[k][1]:.3f} to {res[k][2]:.3f}; item {'W2' if k == 0.25 else 'W1'} (band {BAND[k]}): {ok}"
        )
    if incomplete:
        print("VERDICT q1304 hard-wall family class: INCOMPLETE")
        return
    allok = all(v[0] for v in res.values())
    print(
        f"VERDICT q1304 hard-wall family class: {'PASS' if allok else 'FAIL'}; "
        + "; ".join(f"k {k}: {res[k][0]}" for k in KS)
        + "; ratio ranges "
        + ", ".join(f"{k}: {res[k][1]:.2f}-{res[k][2]:.2f}" for k in KS)
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
