"""q1263 (8 Oct 2026, registered in amendment 2041): blind test of the constant ratio of the immune to the NPMLE median-sum residual at K_b 14 to 23.
Reading that motivates it (q1262, exploratory): with the quintic spline of N_half the residual dev = A_stored - a_med of the immune problem is 1.64 to 1.83 times that of the NPMLE problem at equal K_b over K_b 24 to 48, while dev falls by a factor 1500.
Prediction (inputs: the NPMLE stored amplitudes of K_b 14 to 23 and the immune half-line; no immune amplitude of K_b 14 to 23 is read or computed): dev_imm(K_b) = R dev_NPMLE(K_b) with R in [1.50, 1.85]; central value 1.70 used only to place the locator start a0 = a_med + 1.70 dev_NPMLE - 0.004.
dev is in 1e-9; a_med is the root of the median-sum law (table q1220, theta 0.3) with the quintic spline of the half-line counting function, immune stage D = 92 (atoms to depth 72) as q1250.
--registration-run : controls, the table q1263_predictions.json (sha256), the starts a0; writes nothing from immune amplitudes of K_b 14 to 23.
--read             : items on the located events (files c19_u1263_K<K>.json, key a_t).
Located: a_t not null, the last residual of the history below 1e-20 and |a_t - last a| < 1e-9.
Items: R3 at least 8 of the 10 events K_b 14 to 23 located; R1 every located event has 1.50 <= dev_imm / dev_NPMLE <= 1.85; R2 dev_imm > 0 at every located event.
Controls (first): c1 the immune pipeline reproduces dev(K_b 30) = +27.1162e-9 of q1250 within 0.001e-9 from the file c19_u1231_K30.json; c2 the NPMLE pipeline reproduces dev(K_b 20) = +641.5634e-9 within 0.001e-9; c3 the item R1 accepts R = 1.70 and rejects 1.2 and 2.2; c4 on the stored events K_b 24 to 43 the ratio imm/NPMLE lies in the bar at every event where both exist (the bar can pass real data) and the ratio imm/capacity lies outside the bar at every event K_b 24 to 40 (the bar rejects a wrong comparator).
Usage: py q1263_immune_npmle_ratio_blind.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import hashlib, json, os, re, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import q1250_quintic_residual as Q
import q1252_order16_quintic as R

C, N, m = Q.C, Q.N, R.m
TARGET = list(range(14, 24))
LO, HI = 1.50, 1.85
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))


def setup():
    sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
    pi_ = np.array(sn["p"][: sn["nfree"]])
    pi_ = pi_[pi_ <= 72]
    Ni = Q.make(lambda a: Q.N4(a, (0.22573, 0.06806)), pi_, 5)
    Nn = Q.make(lambda a: Q.N4(a, N.LAW), N.pn, 5)
    return Ni, Nn


def item_r1(r):
    return LO <= r <= HI


def regrun():
    Ni, Nn = setup()
    d30 = (json.load(open("c19_u1231_K30.json"))["a_t"] - Q.root(Ni, 30)) * 1e9
    c1 = abs(d30 - 27.1162) < 0.001
    d20 = (gv(m.NPM_FILES[20]) - Q.root(Nn, 20)) * 1e9
    c2 = abs(d20 - 641.5634) < 0.001
    c3 = item_r1(1.70) and not item_r1(1.2) and not item_r1(2.2)
    D = json.load(open("q1250_dev.json"))
    ks = [
        K
        for K in range(24, 44)
        if str(K) in D["immune_5"] and str(K) in D["npmle_5"] and D["npmle_5"][str(K)] > 0
    ]
    c4a = len(ks) >= 12 and all(item_r1(D["immune_5"][str(K)] / D["npmle_5"][str(K)]) for K in ks)
    kc = [
        K
        for K in range(24, 41)
        if str(K) in D["immune_5"] and str(K) in D["capacity_5"] and D["capacity_5"][str(K)] > 0
    ]
    c4b = len(kc) >= 12 and all(not item_r1(D["immune_5"][str(K)] / D["capacity_5"][str(K)]) for K in kc)
    print(
        f"control c1 (immune dev K_b 30 {d30:+.4f}): {c1}; c2 (NPMLE dev K_b 20 {d20:+.4f}): {c2}; c3 (item function): {c3}; c4a (imm/NPMLE inside the bar at {len(ks)} stored events): {c4a}; c4b (imm/capacity outside at {len(kc)} events): {c4b}"
    )
    if not (c1 and c2 and c3 and c4a and c4b):
        print("REGISTRATION-RUN q1263: VOID (control)")
        sys.exit(2)
    out = {}
    for K in TARGET:
        A = gv(m.NPM_FILES[K])
        am_n = Q.root(Nn, K)
        dn = (A - am_n) * 1e9
        if dn <= 1.0:
            print(f"K_b {K}: NPMLE dev {dn:+.4f}e-9 not above 1e-9; the event cannot test the ratio")
            sys.exit(2)
        am = Q.root(Ni, K)
        ap = am + 1.70 * dn * 1e-9
        out[str(K)] = dict(
            a_med=am, dev_npmle=float(dn), dev_lo=LO * float(dn), dev_hi=HI * float(dn), a0=float(ap - 0.004)
        )
        print(
            f"K_b {K}: NPMLE dev {dn:+.4f}e-9, immune a_med {am:.10f}, predicted immune dev in [{LO * dn:.3f}, {HI * dn:.3f}]e-9, start a0 = {ap - 0.004:.6f}"
        )
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("q1263_predictions.json", "w"), indent=1)
    print("sha256", sh)
    print("REGISTRATION-RUN q1263: controls True, table written")


def read():
    js = json.load(open("q1263_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1263: VOID (table hash)")
        sys.exit(2)
    res = {}
    for K in TARGET:
        try:
            j = json.load(open(f"c19_u1263_K{K}.json"))
            a = j["a_t"]
            hl = [x for x in j["hist"] if "res" in x and "a" in x]
            ok = a is not None and hl and hl[-1]["res"] < 1e-20 and abs(a - hl[-1]["a"]) < 1e-9
        except Exception:
            ok = False
            a = None
        res[K] = (a - js[str(K)]["a_med"]) * 1e9 if ok else None
    loc = {K: d for K, d in res.items() if d is not None}
    for K in TARGET:
        p = js[str(K)]
        print(
            f"K_b {K}: "
            + (
                "NOT LOCATED"
                if res[K] is None
                else f"immune dev {res[K]:+.4f}e-9, NPMLE dev {p['dev_npmle']:+.4f}e-9, ratio {res[K] / p['dev_npmle']:.4f}"
            )
        )
    r3 = len(loc) >= 8
    r1 = r3 and all(item_r1(d / js[str(K)]["dev_npmle"]) for K, d in loc.items())
    r2 = r3 and all(d > 0 for d in loc.values())
    print(
        f"located {len(loc)} of {len(TARGET)}; item R3 (>= 8 located): {r3}; item R1 (all located ratios in [{LO}, {HI}]): {r1}; item R2 (all located dev > 0): {r2}"
    )
    print(
        f"VERDICT q1263 ratio class: {'PASS' if (r3 and r1 and r2) else ('INCOMPLETE' if not r3 else 'FAIL')}; located {len(loc)} of {len(TARGET)}; R1 {r1}; R2 {r2}"
    )


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
