"""q1084 (7 Oct 2026): verdict script of T116 (registered in the amendment that follows 1835): the next six events of the tower, K_b 74 (birth K 74 -> 75, A_cross of tower_K75_birth_out.json), K_b 75 (split K 75 -> 76, A_s of tower_K76_As.json), K_b 76 (birth, tower_K77_birth_out.json), K_b 77 (split, tower_K78_As.json), K_b 78 (birth, tower_K79_birth_out.json) and K_b 79 (split, tower_K80_As.json), against the births law.
r_K = measured - predicted. r73 = A_s of tower_K74_As.json (the split K 73 -> 74 of T104) minus the prediction of K_b 73 of the file q635_capacity_K72_74_predictions.json (sha256 b0a3c934...). The predictions of K_b 74 to 79 are in q635_capacity_K74_79_predictions.json (sha256 a0b13d05...); its K_b 74 value must equal that of the first file within 1e-12.
Items: M1 to M6: the ratio r_K / r_(K-1) lies in [1.08, 1.19] for K = 74 to 79; M7: |r_K| <= 2e-7 for K = 74 to 79.
Verdict (last line): PASS if all six events are measured and M1 to M7 hold; FAIL if a measured item fails; INCOMPLETE if no measured item fails and an event is missing; VOID if a prediction file is missing or has another sha256, the K_b 74 values differ, a file is unreadable or a control misbehaves.
Controls: PC (each event at predicted + 1.13 times the previous residual) must PASS; NC1 (r74 = r73) must FAIL; NC2 (K_b 76 at predicted + 5e-7) must FAIL; NC3 (K_b 79 missing) must be INCOMPLETE.
Usage: py q1084_law_tower_extension4_verdict.py [--registration-run] (controls only with the flag; it prints no measured value)
"""

import hashlib, json, os, sys

ARCH = "scratch"
P2 = f"{ARCH}/q635_capacity_K72_74_predictions.json"
P3 = f"{ARCH}/q635_capacity_K74_79_predictions.json"
SHA2 = "b0a3c9346ea2c463af9f8e4458cf4e50b505c140129438fc9fa853718ef617da"
SHA3 = "a0b13d053585eadc282e0e38a89ee1324f0be4a4015f8fb72c8e9b177c65310f"
BAR = 2e-7
LO, HI = 1.08, 1.19
KS = (74, 75, 76, 77, 78, 79)
FILES = {
    73: (f"{ARCH}/tower_K74_As.json", "A_s"),
    74: (f"{ARCH}/tower_K75_birth_out.json", "A_cross"),
    75: (f"{ARCH}/tower_K76_As.json", "A_s"),
    76: (f"{ARCH}/tower_K77_birth_out.json", "A_cross"),
    77: (f"{ARCH}/tower_K78_As.json", "A_s"),
    78: (f"{ARCH}/tower_K79_birth_out.json", "A_cross"),
    79: (f"{ARCH}/tower_K80_As.json", "A_s"),
}


def void(msg):
    print("VERDICT q1084 T116: VOID (" + msg + ")", flush=True)
    sys.exit(2)


def verdict(meas, pred):
    res = {}
    prev = meas[73] - pred[73]
    for K in KS:
        if meas.get(K) is None:
            res[K] = ("MISSING", None, None)
            prev = None
            continue
        r = meas[K] - pred[K]
        if prev is None:
            res[K] = ("PASS" if abs(r) <= BAR else "FAIL", r, None)
            prev = r
            continue
        q = r / prev
        ok = abs(r) <= BAR and LO <= q <= HI
        res[K] = ("PASS" if ok else "FAIL", r, q)
        prev = r
    s = [x[0] for x in res.values()]
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    reg = "--registration-run" in sys.argv
    for p, s in ((P2, SHA2), (P3, SHA3)):
        if not os.path.exists(p):
            void(f"prediction file {p} missing")
        if sha(p) != s:
            void(f"sha256 of {p} differs from the registered one")
    pr2 = {int(K): v["a_pred"] for K, v in json.load(open(P2))["capacity"]["pred"].items()}
    pr3 = {int(K): v["a_pred"] for K, v in json.load(open(P3))["capacity"]["pred"].items()}
    if abs(pr2[74] - pr3[74]) > 1e-12:
        void("K_b 74 of the two prediction files differ")
    pred = {73: pr2[73], **{K: pr3[K] for K in KS}}
    rs = 1.7e-8  # synthetic start of the controls only (r73 = +1.7e-8 is a stand-in, not a measurement)

    def synth(extra={}):
        d = {73: pred[73] + rs}
        prev = rs
        for K in KS:
            prev = prev * 1.13
            d[K] = pred[K] + prev
        d.update(extra)
        return d

    c = {
        "PC": verdict(synth(), pred)[0] == "PASS",
        "NC1": verdict(synth({74: pred[74] + rs}), pred)[0] == "FAIL",
        "NC2": verdict(synth({76: pred[76] + 5e-7}), pred)[0] == "FAIL",
        "NC3": verdict(synth({79: None}), pred)[0] == "INCOMPLETE",
    }
    print("controls (synthetic start r73 = +1.7e-8):", c)
    if not all(c.values()):
        void("a control does not behave")
    if reg:
        print("VERDICT q1084 T116: INCOMPLETE (controls run only)")
        sys.exit(3)
    meas = {}
    for K, (f, key) in FILES.items():
        if os.path.exists(f):
            try:
                meas[K] = float(json.load(open(f))[key])
            except Exception as e:
                void(f"unreadable {f}: {e}")
    if 73 not in meas:
        void("tower_K74_As.json missing (T104 not finished)")
    print(
        f"r73 = {meas[73] - pred[73]:+.4e}; predicted K_b 74 to 79 = "
        + ", ".join(f"{pred[K]:.9f}" for K in KS)
    )
    v, res = verdict(meas, pred)
    for K, (st, r, q) in res.items():
        if r is None:
            print(f"   K_b {K}: not measured yet -> MISSING")
        else:
            print(
                f"   K_b {K}: predicted {pred[K]:.12f}, measured {meas[K]:.12f}, r {r:+.4e}"
                + (f", ratio r/r_prev {q:.4f}" if q is not None else "")
                + f" -> {st}"
            )
    print(f"VERDICT q1084 T116: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
