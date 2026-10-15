"""q1070 (7 Oct 2026): verdict script of T104 (registered in the amendment that follows 1821): the next two events of the tower, K_b 72 (birth K 72 -> 73, A_cross of tower_K73_birth_out.json) and K_b 73 (split K 73 -> 74, A_s of tower_K74_As.json), against the births law.
r_K = measured - predicted. r71 = A_s of tower_K72_As.json (1820) minus the prediction of K_b 71 of the file locked in 1799 (sha256 c7a1e162...). The predictions of K_b 72 and 73 are in the file q635_capacity_K72_74_predictions.json (sha256 b0a3c934...); its K_b 72 value must equal that of the locked file within 1e-12.
Items: M1: r72 / r71 lies in [1.08, 1.19]; M2: r73 / r72 lies in [1.08, 1.19]; M3: |r72| and |r73| <= 4e-8. (The ratios r_K / r_(K-1) at K = 67 to 71 were 1.1504, 1.1348, 1.1343, 1.1258 and 1.1349.)
Verdict (last line): PASS if both events are measured and M1 to M3 hold; FAIL if a measured item fails; INCOMPLETE if no measured item fails and an event is missing; VOID if a prediction file is missing or has another sha256, the K_b 72 values differ, a file is unreadable or a control misbehaves.
Controls: PC (K_b 72 at predicted + 1.13 r71, K_b 73 at predicted + 1.13^2 r71) must PASS; NC1 (r72 = r71) must FAIL; NC2 (K_b 73 at predicted + 5e-8) must FAIL; NC3 (K_b 73 missing) must be INCOMPLETE.
Usage: py q1070_law_tower_extension3_verdict.py [--registration-run]"""

import hashlib, json, os, sys

ARCH = "scratch"
P1 = f"{ARCH}/q635_capacity_K67_72_predictions.json"
P2 = f"{ARCH}/q635_capacity_K72_74_predictions.json"
SHA1 = "c7a1e1620e8a3bb4e715d3472aedb5d6508cb429fabb13802bd40e4849e03320"
SHA2 = "b0a3c9346ea2c463af9f8e4458cf4e50b505c140129438fc9fa853718ef617da"
BAR = 4e-8
LO, HI = 1.08, 1.19
FILES = {
    71: (f"{ARCH}/tower_K72_As.json", "A_s"),
    72: (f"{ARCH}/tower_K73_birth_out.json", "A_cross"),
    73: (f"{ARCH}/tower_K74_As.json", "A_s"),
}


def void(msg):
    print("VERDICT q1070 T104: VOID (" + msg + ")", flush=True)
    sys.exit(2)


def verdict(meas, pred):
    res = {}
    prev = meas[71] - pred[71]
    for K in (72, 73):
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
    for p, s in ((P1, SHA1), (P2, SHA2)):
        if not os.path.exists(p):
            void(f"prediction file {p} missing")
        if sha(p) != s:
            void(f"sha256 of {p} differs from the registered one")
    pr1 = {int(K): v["a_pred"] for K, v in json.load(open(P1))["capacity"]["pred"].items()}
    pr2 = {int(K): v["a_pred"] for K, v in json.load(open(P2))["capacity"]["pred"].items()}
    if abs(pr1[72] - pr2[72]) > 1e-12:
        void("K_b 72 of the two prediction files differ")
    pred = {71: pr1[71], 72: pr2[72], 73: pr2[73]}
    try:
        a71 = float(json.load(open(FILES[71][0]))[FILES[71][1]])
    except Exception as e:
        void(f"unreadable {FILES[71][0]}: {e}")
    r71 = a71 - pred[71]
    synth = lambda extra={}: {71: a71, 72: pred[72] + 1.13 * r71, 73: pred[73] + 1.13**2 * r71, **extra}
    c = {
        "PC": verdict(synth(), pred)[0] == "PASS",
        "NC1": verdict(synth({72: pred[72] + r71}), pred)[0] == "FAIL",
        "NC2": verdict(synth({73: pred[73] + 5e-8}), pred)[0] == "FAIL",
        "NC3": verdict(synth({73: None}), pred)[0] == "INCOMPLETE",
    }
    print(f"r71 = {r71:+.4e}; predicted K_b 72, 73 = {pred[72]:.12f}, {pred[73]:.12f}")
    print("controls:", c)
    if not all(c.values()):
        void("a control does not behave")
    meas = {71: a71}
    if not reg:
        for K in (72, 73):
            f, key = FILES[K]
            if os.path.exists(f):
                try:
                    meas[K] = float(json.load(open(f))[key])
                except Exception as e:
                    void(f"unreadable {f}: {e}")
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
    print(f"VERDICT q1070 T104: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
