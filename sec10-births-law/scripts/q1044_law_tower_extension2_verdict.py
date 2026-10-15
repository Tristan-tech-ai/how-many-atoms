"""q1044 (7 Oct 2026): verdict script of T98 (registered in the amendment that follows 1798): the next two events of the tower, K_b 70 (birth K 70 -> 71, A_cross of tower_K71_birth_out.json) and K_b 71 (split K 71 -> 72, A_s of tower_K72_As.json), against the births law (predictions of q1043 = q951 with K_b 67 to 72, file q635_capacity_K67_72_predictions.json, sha256 registered in the amendment; K_b 67 to 69 reproduce the locked file of 1731 to 1e-13).
r_K = measured - predicted; r69 = A_s of tower_K70_As_T83.json (the copy of the file of T83) minus the prediction of K_b 69.
Items: M1: the step r70 - r69 lies in [1.4e-9, 2.6e-9]; M2: the step r71 - r70 lies in [1.4e-9, 2.6e-9]; M3: |r70| and |r71| <= 3e-8 (the bar of T69 was 2e-8; the residual is expected to pass 2e-8).
Verdict (last line): PASS if both events are measured and M1 to M3 hold; FAIL if a measured item fails; INCOMPLETE if no measured item fails and an event is missing; VOID if the prediction file is missing or has another sha256, a file is unreadable or a control misbehaves.
Controls: PC (K_b 70 at predicted + r69 + 1.9e-9, K_b 71 at + 3.8e-9) must PASS; NC1 (step 0.5e-9 at K_b 70) must FAIL; NC2 (K_b 71 at predicted + 5e-8) must FAIL; NC3 (K_b 71 missing) must be INCOMPLETE.
Usage: py q1044_law_tower_extension2_verdict.py [--registration-run]"""

import hashlib, json, os, sys

ARCH = "scratch"
PRED = f"{ARCH}/q635_capacity_K67_72_predictions.json"
SHA = "c7a1e1620e8a3bb4e715d3472aedb5d6508cb429fabb13802bd40e4849e03320"
BAR = 3e-8
LO, HI = 1.4e-9, 2.6e-9
FILES = {
    69: (f"{ARCH}/tower_K70_As_T83.json", "A_s"),
    70: (f"{ARCH}/tower_K71_birth_out.json", "A_cross"),
    71: (f"{ARCH}/tower_K72_As.json", "A_s"),
}


def void(msg):
    print("VERDICT q1044 T98: VOID (" + msg + ")", flush=True)
    sys.exit(2)


def verdict(meas, pred):
    res = {}
    prev = meas[69] - pred[69]
    for K in (70, 71):
        if meas.get(K) is None:
            res[K] = ("MISSING", None, None)
            prev = None
            continue
        r = meas[K] - pred[K]
        if prev is None:
            res[K] = ("PASS" if abs(r) <= BAR else "FAIL", r, None)
            prev = r
            continue
        st = r - prev
        ok = abs(r) <= BAR and LO <= st <= HI
        res[K] = ("PASS" if ok else "FAIL", r, st)
        prev = r
    s = [x[0] for x in res.values()]
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def main():
    reg = "--registration-run" in sys.argv
    if not os.path.exists(PRED):
        void("prediction file missing")
    if hashlib.sha256(open(PRED, "rb").read()).hexdigest() != SHA:
        void("sha256 of the prediction file differs from the registered one")
    P = json.load(open(PRED))["capacity"]["pred"]
    pred = {int(K): v["a_pred"] for K, v in P.items()}
    try:
        a69 = float(json.load(open(FILES[69][0]))[FILES[69][1]])
    except Exception as e:
        void(f"unreadable {FILES[69][0]}: {e}")
    r69 = a69 - pred[69]
    synth = lambda extra={}: {69: a69, 70: pred[70] + r69 + 1.9e-9, 71: pred[71] + r69 + 3.8e-9, **extra}
    c = {
        "PC": verdict(synth(), pred)[0] == "PASS",
        "NC1": verdict(synth({70: pred[70] + r69 + 0.5e-9}), pred)[0] == "FAIL",
        "NC2": verdict(synth({71: pred[71] + 5e-8}), pred)[0] == "FAIL",
        "NC3": verdict(synth({71: None}), pred)[0] == "INCOMPLETE",
    }
    print(f"r69 = {r69:+.4e}; predicted K_b 70, 71 = {pred[70]:.12f}, {pred[71]:.12f}")
    print("controls:", c)
    if not all(c.values()):
        void("a control does not behave")
    meas = {69: a69}
    if not reg:
        for K in (70, 71):
            f, key = FILES[K]
            if os.path.exists(f):
                try:
                    meas[K] = float(json.load(open(f))[key])
                except Exception as e:
                    void(f"unreadable {f}: {e}")
    v, res = verdict(meas, pred)
    for K, (st, r, step) in res.items():
        if r is None:
            print(f"   K_b {K}: not measured yet -> MISSING")
        else:
            print(
                f"   K_b {K}: predicted {pred[K]:.12f}, measured {meas[K]:.12f}, r {r:+.4e}"
                + (f", step {step:+.4e}" if step is not None else "")
                + f" -> {st}"
            )
    print(f"VERDICT q1044 T98: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
