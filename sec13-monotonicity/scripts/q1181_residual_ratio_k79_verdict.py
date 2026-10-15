"""q1181 (7 Oct 2026, registered in amendment 1917): verdict script of T143. After the failure of T116 at K_b 78 (amendment 1916: ratio r_K / r_(K-1) = 1.1917 above the band edge 1.19) the sequence of the ratios is 1.1555, 1.1663, 1.1729, 1.1856, 1.1917 for K_b = 74 to 78;
the increments are +0.0108, +0.0066, +0.0127, +0.0061 (mean +0.0090, largest deviation from the mean 0.0037). Model: the ratio keeps rising by a similar amount. Prediction for K_b 79: r_79 / r_78 lies in [1.185, 1.215] (the mean step gives 1.2007).
r_K = measured - predicted amplitude. Predictions: q635_capacity_K74_79_predictions.json (sha256 a0b13d053585eadc282e0e38a89ee1324f0be4a4015f8fb72c8e9b177c65310f), keys 78 and 79, field a_pred. Measured: A_cross of tower_K79_birth_out.json (K_b 78) and A_s of tower_K80_As.json (K_b 79).
Verdict (last line): PASS if both files exist and the ratio lies in [1.185, 1.215]; FAIL if the ratio is outside; INCOMPLETE if tower_K80_As.json is missing; VOID if the prediction file is missing or has another sha256, or a control misbehaves.
Controls (registration run, no K_b 79 value): PC a synthetic r_79 = 1.2007 r_78 passes; NC1 1.17 r_78 fails; NC2 1.23 r_78 fails; NC3 a missing file is INCOMPLETE.
Usage: py q1181_residual_ratio_k79_verdict.py [--registration-run]"""

import hashlib, json, os, sys

ARCH = "scratch"
P3 = f"{ARCH}/q635_capacity_K74_79_predictions.json"
SHA3 = "a0b13d053585eadc282e0e38a89ee1324f0be4a4015f8fb72c8e9b177c65310f"
LO, HI = 1.185, 1.215
F78 = f"{ARCH}/tower_K79_birth_out.json"
F79 = f"{ARCH}/tower_K80_As.json"


def judge(ratio):
    return "INCOMPLETE" if ratio is None else ("PASS" if LO <= ratio <= HI else "FAIL")


def void(msg):
    print("VERDICT q1181 T143: VOID (" + msg + ")", flush=True)
    sys.exit(2)


def main():
    reg = "--registration-run" in sys.argv
    c = {
        "PC": judge(1.2007) == "PASS",
        "NC1": judge(1.17) == "FAIL",
        "NC2": judge(1.23) == "FAIL",
        "NC3": judge(None) == "INCOMPLETE",
    }
    print("controls:", c)
    if not all(c.values()):
        void("a control does not behave")
    if reg:
        print("VERDICT q1181 T143: INCOMPLETE (controls only)")
        sys.exit(3)
    if not os.path.exists(P3) or hashlib.sha256(open(P3, "rb").read()).hexdigest() != SHA3:
        void("prediction file missing or changed")
    pred = json.load(open(P3))["capacity"]["pred"]
    p78 = pred["78"]["a_pred"]
    p79 = pred["79"]["a_pred"]
    if not os.path.exists(F78):
        void("tower_K79_birth_out.json missing")
    r78 = json.load(open(F78))["A_cross"] - p78
    print(f"r_78 = {r78:+.4e} (measured {json.load(open(F78))['A_cross']:.12f}, predicted {p78:.12f})")
    if not os.path.exists(F79):
        print("VERDICT q1181 T143: INCOMPLETE (tower_K80_As.json missing)")
        sys.exit(3)
    A79 = json.load(open(F79))["A_s"]
    r79 = A79 - p79
    ratio = r79 / r78
    print(
        f"r_79 = {r79:+.4e} (measured {A79:.12f}, predicted {p79:.12f}); ratio r_79 / r_78 = {ratio:.4f}; band [{LO}, {HI}]"
    )
    v = judge(ratio)
    print(f"VERDICT q1181 T143: {v} (ratio {ratio:.4f})")
    sys.exit({"PASS": 0, "FAIL": 1}[v])


if __name__ == "__main__":
    main()
