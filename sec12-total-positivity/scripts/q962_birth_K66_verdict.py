"""q962 (7 Oct 2026): verdict script of T74 (amendment 1745): blind test of the seventh event of the tower, the birth K 66 -> 67 (K_b = 66), against the births law (q635 predictions of q951, sha256 registered in 1731) and against the step of the law residual.
Inputs: tower_K67_birth_out.json (key A_cross, written by q944 for the birth), tower_K67_q938.txt (last line of q938 on the same birth), tower_K66_As.json (A_s of the split K 65 -> 66, key A_s).
Items:
N1 |A_cross - 44.365124496108265| <= 2e-8 (the bar of T69, prediction of K_b 66 in the registered file).
N2 the step r66 - r65 of the residual (r = measured - predicted) lies between +1.3e-9 and +2.1e-9 (the five steps of 1744 are +2.058e-9, +1.958e-9, +1.774e-9, +1.669e-9, +1.818e-9).
N3 the last line of tower_K67_q938.txt starts with "VERDICT q938 T67: PASS;".
Verdict (last line): PASS if N1, N2 and N3 hold; FAIL if the birth is measured and an item fails; INCOMPLETE if tower_K67_birth_out.json or tower_K67_q938.txt does not exist; VOID if the prediction file is missing or has another sha256, a file is unreadable, or a control does not behave.
Controls: PC (r66 = r65 + 1.7e-9, q938 line PASS) must PASS; NC1 (r66 = r65 + 3.0e-9) must FAIL; NC2 (r66 = r65 + 0.5e-9) must FAIL; NC3 (q938 line FAIL) must FAIL; NC4 (A_cross = predicted + 5e-8) must FAIL; NC5 (birth file missing) must be INCOMPLETE.
Usage: py q962_birth_K66_verdict.py [--registration-run]   (the flag skips the birth files and prints r65 and the registered band for A_cross)
"""

import hashlib, json, os, sys

ARCH = "scratch"
PRED = f"{ARCH}/q635_capacity_K44_69_predictions.json"
SHA = "519f55a4aca0d7c64dba60561ef3544f19904e98f61e8b1e812234be9a5b422c"
BAR = 2e-8
LO, HI = 1.3e-9, 2.1e-9
BIRTH = f"{ARCH}/tower_K67_birth_out.json"
Q938 = f"{ARCH}/tower_K67_q938.txt"
ASF = f"{ARCH}/tower_K66_As.json"


def void(msg):
    print("VOID q962: " + msg, flush=True)
    print("VERDICT q962 T74: VOID (" + msg + ")", flush=True)
    sys.exit(2)


def verdict(a_cross, q938_line, pred66, r65):
    if a_cross is None or q938_line is None:
        return "INCOMPLETE", {}
    r66 = a_cross - pred66
    it = {
        "N1": abs(r66) <= BAR,
        "N2": LO <= r66 - r65 <= HI,
        "N3": q938_line.startswith("VERDICT q938 T67: PASS;"),
    }
    return ("PASS" if all(it.values()) else "FAIL"), it


def main():
    reg = "--registration-run" in sys.argv
    if not os.path.exists(PRED):
        void("prediction file missing")
    if hashlib.sha256(open(PRED, "rb").read()).hexdigest() != SHA:
        void("sha256 of the prediction file differs from the registered one")
    P = json.load(open(PRED))["capacity"]["pred"]
    pred = {int(K): v["a_pred"] for K, v in P.items()}
    try:
        r65 = float(json.load(open(ASF))["A_s"]) - pred[65]
    except Exception as e:
        void(f"unreadable {ASF}: {e}")
    p66 = pred[66]
    ok = "VERDICT q938 T67: PASS; failed items []"
    bad = "VERDICT q938 T67: FAIL; failed items ['B2']"
    print(
        f"r65 = {r65:+.4e}; predicted K_b 66 = {p66:.12f}; registered band of A_cross for N2: {p66 + r65 + LO:.12f} to {p66 + r65 + HI:.12f}"
    )
    c = {
        "PC": verdict(p66 + r65 + 1.7e-9, ok, p66, r65)[0] == "PASS",
        "NC1": verdict(p66 + r65 + 3.0e-9, ok, p66, r65)[0] == "FAIL",
        "NC2": verdict(p66 + r65 + 0.5e-9, ok, p66, r65)[0] == "FAIL",
        "NC3": verdict(p66 + r65 + 1.7e-9, bad, p66, r65)[0] == "FAIL",
        "NC4": verdict(p66 + 5e-8, ok, p66, r65)[0] == "FAIL",
        "NC5": verdict(None, ok, p66, r65)[0] == "INCOMPLETE",
    }
    print("controls:", c)
    if not all(c.values()):
        void("a control does not behave")
    a = line = None
    if not reg and os.path.exists(BIRTH) and os.path.exists(Q938):
        try:
            a = float(json.load(open(BIRTH))["A_cross"])
            ls = [x for x in open(Q938, encoding="utf-8").read().splitlines() if x.strip()]
            line = ls[-1]
        except Exception as e:
            void(f"unreadable birth files: {e}")
    v, it = verdict(a, line, p66, r65)
    if a is not None:
        print(
            f"K_b 66: predicted {p66:.12f}, measured {a:.12f}, r66 {a - p66:+.4e}, step r66 - r65 {a - p66 - r65:+.4e}, items {it}, q938 line: {line}"
        )
    print(f"VERDICT q962 T74: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
