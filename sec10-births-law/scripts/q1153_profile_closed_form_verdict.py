"""q1153 (7 Oct 2026, registered in amendment 1895): verdict script of T133. The universal flow profile at the end of a family is f(xi) = coth(pi xi) - pi xi / sinh^2(pi xi), no parameter fitted.
Data: figsrc.event_profiles (xi_j, f_j), j = 1..10: S0 capacity K 66 to 75, S1 capacity K 63 to 65, S2 immune, S3 NPMLE. Items per set: rms of f_j - f_pred(xi_j) <= 0.0050 and largest |difference| <= 0.012.
Last line: VERDICT q1153 T133: PASS if the four sets satisfy both items, FAIL if a measured item fails, INCOMPLETE if a set has no points, VOID if a control misbehaves.
Usage: py q1153_profile_closed_form_verdict.py [--registration-run]"""

import sys, math
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "paper_births/figures")
RMS_BAR = 0.0050
MAX_BAR = 0.012


def f_pred(xi):
    z = math.pi * np.asarray(xi, float)
    return 1 / np.tanh(z) - z / np.sinh(z) ** 2


def judge(sets):
    res = {}
    for k, (xi, f) in sets.items():
        if xi.size == 0:
            res[k] = "INCOMPLETE"
            continue
        d = f - f_pred(xi)
        res[k] = "PASS" if (np.sqrt(np.mean(d**2)) <= RMS_BAR and np.max(np.abs(d)) <= MAX_BAR) else "FAIL"
    s = list(res.values())
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "INCOMPLETE" in s else "PASS")), res


def controls():
    import figsrc

    prof, _ = figsrc.event_profiles(10)
    rng = np.random.default_rng(3)
    c = {}
    xi = np.array([x for _, _, xs, _ in prof for x in xs])
    base = {k: xi for k in ("S0", "S1", "S2", "S3")}
    c["PC"] = judge({k: (v, f_pred(v) + rng.normal(0, 0.001, v.size)) for k, v in base.items()})[0] == "PASS"

    def p2(x):
        return 1 - (1 + 4.0734 * x + 8.4643 * x**2 - 3.2957 * x**3) * np.exp(-2 * math.pi * x)

    c["NC1"] = judge({k: (v, p2(v) + 0.013) for k, v in base.items()})[0] == "FAIL"
    c["NC2"] = judge({k: (v, np.tanh(6 * v)) for k, v in base.items()})[0] == "FAIL"
    import mpmath as mp

    mp.mp.dps = 40
    z = mp.pi * mp.mpf("0.02")
    exact = float(mp.coth(z) - z / mp.sinh(z) ** 2)
    c["PC2"] = abs(float(f_pred(0.02)) - exact) < 1e-9
    return c


def load():
    import figsrc

    prof, _ = figsrc.event_profiles(10)
    sets = {"S0": [], "S1": [], "S2": [], "S3": []}
    for pb, K, xi, f in prof:
        if pb == "capacity":
            sets["S0" if K >= 66 else "S1"].append((xi, f))
        elif pb == "immune":
            sets["S2"].append((xi, f))
        elif pb == "NPMLE":
            sets["S3"].append((xi, f))
    return {
        k: (np.array([x for xi, f in v for x in xi]), np.array([y for xi, f in v for y in f]))
        for k, v in sets.items()
    }, {k: len(v) for k, v in sets.items()}


def main():
    c = controls()
    print("controls:", {k: bool(v) for k, v in c.items()})
    if not all(c.values()):
        print("VERDICT q1153 T133: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1153 T133: INCOMPLETE (controls run only)")
        sys.exit(3)
    data, counts = load()
    print("families per set:", counts)
    v, res = judge(data)
    for k, (xi, f) in data.items():
        if xi.size:
            d = f - f_pred(xi)
            print(
                f"   {k}: points {xi.size}, rms {np.sqrt(np.mean(d ** 2)):.5f} (bar {RMS_BAR}), largest |diff| {np.max(np.abs(d)):.5f} (bar {MAX_BAR}), mean diff {np.mean(d):+.5f} -> {res[k]}"
            )
    print(f"VERDICT q1153 T133: {v} (" + "; ".join(f"{k} {x}" for k, x in res.items()) + ")")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
