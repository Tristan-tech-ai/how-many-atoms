"""q1071 (7 Oct 2026): verdict script of T105 (registered in the amendment that follows 1822; a copy of q1066 for the odd family K 71): the scaling J (t* - 1) of the margin of the Cramer minor of the innermost flow (J the number of positive atoms; t* from q988) at the first, the median (index n // 2 = 52) and the last of the 104 OPTIMAL states of tower/K71: A = 46.4359, 46.6889, 46.9439 (J = 34, odd K).
Bands for J (t* - 1), from the odd series of q1054 and T100 (first state 4.025, 4.020, 4.014, 4.010, 4.0062 for K 61, 63, 65, 67, 69; median 4.254, 4.236, 4.228, 4.221, 4.2141; last 4.768, 4.743, 4.728, 4.714, 4.6926): first [3.98, 4.03], median [4.18, 4.235], last [4.635, 4.715]. N1(1) must be positive.
Controls on the K 65 state A = 43.5803 (J 31; J (t* - 1) = 4.2284 in q991): PC (median band) passes; NC1 (first band) fails; NC2 (last band) fails.
Usage: py q1071_margin_scaling_k71_verdict.py controls | py q1071_margin_scaling_k71_verdict.py
Last line: VERDICT q1071 T105: PASS / FAIL / VOID / INCOMPLETE."""

import sys, re, subprocess

BANDS = {
    "even": {"first": (4.03, 4.085), "median": (4.20, 4.29), "last": (4.70, 4.80)},
    "odd": {"first": (3.98, 4.03), "median": (4.18, 4.235), "last": (4.635, 4.715)},
}
CONTROL = "tower/K65/results_q220_gauss_K65_A43.5803.json"
STATES = {
    71: {
        "first": "tower/K71/results_q220_gauss_K71_A46.4359.json",
        "median": "tower/K71/results_q220_gauss_K71_A46.6889.json",
        "last": "tower/K71/results_q220_gauss_K71_A46.9439.json",
    }
}
Jof = lambda K: (K - 2) // 2 if K % 2 == 0 else (K - 3) // 2


def measure(path):
    r = subprocess.run(
        [sys.executable, "q988_margin_along_family_reading.py", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    for line in r.stdout.splitlines():
        m = re.match(
            r"results_q220_gauss_K(\d+)_A([0-9.]+)\.json: A [0-9.]+, m (\d+); N1\(1\) positive (True|False); .*crossing ([0-9.]+)",
            line,
        )
        if m:
            return int(m.group(1)), int(m.group(3)), m.group(4) == "True", float(m.group(5))
    return None


def main():
    c = measure(CONTROL)
    if c is None:
        print("VERDICT q1071 T105: VOID (control state not measured)")
        sys.exit(2)
    K, m, pos, t = c
    val = m * (t - 1)
    inb = lambda v, par, w: BANDS[par][w][0] <= v <= BANDS[par][w][1]
    pc = pos and inb(val, "odd", "median")
    nc1 = inb(val, "odd", "first")
    nc2 = inb(val, "odd", "last")
    print(
        f"control K {K}: m {m}, N1(1) positive {pos}, t* {t}, J (t* - 1) {val:.4f}; PC (median band) {pc}; NC1 (first band) inside {nc1}; NC2 (last band) inside {nc2}",
        flush=True,
    )
    if not (pc and not nc1 and not nc2):
        print("VERDICT q1071 T105: VOID (a control misbehaves)")
        sys.exit(2)
    if len(sys.argv) > 1 and sys.argv[1] == "controls":
        print("VERDICT q1071 T105: INCOMPLETE (controls run only)")
        return
    bad = 0
    for Kf, d in STATES.items():
        par = "even" if Kf % 2 == 0 else "odd"
        for which, path in d.items():
            r = measure(path)
            if r is None:
                print(f"VERDICT q1071 T105: VOID (state {path} not measured)")
                sys.exit(2)
            K, m, pos, t = r
            if m != Jof(K):
                print(f"VERDICT q1071 T105: VOID (m {m} differs from J(K) {Jof(K)} for {path})")
                sys.exit(2)
            val = m * (t - 1)
            ok = pos and inb(val, par, which)
            bad += not ok
            print(
                f"K {K} {which} state {path}: m {m}, N1(1) positive {pos}, t* {t}, J (t* - 1) {val:.4f}, band {BANDS[par][which]}, {'inside' if ok else 'OUTSIDE'}",
                flush=True,
            )
    print("VERDICT q1071 T105: " + ("PASS" if bad == 0 else "FAIL") + f"; states 3; outside {bad}")


if __name__ == "__main__":
    main()
