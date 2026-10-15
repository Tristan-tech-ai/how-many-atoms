"""q1055 (7 Oct 2026): verdict script of T100 (registered in the amendment that follows 1805): the scaling J (t* - 1) of the margin of the Cramer minor of the innermost flow (J the number of positive atoms; t* from q988) at the first, the median and the last state of the families K 68 and K 69.
Bands (fixed from the series of q1054 before any K 68 or K 69 state was read for t*): even K (K 68, J = 33): first [4.035, 4.095], median [4.20, 4.30], last [4.72, 4.82]; odd K (K 69, J = 33): first [3.975, 4.035], median [4.18, 4.25], last [4.66, 4.74]. N1(1) must be positive.
States: sorted by A, index 0, n // 2 and n - 1 of the OPTIMAL states of the family (K 68: A = 44.8898, 45.1325, 45.4025; K 69: A = 45.4075, 45.6625, 45.9175).
Controls on the K 65 state A = 43.5803 (J 31; J (t* - 1) = 4.228 in q991): PC (odd median band) passes; NC1 (odd first band) fails; NC2 (odd last band) fails.
Usage: py q1055_margin_scaling_blind_verdict.py controls | py q1055_margin_scaling_blind_verdict.py
Last line: VERDICT q1055 T100: PASS / FAIL / VOID / INCOMPLETE."""

import sys, re, subprocess

BANDS = {
    "even": {"first": (4.035, 4.095), "median": (4.20, 4.30), "last": (4.72, 4.82)},
    "odd": {"first": (3.975, 4.035), "median": (4.18, 4.25), "last": (4.66, 4.74)},
}
CONTROL = "tower/K65/results_q220_gauss_K65_A43.5803.json"
STATES = {
    68: {
        "first": "tower/K68/results_q220_gauss_K68_A44.8898.json",
        "median": "tower/K68/results_q220_gauss_K68_A45.1325.json",
        "last": "tower/K68/results_q220_gauss_K68_A45.4025.json",
    },
    69: {
        "first": "tower/K69/results_q220_gauss_K69_A45.4075.json",
        "median": "tower/K69/results_q220_gauss_K69_A45.6625.json",
        "last": "tower/K69/results_q220_gauss_K69_A45.9175.json",
    },
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
        print("VERDICT q1055 T100: VOID (control state not measured)")
        sys.exit(2)
    K, m, pos, t = c
    val = m * (t - 1)
    inb = lambda v, par, w: BANDS[par][w][0] <= v <= BANDS[par][w][1]
    pc = pos and inb(val, "odd", "median")
    nc1 = inb(val, "odd", "first")
    nc2 = inb(val, "odd", "last")
    print(
        f"control K {K}: m {m}, N1(1) positive {pos}, t* {t}, J (t* - 1) {val:.4f}; PC (odd median band) {pc}; NC1 (odd first band) inside {nc1}; NC2 (odd last band) inside {nc2}",
        flush=True,
    )
    if not (pc and not nc1 and not nc2):
        print("VERDICT q1055 T100: VOID (a control misbehaves)")
        sys.exit(2)
    if len(sys.argv) > 1 and sys.argv[1] == "controls":
        print("VERDICT q1055 T100: INCOMPLETE (controls run only)")
        return
    bad = 0
    for Kf, d in STATES.items():
        par = "even" if Kf % 2 == 0 else "odd"
        for which, path in d.items():
            r = measure(path)
            if r is None:
                print(f"VERDICT q1055 T100: VOID (state {path} not measured)")
                sys.exit(2)
            K, m, pos, t = r
            if m != Jof(K):
                print(f"VERDICT q1055 T100: VOID (m {m} differs from J(K) {Jof(K)} for {path})")
                sys.exit(2)
            val = m * (t - 1)
            ok = pos and inb(val, par, which)
            bad += not ok
            print(
                f"K {K} {which} state {path}: m {m}, N1(1) positive {pos}, t* {t}, J (t* - 1) {val:.4f}, band {BANDS[par][which]}, {'inside' if ok else 'OUTSIDE'}",
                flush=True,
            )
    print("VERDICT q1055 T100: " + ("PASS" if bad == 0 else "FAIL") + f"; states 6; outside {bad}")


if __name__ == "__main__":
    main()
