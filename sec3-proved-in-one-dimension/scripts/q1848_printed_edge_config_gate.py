r"""q1848_printed_edge_config_gate.py (13 Oct 2026, task PR2 step 3): gate for the lower bound of the strip constant given by ONE explicit edge
configuration, the n = 3 configuration exactly as q1846 printed it (gap and atoms to 5 or 6 decimals).

Record of what came before. The gate q1847 ended FAIL by the letter: its criterion G2 compared the mpmath value that q1846 computed from its unrounded
optimiser output (3.97966462205876) with the value of the configuration as printed (rounded to 5 and 6 decimals), and the two differ by 4.8e-12, above
the bar 1e-12. That FAIL stands. This gate does not compare with q1846 at all: its object is a fixed, written configuration, which is a valid input for
every A >= g, so its edge functional B is a lower bound of liminf (e^(C+h_1) - 2A) by the argument in the docstring of q1846 (no optimisation enters).
Configuration (wall coordinate s <= 0, mass nu): g = 4.361045; (0, 3.49413), (-2.68983, 2.25076), (-4.36104, 0.62400).
Criteria, fixed before this script was run:
 H1 positive control: the same evaluation with no atom and g = 0 gives 2 kappa, kappa = -int Phi log Phi computed separately, within 1e-25;
 H2 feasibility: every offset lies in [-g, 0] and every mass is positive;
 H3 B is evaluated at 30 and at 50 digits with breakpoints at -g and at every atom; the two agree within 1e-20;
 H4 B <= sqrt(2 pi e) = 4.13273135412 (the proved upper bound of reading 291; a larger value is an error);
 H5 a negative control: the same configuration with the bulk end moved to -g + 1 while the atoms stay (an input that is still valid) gives a B that
    differs from B by more than 1e-6, so the evaluation is sensitive to the configuration.
Verdict (last line): VERDICT q1848: PASS if H1 to H5 hold, with the bound printed rounded down at the 10th decimal; FAIL if H2, H3 or H4 fails;
VOID if H1 or H5 fails or the evaluation crashes; INCOMPLETE is not used (the script has no external input).
Usage: py q1848_printed_edge_config_gate.py [--registration-run] [--selftest]   (from temporal_within/)"""

import sys

import mpmath as mp

G = "4.361045"
ATOMS = [("0", "3.49413"), ("-2.68983", "2.25076"), ("-4.36104", "0.62400")]
UPPER = "4.13273135412"


def B_eval(s, nu, g, dps):
    with mp.workdps(dps):
        sm = [mp.mpf(x) for x in s]
        nm = [mp.mpf(x) for x in nu]
        gm = mp.mpf(g)
        Gf = lambda t: mp.ncdf(-gm - t) + sum(n_ * mp.npdf(t - x) for x, n_ in zip(sm, nm))
        f = lambda t: (lambda v: v * mp.log(v) if v > 0 else mp.mpf(0))(Gf(t))
        pts = sorted(set([-gm] + sm))
        E = mp.quad(f, [-mp.inf, -30, -8] + pts + [8, 30, mp.inf])
        return 2 * (sum(nm) - gm - E)


def gate(g, atoms):
    s = [a for a, _ in atoms]
    nu = [b for _, b in atoms]
    lines = []
    with mp.workdps(40):
        k = -(
            mp.quad(lambda t: mp.ncdf(t) * mp.log(mp.ncdf(t)), [-mp.inf, -8, 0])
            + mp.quad(lambda t: mp.ncdf(t) * mp.log(mp.ncdf(t)), [0, 8, mp.inf])
        )
        h1 = abs(B_eval([], [], "0", 40) - 2 * k) < mp.mpf("1e-25")
    lines.append("H1 control 2 kappa: %s" % h1)
    h2 = all(-float(g) <= float(x) <= 0 for x in s) and all(float(m) > 0 for m in nu)
    b30 = B_eval(s, nu, g, 30)
    b50 = B_eval(s, nu, g, 50)
    h3 = abs(b30 - b50) < mp.mpf("1e-20")
    h4 = b50 <= mp.mpf(UPPER)
    bneg = B_eval(s, nu, str(float(g) - 1), 50)
    h5 = abs(bneg - b50) > mp.mpf("1e-6")
    lines.append(
        "H2 feasible %s; B at 50 digits %s; 30 vs 50 %s (H3 %s); H4 %s; negative control B %s (H5 %s)"
        % (h2, mp.nstr(b50, 20), mp.nstr(b30 - b50, 3), h3, h4, mp.nstr(bneg, 12), h5)
    )
    if not (h1 and h5):
        return lines, "VOID", None
    if not (h2 and h3 and h4):
        return lines, "FAIL", None
    bound = mp.floor(b50 * 10**10) / 10**10
    return lines, "PASS", bound


def selftest():
    ok = True
    cases = [
        ("the registered configuration", G, ATOMS, "PASS"),
        ("an atom outside [-g, 0]", G, [("0.5", "3.49413")] + ATOMS[1:], "FAIL"),
        ("a negative mass", G, [ATOMS[0], ("-2.68983", "-2.25076"), ATOMS[2]], "FAIL"),
    ]
    for name, g, at, want in cases:
        _, v, _ = gate(g, at)
        print("selftest %-40s -> %s (want %s)" % (name, v, want))
        ok = ok and v == want
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--registration-run" in sys.argv:
        print(__doc__.split("Usage:")[0][-1500:])
        print("registration run: criteria printed, nothing evaluated")
        sys.exit(0)
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    try:
        lines, v, bound = gate(G, ATOMS)
    except Exception as e:  # a crash is VOID
        print("crash:", repr(e))
        print("VERDICT q1848: VOID")
        sys.exit(2)
    for l in lines:
        print(l)
    if bound is not None:
        print("lower bound of liminf (e^(C+h_1) - 2A) from this configuration: %s" % mp.nstr(bound, 11))
    print("VERDICT q1848: %s" % v)
