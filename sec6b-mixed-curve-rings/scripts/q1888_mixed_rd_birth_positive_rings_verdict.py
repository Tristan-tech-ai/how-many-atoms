"""q1888 (14 Oct 2026, task MIX301): the registered test of amendment 1463 applied to NPMLE rings on the mixed curve whose masses are all
positive. Written before any such ring at rm <= 1.7410 exists (the LAM continuation to 0.2 at rm 1.7416 was still running).

Criterion of 1463, unchanged: rd boundary rings on the mixed curve LAM 0.2, rp 2.1, s 2 (D_solver.py FUNC rd SRC_S 2 LAM 0.2, 200
Gauss-Hermite nodes) have center excess D(0) - C changing sign at rm in [1.73615, 1.73655]; FAIL otherwise.
Precondition added by the object audit of 14 Oct (a ring with a mass <= 0 is not a probability law): every ring read here must have all
masses > 0 and residual < 1e-25.
Input: the output of mix_rm_walk.py (lines 'rm R: residual X, smallest mass M, scan max S, D(0) - C E').
Readout:
  VOID        input missing or no parsable line;
  INCOMPLETE  a ring at 1.73655 or at 1.73615 is missing, or any ring in [1.73615, 1.73655] fails the precondition;
  PASS        D(0) - C at 1.73655 and at 1.73615 have opposite signs (a sign change inside the band) and no ring with rm > 1.73655 read
              has a sign change between adjacent radii (the zero is not above the band);
  FAIL        otherwise (no sign change inside the band, or a sign change above it).
Last line: 'VERDICT q1888: PASS|FAIL|INCOMPLETE|VOID'.
Usage (from temporal_within): py small_gap_q206/q1888_mixed_rd_birth_positive_rings_verdict.py FILE; --registration-run runs the controls.
"""

import re
import sys

LINE = re.compile(
    r"^rm ([0-9.]+): residual ([-+0-9.e]+), smallest mass ([-+0-9.e]+), scan max .*, D\(0\) - C ([-+0-9.e]+)\s*$"
)
HI, LO = 1.73655, 1.73615


def parse(text):
    rows = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            rows[float(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return rows


def verdict(rows):
    if not rows:
        return "VOID"
    hi = min(rows, key=lambda r: abs(r - HI))
    lo = min(rows, key=lambda r: abs(r - LO))
    if abs(hi - HI) > 1e-9 or abs(lo - LO) > 1e-9:
        return "INCOMPLETE"
    for r, (res, mm, ex) in rows.items():
        if LO - 1e-12 <= r <= HI + 1e-12 and not (mm > 0 and res < 1e-25):
            return "INCOMPLETE"
    above = sorted(r for r in rows if r >= HI - 1e-12)
    for a, b in zip(above, above[1:]):
        if rows[a][2] * rows[b][2] < 0:
            return "FAIL"
    return "PASS" if rows[hi][2] * rows[lo][2] < 0 else "FAIL"


def fmt(rm, ex, mm=0.03, res=1e-30):
    return "rm %s: residual %.3e, smallest mass %+.6e, scan max [9.0, 1e-13], D(0) - C %+.10e" % (
        rm,
        res,
        mm,
        ex,
    )


def controls():
    good = "\n".join([fmt("1.7380", 5e-3), fmt("1.73655", 1e-5), fmt("1.73635", 2e-7), fmt("1.73615", -1e-5)])
    above = "\n".join([fmt("1.7380", -5e-3), fmt("1.73655", -1e-5), fmt("1.73615", -2e-5)])
    nochange = "\n".join([fmt("1.73655", 1e-5), fmt("1.73615", 2e-6)])
    negmass = "\n".join([fmt("1.73655", 1e-5), fmt("1.73635", 1e-6, mm=-14.1), fmt("1.73615", -1e-5)])
    missing = "\n".join([fmt("1.73655", 1e-5)])
    ok = (
        verdict(parse(good)) == "PASS"
        and verdict(parse(above)) == "FAIL"
        and verdict(parse(nochange)) == "FAIL"
        and verdict(parse(negmass)) == "INCOMPLETE"
        and verdict(parse(missing)) == "INCOMPLETE"
        and verdict(parse("")) == "VOID"
    )
    print("controls:", "PASS" if ok else "FAIL")
    return ok


def main(argv):
    if argv[:1] == ["--registration-run"]:
        ok = controls()
        print(
            "VERDICT q1888: " + ("VOID (controls only, registration run)" if ok else "VOID (controls failed)")
        )
        return 0 if ok else 2
    if not controls():
        print("VERDICT q1888: VOID")
        return 2
    try:
        text = open(argv[0], encoding="utf-8").read()
    except (IndexError, OSError) as e:
        print("cannot read input: %s" % e)
        print("VERDICT q1888: VOID")
        return 2
    rows = parse(text)
    for r in sorted(rows, reverse=True):
        res, mm, ex = rows[r]
        print("  rm %.5f  residual %.2e  smallest mass %+.4e  D(0) - C %+.6e" % (r, res, mm, ex))
    v = verdict(rows)
    print("VERDICT q1888: " + v)
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
