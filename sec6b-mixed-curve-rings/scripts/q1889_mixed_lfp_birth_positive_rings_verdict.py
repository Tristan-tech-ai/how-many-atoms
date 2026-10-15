"""q1889 (15 Oct 2026, task MIXL; text copy of q1888 for the LFP): the registered test of amendment 1461 applied to LFP rings on the
mixed curve whose masses are all positive. Written before any such ring at rm < 1.4036 on LAM 0.1 exists (the LAM continuation was
still running).

Criterion of 1461, unchanged: LFP boundary rings on the mixed curve LAM 0.1, rp 1.7 (D_solver.py FUNC lfp LAM 0.1; here 120
Gauss-Hermite nodes, 128 bits, as the stored rings of 1462) have center excess r(0) - C changing sign at rm in [1.40137, 1.40177]; FAIL
otherwise.
Precondition added by the object audit of 14 Oct (a ring with a mass <= 0 is not a probability law): every ring read here must have all
masses > 0 and residual < 1e-25.
Input: the output of mix_rm_walk_lfp.py (lines 'rm R: residual X, smallest mass M, scan max S, r(0) - C E').
Readout:
  VOID        input missing or no parsable line;
  INCOMPLETE  a ring at 1.40177 or at 1.40137 is missing, or any ring in [1.40137, 1.40177] fails the precondition;
  PASS        r(0) - C at 1.40177 and at 1.40137 have opposite signs (a sign change inside the band) and no ring with rm > 1.40177 read
              has a sign change between adjacent radii (the zero is not above the band);
  FAIL        otherwise (no sign change inside the band, or a sign change above it).
Last line: 'VERDICT q1889: PASS|FAIL|INCOMPLETE|VOID'.
Usage (from temporal_within): py small_gap_q206/q1889_mixed_lfp_birth_positive_rings_verdict.py FILE; --registration-run runs the controls.
"""

import re
import sys

LINE = re.compile(
    r"^rm ([0-9.]+): residual ([-+0-9.e]+), smallest mass ([-+0-9.e]+), scan max .*, r\(0\) - C ([-+0-9.e]+)\s*$"
)
HI, LO = 1.40177, 1.40137


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
    return "rm %s: residual %.3e, smallest mass %+.6e, scan max [9.0, 1e-13], r(0) - C %+.10e" % (
        rm,
        res,
        mm,
        ex,
    )


def controls():
    good = "\n".join([fmt("1.4030", 5e-3), fmt("1.40177", 1e-5), fmt("1.40157", 2e-7), fmt("1.40137", -1e-5)])
    above = "\n".join([fmt("1.4030", -5e-3), fmt("1.40177", -1e-5), fmt("1.40137", -2e-5)])
    nochange = "\n".join([fmt("1.40177", 1e-5), fmt("1.40137", 2e-6)])
    negmass = "\n".join([fmt("1.40177", 1e-5), fmt("1.40157", 1e-6, mm=-14.1), fmt("1.40137", -1e-5)])
    missing = "\n".join([fmt("1.40177", 1e-5)])
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
            "VERDICT q1889: " + ("VOID (controls only, registration run)" if ok else "VOID (controls failed)")
        )
        return 0 if ok else 2
    if not controls():
        print("VERDICT q1889: VOID")
        return 2
    try:
        text = open(argv[0], encoding="utf-8").read()
    except (IndexError, OSError) as e:
        print("cannot read input: %s" % e)
        print("VERDICT q1889: VOID")
        return 2
    rows = parse(text)
    for r in sorted(rows, reverse=True):
        res, mm, ex = rows[r]
        print("  rm %.5f  residual %.2e  smallest mass %+.4e  r(0) - C %+.6e" % (r, res, mm, ex))
    v = verdict(rows)
    print("VERDICT q1889: " + v)
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
