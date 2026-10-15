"""q1891_readout.py (15 Oct 2026): readout of amendment 2255 over the 39 odd-K states of q1843_pr1b_states.txt.
Reads small_gap_q206/q1891_out/<state basename>.txt, takes the last VERDICT line of each file and the exit line.
Readout (registered in 2255): PASS if every one of the 39 states gives PASS; FAIL if one gives FAIL; otherwise INCOMPLETE (a state missing,
unfinished or INCOMPLETE) or VOID (a state VOID). Prints one line per state, the smallest dw_0/dA and relative growth, then the VERDICT line.
--selftest runs the decision rule on synthetic lists only. Usage: py small_gap_q206/q1891_readout.py [--selftest] (from temporal_within/)
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def decide(vs, n_expected):
    if len(vs) < n_expected or any(v is None for v in vs):
        base = "INCOMPLETE"
    else:
        base = None
    if any(v == "VOID" for v in vs):
        return "VOID"
    if any(v == "FAIL" for v in vs):
        return "FAIL"
    if base or any(v != "PASS" for v in vs):
        return "INCOMPLETE"
    return "PASS"


def selftest():
    ok = (
        decide(["PASS"] * 3, 3) == "PASS"
        and decide(["PASS", "FAIL", "PASS"], 3) == "FAIL"
        and decide(["PASS", None, "PASS"], 3) == "INCOMPLETE"
        and decide(["PASS"] * 2, 3) == "INCOMPLETE"
        and decide(["PASS", "VOID", "FAIL"], 3) == "VOID"
        and decide(["PASS", "INCOMPLETE", "PASS"], 3) == "INCOMPLETE"
    )
    print("selftest decision rule:", ok)
    return ok


def main():
    states = [
        l.split()[0] for l in open(os.path.join(HERE, "q1843_pr1b_states.txt"), encoding="utf-8") if l.strip()
    ]
    odd = [s for s in states if int(re.search(r"_K(\d+)_", s).group(1)) % 2 == 1]
    if len(odd) != 39:
        print("VERDICT q1891 readout: VOID (expected 39 odd-K states, found %d)" % len(odd))
        return 2
    vs = []
    dws = []
    rel = []
    for s in odd:
        f = os.path.join(HERE, "q1891_out", os.path.basename(s).replace(".json", ".txt"))
        v = None
        if os.path.exists(f):
            txt = open(f, encoding="utf-8", errors="replace").read()
            vm = re.findall(r"^VERDICT q1891: (\w+)", txt, re.M)
            ex = re.findall(r"^exit (\d+)", txt, re.M)
            v = vm[-1] if vm and ex else None
            m1 = re.search(r"dw_0/dA = \[([-0-9.e+]+) \+/- ([0-9.e+-]+)\]", txt)
            m2 = re.search(r"relative growth \(dw_0/dA\)/w_0 = ([-0-9.e+]+)", txt)
            if m1:
                dws.append(float(m1.group(1)) - float(m1.group(2)))
            if m2:
                rel.append(float(m2.group(1)))
        vs.append(v)
        print("%-60s %s" % (s, v))
    print(
        "states 39; files with a verdict %d; smallest lower end of dw_0/dA %s; smallest relative growth %s"
        % (sum(v is not None for v in vs), min(dws) if dws else "none", min(rel) if rel else "none")
    )
    d = decide(vs, 39)
    print("VERDICT q1891 readout: %s" % d)
    return 0 if d == "PASS" else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--registration-run" in sys.argv:
        print(__doc__)
        print("registration run: nothing read")
        sys.exit(0)
    sys.exit(main())
