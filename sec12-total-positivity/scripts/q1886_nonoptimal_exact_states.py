"""q1886_nonoptimal_exact_states.py (14 Oct 2026: exact proofs for the three non-optimal states of Section VIII).

The three stored states of Section VIII that are not optima (stationary states beyond the end of their family):
  results_q207_gauss_K40_A30.1191.json, results_q207_gauss_K70_A46.5787.json, results_q206_gauss_K80_A51.513057.json.
q1882 (box radius 1e-110, grid step 0.01) gave PASS for the first two and FAIL for the K80 state (46 of 1560 initial minors not certainly positive
over the box; audit q1885: all 1560 certainly positive at the polished point). That FAIL stays recorded. This is a NEW run with two instrument
settings changed and registered here before any run: box radius r = 1e-150 (q1885 run 2 found the Krawczyk enclosure 9.2463e-164 at step 0.005)
and grid step h = 0.005 (half of q1882's 0.01; the quadrature widths of q1885 run 2 were 1.5146e-267). Everything else is the unchanged code of
q1882 (main, with its module constants), called in a fresh process per state.

Registered criteria (14 Oct 2026, before the first run): for each of the three states, C1, C2, C3 and C5 of q1882 with these settings (C4 is not
run for a non-optimal state); verdict per state by q1882's rules (PASS; FAIL if C2 or C3 fails with C1 holding; INCOMPLETE if C5 fails; VOID if C1
fails, the polish does not converge or the run crashes). Overall: VERDICT q1886: PASS if all three PASS; FAIL if any is FAIL; else INCOMPLETE or VOID.
A PASS means: the box of radius 1e-150 around the polished state contains exactly one stationary state (Krawczyk), all its masses are positive,
and all m(m + 1) initial minors of X are positive over the box, so X is strictly totally positive at the exact stationary state and its positions
move outward along the branch (Lemma lem:cramer). It says nothing about optimality.
Control (registered): the OPTIMAL state results_q206_gauss_K40_A29.85.json run with the same settings and --optimal 0 must give PASS.
Selftest (--selftest): the same control state with its first position moved by 1e-30 and no polish must give FAIL (not a zero within 1e-150).
Usage: py small_gap_q206/q1886_nonoptimal_exact_states.py [--registration-run | --selftest | --control | --one STATE]   (from temporal_within/)
Without arguments it runs the three states, each in its own process (no console window), and prints the overall verdict last.
"""

import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
STATES = [
    "results_q207_gauss_K40_A30.1191.json",
    "results_q207_gauss_K70_A46.5787.json",
    "results_q206_gauss_K80_A51.513057.json",
]
CONTROL = "results_q206_gauss_K40_A29.85.json"
NOWIN = 0x08000000 if os.name == "nt" else 0


def one(state, corrupt=False):
    """runs q1882.main on one state with r = 1e-150 and h = 0.005; returns q1882's exit code (0 PASS, 1 FAIL or INCOMPLETE, 2 VOID)"""
    sys.path.insert(0, HERE)
    sys.path.insert(0, TW)
    os.chdir(TW)
    import q1882_exact_state_closed_chain as C
    from flint import arb

    C.R_BOX = arb("1e-150")
    C.F.HD = C.F.HD / 2  # q1882.main halves HD and G.H once more, so the step becomes 0.005
    C.F.G.H = C.F.G.H / 2
    return C.main(state, False, corrupt=corrupt)


def run_child(args):
    p = subprocess.run(
        [sys.executable, os.path.abspath(__file__)] + args,
        cwd=TW,
        capture_output=True,
        text=True,
        creationflags=NOWIN,
    )
    sys.stdout.write(p.stdout)
    if p.stderr.strip():
        sys.stdout.write("[stderr] " + p.stderr.strip()[-400:] + "\n")
    lines = [l.strip() for l in p.stdout.splitlines() if l.startswith("VERDICT q1882")]
    return lines[-1].split(":")[-1].strip() if lines else "VOID"


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--registration-run" in a:
        print(__doc__.split("Usage:")[0])
        print("registration run: criteria printed, nothing evaluated")
        sys.exit(0)
    if "--one" in a:
        try:
            sys.exit(one(a[a.index("--one") + 1], corrupt="--corrupt" in a))
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("VERDICT q1882: VOID (crash: %r)" % (e,))
            sys.exit(2)
    if "--selftest" in a:
        v = run_child(["--one", CONTROL, "--corrupt"])
        print("SELFTEST: %s (the corrupted control gave %s)" % ("PASS" if v == "FAIL" else "FAIL", v))
        sys.exit(0 if v == "FAIL" else 1)
    if "--control" in a:
        v = run_child(["--one", CONTROL])
        print("CONTROL: %s (the optimal control state gave %s)" % ("PASS" if v == "PASS" else "FAIL", v))
        sys.exit(0 if v == "PASS" else 1)
    res = {}
    for s in STATES:
        print("=== state", s, flush=True)
        res[s] = run_child(["--one", s])
        print("state verdict:", s, res[s], flush=True)
    vals = list(res.values())
    if all(v == "PASS" for v in vals):
        overall = "PASS"
    elif any(v == "FAIL" for v in vals):
        overall = "FAIL"
    elif any(v == "VOID" for v in vals):
        overall = "VOID"
    else:
        overall = "INCOMPLETE"
    print(
        "summary:",
        ", ".join("%s %s" % (s.split("_gauss_")[1].replace(".json", ""), v) for s, v in res.items()),
    )
    print("VERDICT q1886: %s" % overall)
    sys.exit(0 if overall == "PASS" else 1)
