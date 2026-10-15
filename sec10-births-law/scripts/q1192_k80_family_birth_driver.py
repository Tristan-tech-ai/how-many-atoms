"""q1192_k80_family_birth_driver.py (8 Oct 2026, registered in amendment 1941): driver of the family K 80 and of the birth K 80 -> 81 (event K_b = 80), using the functions of q945_tower (grid 60.0, step 0.005).
Steps: (1) trace of the family K 80 from the root state of the split K 79 -> 80 (tower/K80_split_hi2/results_q220_gauss_K80_A50.991688_eps0.13835.json) with q942 and the watcher q946; (2) the registered readouts of the family (q926b structure, q933 even family) via T.family_readouts;
(3) the birth: q922 (born centre state K 81), q944 (A_cross), q938 (verdict T67); the family K 81 is NOT traced. Outputs scratch/tower_K80_*.txt, tower_K81_birth_out.json, tower_K81_q938.txt. The driver only runs scripts; every verdict is the last line of its script.
Usage: py q1192_k80_family_birth_driver.py  (from temporal_within/, environment Q945_GRID = 60.0)"""

import glob, json, os, sys

os.environ.setdefault("Q945_GRID", "60.0")
sys.path.insert(0, ".")
import q945_tower as T

BASE = "tower/K80_split_hi2/results_q220_gauss_K80_A50.991688_eps0.13835.json"
FAM = "tower/K80"
A = T.A


def main():
    if not os.path.exists(BASE):
        T.fail(f"root state {BASE} missing")
    T.log("family K 80: trace from the root state", BASE)
    tr, wt = T.trace_family(BASE, FAM, 80, 50.991688)
    tr.wait()
    wt.wait()
    T.log("trace K 80 ended, exit", tr.returncode)
    T.family_readouts(FAM, 80, True)
    opt, later = T.split_family(T.states(FAM, 80))
    if not opt or not later or later[0][2] != "centre positive":
        T.fail("family K 80 does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    T.log(f"birth K 80 -> 81: first non-optimal state A {hi[0]}, last optimal A {lo[0]}")
    stage = "tower/K81_stage2"
    os.makedirs(stage, exist_ok=True)
    rc = T.run(["q922_born_centre_run.py", hi[1], stage], f"{A}/tower_K81_born.log")
    base = sorted(glob.glob(f"{stage}/results_q220_gauss_K81_A*.json"))
    if rc != 0 or not base:
        T.fail(f"q922 for K 81 exit {rc}")
    bj = f"{A}/tower_K81_birth_out.json"
    rcb = T.run(["q944_birth_generic.py", base[0], hi[1], lo[1], bj, T.GRID], f"{A}/tower_K81_birth.log")
    if rcb != 0:
        T.fail(f"q944 for K 81 exit {rcb}")
    T.run(["q938_birth_verdict.py", bj], f"{A}/tower_K81_q938.txt")
    T.log("RESULT q1192: family K 80 and birth K 80 -> 81 run as logged")


if __name__ == "__main__":
    main()
