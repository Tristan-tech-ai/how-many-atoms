"""q1202_k81_family_split_driver.py (8 Oct 2026, registered in amendment 1960): driver of the family K 81 and of the split K 81 -> 82 (event K_b = 81), using the functions of q945_tower (grid 60.0, step 0.005). The driver only runs scripts; every verdict is the last line of the script that produced it.
Steps: (1) trace of the family K 81 from the born-centre state of q922 (tower/K81_stage2, A 51.4868, written by q1192) with q942 and the watcher q946, started while q944 of q1192 still locates the birth amplitude; (2) the registered readouts of the odd family (q926b structure of all states, q911 odd family verdict);
(3) the split K 81 -> 82 through q945_tower.split: A_s (q935, file tower_K82_As.json, the measured amplitude of K_b 81), the three scans, q931 (verdict T64) and the trace of the family K 82 with its readouts (even).
Usage: py q1202_k81_family_split_driver.py  (from temporal_within/, environment Q945_GRID = 60.0); log scratch/q1202_run.log
"""

import glob, os, sys

os.environ["Q945_GRID"] = "60.0"
sys.path.insert(0, ".")
import q945_tower as T

A = T.A
FAM = "tower/K81"
A_START = 51.486843


def main():
    base = sorted(glob.glob("tower/K81_stage2/results_q220_gauss_K81_A*.json"))
    if not base:
        T.fail("born-centre state of K 81 (tower/K81_stage2) missing")
    T.log("family K 81: trace from the born-centre state", base[0], "a_start", A_START)
    tr, wt = T.trace_family(base[0], FAM, 81, A_START)
    tr.wait()
    wt.wait()
    T.log("trace K 81 ended, exit", tr.returncode)
    T.family_readouts(FAM, 81, False)
    fam, K = T.split(FAM, 81)
    T.log(f"split K 81 -> 82 done: family K {K} in {fam}")
    print("RESULT q1202: family K 81 and split K 81 -> 82 with the family K 82 run as logged", flush=True)


if __name__ == "__main__":
    main()
