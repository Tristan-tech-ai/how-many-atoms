"""q1083 (7 Oct 2026): driver for the extension of the tower beyond K 74 (T116, registered in the amendment that follows 1835). It waits for the driver q1068 (process id given by the environment variable Q1083_WAIT_PID) to end, then runs
(1) the split K 73 -> 74 without its first step (the A_s of tower_K74_As.json comes from q1068): three scans, q931, the K 74 trace, q926b and q933;
(2) alternately the birth K -> K + 1 for even K (q922, the K + 1 trace, q1063 adaptive birth script -> tower_K<K+1>_birth_out.json, q938, q926b and q911) and the full split K -> K + 1 for odd K (q945.split: q935 As -> tower_K<K+1>_As.json, three scans, q931, trace, q926b and q933), until the family K 79 is traced and the first step of the split K 79 -> 80 (q935 As -> tower_K80_As.json) is done.
The grid is frozen at 60.0 (range -69 to +69) from the family K 74 on, because the wall atoms of the families above K 73 lie closer than 7 standard deviations to the edge of the grid 46.0 (range -55 to +55). The base state of the K 74 trace is polished on the new grid by q942.
Like q945 and q1068 it decides nothing; every verdict is the last line of the script that produced it. Usage: py q1083_tower_resume4.py (from temporal_within/); log in scratch/q1083_resume.log (stdout).
"""

import os, subprocess, sys, time

os.environ["Q945_GRID"] = "60.0"
import q945_tower as T
import q1068_tower_resume3 as Q68

A = T.A
log = T.log
run = T.run
fail = T.fail


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    pid = os.environ.get("Q1083_WAIT_PID")
    if pid:
        log(f"q1083 waits for process {pid} (q1068)")
        while alive(pid):
            time.sleep(30)
    if not os.path.exists(f"{A}/tower_K74_As.json"):
        fail("tower_K74_As.json does not exist (q1068 did not finish its last step)")
    log(f"q1083 start: tower beyond K 73 with the grid {T.GRID}")
    fam, K = Q68.split_after_as("tower/K73", 73)
    log(f"split K 73 -> 74 done: family K {K} in {fam}")
    while K < 79:
        if K % 2 == 0:
            fam, K = Q68.birth_adaptive(fam, K)
            log(f"birth K {K - 1} -> {K} done: family K {K} in {fam}")
        else:
            fam, K = T.split(fam, K)
            log(f"split K {K - 1} -> {K} done: family K {K} in {fam}")
    opt, later = T.split_family(T.states(fam, K))
    if not opt or not later or later[0][2] != "centre positive":
        fail("odd family K 79 does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    env = dict(os.environ, Q935_GRID=T.GRID, Q935_ODIR="tower/K80_split")
    os.makedirs("tower/K80_split", exist_ok=True)
    log(f"split K 79 -> 80: first non-optimal A {hi[0]}, last optimal A {lo[0]}")
    rc = run(
        ["q935_split_generic.py", "As", f"{A}/tower_K80_As.json", lo[1], hi[1]], f"{A}/tower_K80_As.log", env
    )
    log(f"A_s for K 80: exit {rc}")
    print("RESULT q1083: events K_b 74 to 79 measured or failed as logged", flush=True)


if __name__ == "__main__":
    main()
