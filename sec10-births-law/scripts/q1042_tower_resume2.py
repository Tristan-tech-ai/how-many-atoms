"""q1042 (7 Oct 2026): driver for the extension of the tower beyond K 69 (T98, amendment 1799). It imports the stage functions of q945_tower.py (unchanged) and runs, one after the other,
(1) the split K 69 -> 70 as a stage (q935 As again, three scans, q931, K 70 trace, q926b and q933); the A_s of the first K 70 run is kept in tower_K70_As_T83.json;
(2) the birth K 70 -> 71 (q922, K 71 trace, q944 -> tower_K71_birth_out.json, q938, q926b and q911);
(3) the first step of the split K 71 -> 72 (q935 As -> tower_K72_As.json) and nothing more.
Like q945 and q994 it decides nothing; every verdict is the last line of the script that produced it. Usage: py q1042_tower_resume2.py (from temporal_within/); log in scratch/q1042_resume.log (stdout).
"""

import os
import q945_tower as T

A = T.A
log = T.log


def main():
    log("q1042 start: tower beyond K 69")
    fam70, K70 = T.split("tower/K69", 69)
    log(f"split K 69 -> 70 done: family K {K70} in {fam70}")
    fam71, K71 = T.birth(fam70, 70)
    log(f"birth K 70 -> 71 done: family K {K71} in {fam71}")
    opt, later = T.split_family(T.states(fam71, 71))
    if not opt or not later or later[0][2] != "centre positive":
        T.fail("odd family K 71 does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    env = dict(os.environ, Q935_GRID=T.GRID, Q935_ODIR="tower/K72_split")
    os.makedirs("tower/K72_split", exist_ok=True)
    log(f"split K 71 -> 72: first non-optimal A {hi[0]}, last optimal A {lo[0]}")
    rc = T.run(
        ["q935_split_generic.py", "As", f"{A}/tower_K72_As.json", lo[1], hi[1]], f"{A}/tower_K72_As.log", env
    )
    log(f"A_s for K 72: exit {rc}")
    print("RESULT q1042: events K_b 70 and 71 measured or failed as logged", flush=True)


if __name__ == "__main__":
    main()
