"""q1270_k82_birth_resume.py (8 Oct 2026, registered in amendment 2045): resume of the birth K 82 -> 83 after the VOID of q944 (amendment 2044 driver q1265: 'VOID q937: K 61 corrector failed at A 52.472399'). Same inputs as q1265 (first non-optimal and last optimal state of tower/K82, born centre state of K 83 from q922 in tower/K83_stage2), but the script that finds A_cross is q1138_birth_generic_adaptive2.py (substeps after a failed corrector), as in the resumes of the events K_b 74 to 79. Then q938. Outputs scratch/tower_K83_birth2.log, tower_K83_birth_out.json, tower_K83_q938.txt.
Usage: py q1270_k82_birth_resume.py  (from temporal_within/, environment Q945_GRID = 60.0)"""

import glob, os, sys

os.environ["Q945_GRID"] = "60.0"
sys.path.insert(0, ".")
import q945_tower as T

FAM = "tower/K82"
A = T.A


def main():
    opt, later = T.split_family(T.states(FAM, 82))
    if not opt or not later or later[0][2] != "centre positive":
        T.fail("family K 82 does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    T.log(f"birth K 82 -> 83 (resume): first non-optimal state A {hi[0]}, last optimal A {lo[0]}")
    base = sorted(glob.glob("tower/K83_stage2/results_q220_gauss_K83_A*.json"))
    if not base:
        T.fail("born centre state of K 83 missing")
    bj = f"{A}/tower_K83_birth_out.json"
    rcb = T.run(
        ["q1138_birth_generic_adaptive2.py", base[0], hi[1], lo[1], bj, T.GRID], f"{A}/tower_K83_birth2.log"
    )
    if rcb != 0:
        T.fail(f"q1138 for K 83 exit {rcb}")
    T.run(["q938_birth_verdict.py", bj], f"{A}/tower_K83_q938.txt")
    T.log("RESULT q1270: birth K 82 -> 83 run as logged")


if __name__ == "__main__":
    main()
