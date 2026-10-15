"""q1068 (7 Oct 2026): driver for the extension of the tower beyond K 71 (T104, registered in the amendment that follows 1821). It imports the helper functions of q945_tower.py (unchanged) and runs, one after the other,
(1) the split K 71 -> 72 without its first step (the A_s of tower_K72_As.json exists since 1820): three scans, q931, the K 72 trace, q926b and q933 (text of q945.split after the q935 As call);
(2) the birth K 72 -> 73 (text of q945.birth with q1063_birth_generic_adaptive.py in place of q944_birth_generic.py, 1813): q922, K 73 trace, birth script -> tower_K73_birth_out.json, q938, q926b and q911;
(3) the first step of the split K 73 -> 74 (q935 As -> tower_K74_As.json) and nothing more.
Like q945 it decides nothing; every verdict is the last line of the script that produced it. Usage: py q1068_tower_resume3.py (from temporal_within/); log in scratch/q1068_resume.log (stdout).
"""

import glob, json, os
import q945_tower as T

A = T.A
log = T.log
run = T.run
popen = T.popen
fail = T.fail
clean = T.clean


def split_after_as(odd_dir, K):
    opt, later = T.split_family(T.states(odd_dir, K))
    if not opt or not later or later[0][2] != "centre positive":
        fail(f"odd family K {K} in {odd_dir} does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    mid = min(opt, key=lambda s: abs(s[0] - (lo[0] - 0.08)))
    Ke = K + 1
    odir = f"tower/K{Ke}_split"
    fam = f"tower/K{Ke}"
    os.makedirs(odir, exist_ok=True)
    log(
        f"split K {K} -> {Ke}: first non-optimal A {hi[0]}, last optimal A {lo[0]}, negative control A {mid[0]}"
    )
    env = dict(os.environ, Q935_GRID=T.GRID, Q935_ODIR=odir)
    asj = f"{A}/tower_K{Ke}_As.json"
    if not os.path.exists(asj):
        fail(f"{asj} does not exist")
    scans = {
        tag: (
            f"{A}/tower_K{Ke}_scan_{tag}.json",
            popen(
                ["q935_split_generic.py", "scan", s[1], f"{A}/tower_K{Ke}_scan_{tag}.json"],
                f"{A}/tower_K{Ke}_scan_{tag}.log",
                env,
            ),
        )
        for tag, s in (("hi", hi), ("lo", lo), ("mid", mid))
    }
    for tag, (jf, p) in scans.items():
        p.wait()
        if p.returncode != 0:
            fail(f"scan {tag} for K {Ke} exit {p.returncode}")
    run(
        ["q931_split_verdict.py", scans["hi"][0], scans["lo"][0], scans["mid"][0], asj],
        f"{A}/tower_K{Ke}_q931.txt",
    )
    root = json.load(open(scans["hi"][0]))["root"]
    if not root or "eps" not in root:
        fail(f"no root for K {Ke}")
    cands = [
        f
        for f in glob.glob(f"{odir}/results_q220_gauss_K{Ke}_A*_eps*.json")
        if abs(clean(json.load(open(f))["A"]) - hi[0]) < 1e-9
    ]
    if len(cands) != 1:
        fail(f"root state file for K {Ke}: {len(cands)} candidates")
    tr, wt = T.trace_family(cands[0], fam, Ke, hi[0])
    tr.wait()
    wt.wait()
    T.family_readouts(fam, Ke, True)
    return fam, Ke


def birth_adaptive(even_dir, K):
    opt, later = T.split_family(T.states(even_dir, K))
    if not opt or not later or later[0][2] != "centre positive":
        fail(f"even family K {K} in {even_dir} does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    Ko = K + 1
    stage = f"tower/K{Ko}_stage"
    fam = f"tower/K{Ko}"
    os.makedirs(stage, exist_ok=True)
    log(f"birth K {K} -> {Ko}: first non-optimal state A {hi[0]}, last optimal A {lo[0]}")
    rc = run(["q922_born_centre_run.py", hi[1], stage], f"{A}/tower_K{Ko}_born.log")
    base = glob.glob(f"{stage}/results_q220_gauss_K{Ko}_A*.json")
    if rc != 0 or not base:
        fail(f"q922 for K {Ko} exit {rc}")
    base = sorted(base)[0]
    tr, wt = T.trace_family(base, fam, Ko, hi[0])
    bj = f"{A}/tower_K{Ko}_birth_out.json"
    rcb = run(
        ["q1063_birth_generic_adaptive.py", base, hi[1], lo[1], bj, T.GRID], f"{A}/tower_K{Ko}_birth.log"
    )
    if rcb != 0:
        fail(f"q1063 for K {Ko} exit {rcb}")
    run(["q938_birth_verdict.py", bj], f"{A}/tower_K{Ko}_q938.txt")
    tr.wait()
    wt.wait()
    T.family_readouts(fam, Ko, False)
    return fam, Ko


def main():
    log("q1068 start: tower beyond K 71")
    fam72, K72 = split_after_as("tower/K71", 71)
    log(f"split K 71 -> 72 done: family K {K72} in {fam72}")
    fam73, K73 = birth_adaptive(fam72, 72)
    log(f"birth K 72 -> 73 done: family K {K73} in {fam73}")
    opt, later = T.split_family(T.states(fam73, 73))
    if not opt or not later or later[0][2] != "centre positive":
        fail("odd family K 73 does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    env = dict(os.environ, Q935_GRID=T.GRID, Q935_ODIR="tower/K74_split")
    os.makedirs("tower/K74_split", exist_ok=True)
    log(f"split K 73 -> 74: first non-optimal A {hi[0]}, last optimal A {lo[0]}")
    rc = run(
        ["q935_split_generic.py", "As", f"{A}/tower_K74_As.json", lo[1], hi[1]], f"{A}/tower_K74_As.log", env
    )
    log(f"A_s for K 74: exit {rc}")
    print("RESULT q1068: events K_b 72 and 73 measured or failed as logged", flush=True)


if __name__ == "__main__":
    main()
