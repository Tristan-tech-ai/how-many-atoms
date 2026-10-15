"""rows_cycle2 (9 Oct 2026, helper, no verdict word): the block K = 123..130 of rows_cycle.py. After cap_cycle.sh has merged new capacity events into fw_prod, make every margin row K = 123..130 whose events K-1 and K exist in fw_prod and whose states directory exists and which is not yet in small_gap_q206/q1416_rows.json (controlled route of q1416_stage_row.py, log only, row values are never printed), then run the gate q1494 (centre_scripts/q1494_margin_series_ext_verdict.py) once per new row and save its output as q1494_out_<tag>_row<K>.txt. The first run of q1494 after all eight rows exist is the verdict (amendment). The states directory of a row K is the first of fw_cap_par_chain<K> (even K) or fw_cap_par_chain<K+1> (odd K) (for the chain 128 the directories 128d and 128e) and fw_gpu_cap whose nearest saved state of the family K lies within 0.02 of the row target; the directory used is printed. Usage: py scratch/rows_cycle2.py TAG (from temporal_within/)."""

import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SG = os.path.join(ROOT, "small_gap_q206")
CS = os.path.join(ROOT, "centre_scripts")
A = os.path.join(ROOT, "scratch")
tag = sys.argv[1] if len(sys.argv) > 1 else "r"
env = dict(os.environ, PYTHONIOENCODING="utf-8")


def have_event(k):
    return os.path.exists(os.path.join(SG, "fw_prod", f"event_Kb{k}.json"))


def have_row(k):
    try:
        return any(r["K"] == k for r in json.load(open(os.path.join(SG, "q1416_rows.json"))))
    except Exception:
        return False


def ev_A(k):
    return float(json.load(open(os.path.join(SG, "fw_prod", f"event_Kb{k}.json")))["A_event"])


def near_dist(name, K):
    """distance of the saved state of the family K nearest to the row target (97 percent of the family between the events K - 1 and K); None when the directory has no state of K"""
    a0, a1 = ev_A(K - 1), ev_A(K)
    At = a0 + 0.97 * (a1 - a0)
    import glob

    As = [
        float(os.path.basename(f).split("_A")[1][:-5])
        for f in glob.glob(os.path.join(SG, name, f"results_q1411_gauss_K{K}_A*.json"))
    ]
    return min(abs(a - At) for a in As) if As else None


def states_dir(K):
    """first directory whose nearest state of the family K lies within 0.02 of the row target: the chain directory, else the GPU walk directory fw_gpu_cap (rule R4 of amendment 2186; the dump re-solves at the target to a residual below 1e-40, so the source of the start state does not enter the row)"""
    c = K if K % 2 == 0 else K + 1
    names = ([f"fw_cap_par_chain{c}d", f"fw_cap_par_chain{c}e"] if c == 128 else [f"fw_cap_par_chain{c}"]) + [
        "fw_gpu_cap"
    ]
    for name in names:
        if os.path.isdir(os.path.join(SG, name)):
            d = near_dist(name, K)
            if d is not None and d <= 0.02:
                return name
    return None


made = []
for K in range(123, 131):
    if have_row(K):
        print(f"row {K}: already present")
        continue
    if not (have_event(K - 1) and have_event(K)):
        print(f"row {K}: skipped (events K_b {K-1} and {K} not both in fw_prod)")
        continue
    sd = states_dir(K)
    if sd is None:
        print(f"row {K}: skipped (no saved state of the family within 0.02 of the target yet)")
        continue
    e = dict(env, Q1416_STATES=sd, Q1416_DUMP=f"q1415_jb_K{K}_f0.97.json")
    r = subprocess.run(
        [sys.executable, "q1416_stage_row.py", str(K), str(K - 1)],
        cwd=SG,
        env=e,
        capture_output=True,
        text=True,
    )
    open(os.path.join(SG, f"q1416_row{K}.log"), "w", encoding="utf-8").write(r.stdout + r.stderr)
    print(
        f"row {K}: {'made' if r.returncode == 0 and have_row(K) else 'FAILED (see q1416_row%d.log, not printed here)' % K} (states {sd})"
    )
    if have_row(K):
        g = subprocess.run(
            [sys.executable, "q1494_margin_series_ext_verdict.py"],
            cwd=CS,
            env=env,
            capture_output=True,
            text=True,
        )
        open(os.path.join(A, f"q1494_out_{tag}_row{K}.txt"), "w", encoding="utf-8").write(g.stdout + g.stderr)
        print(
            "   q1494:",
            [l for l in g.stdout.splitlines() if l.startswith("VERDICT")][-1:] or ["NO VERDICT LINE"],
            "exit",
            g.returncode,
        )
        made.append(K)
print("rows made this run:", made)
