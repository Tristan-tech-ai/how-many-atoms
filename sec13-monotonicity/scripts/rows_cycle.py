"""rows_cycle (9 Oct 2026, helper, no verdict): after cap_cycle.sh has merged new capacity events into fw_prod, make every margin row K = 115..122 whose events K-1 and K exist and whose states directory exists and whose row is not yet in small_gap_q206/q1416_rows.json, then run the registered gate q1452 once per new row. Row K needs the events K_b K-1 and K_b K in fw_prod and the states of the family K: even K from fw_cap_par_chain<K>, odd K from fw_cap_par_chain<K+1> (route controlled in amendment 2177). The row values are NEVER printed here; only 'made' or 'skipped' and the gate output (counts and violations). Usage: py scratch/rows_cycle.py TAGPREFIX (from temporal_within/); outputs q1452_out_<TAGPREFIX>_row<K>.txt in scratch/."""

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


made = []
for K in range(115, 123):
    if have_row(K):
        print(f"row {K}: already present")
        continue
    sd = f"fw_cap_par_chain{K if K % 2 == 0 else K + 1}"
    if not (have_event(K - 1) and have_event(K)):
        print(f"row {K}: skipped (events K_b {K-1} and {K} not both in fw_prod)")
        continue
    if not os.path.isdir(os.path.join(SG, sd)):
        print(f"row {K}: skipped (no states directory {sd})")
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
        f"row {K}: {'made' if r.returncode == 0 and have_row(K) else 'FAILED (see q1416_row%d.log, not printed here)' % K}"
    )
    if have_row(K):
        g = subprocess.run(
            [sys.executable, "q1452_margin_law_ext_verdict.py"],
            cwd=CS,
            env=env,
            capture_output=True,
            text=True,
        )
        open(os.path.join(A, f"q1452_out_{tag}_row{K}.txt"), "w", encoding="utf-8").write(g.stdout + g.stderr)
        print(
            "   q1452:",
            [l for l in g.stdout.splitlines() if l.startswith("VERDICT")][-1:] or ["NO VERDICT LINE"],
            "exit",
            g.returncode,
        )
        made.append(K)
        g2 = subprocess.run(
            [sys.executable, "q1462_margin_series_verdict.py"],
            cwd=CS,
            env=env,
            capture_output=True,
            text=True,
        )
        open(os.path.join(A, f"q1462_out_{tag}_row{K}.txt"), "w", encoding="utf-8").write(
            g2.stdout + g2.stderr
        )
        print(
            "   q1462:",
            [l for l in g2.stdout.splitlines() if l.startswith("VERDICT")][-1:] or ["NO VERDICT LINE"],
            "exit",
            g2.returncode,
        )
print("rows made this run:", made)
