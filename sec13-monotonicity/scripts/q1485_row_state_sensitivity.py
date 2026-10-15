"""q1485 (9 Oct 2026, READING, no verdict word; closes the open point of readings 18 and 24): does an error of the starting state change a margin row? The row script q1416_stage_row.py takes the stored state nearest to the stage
f0.97 of family K, re-solves it at A_target (q1415_dump_fast.py) and builds P/N and R_e. Here the row K = 110 (non-blind, even, states of fw_cap_par_chain110) is rebuilt from a COPY of that nearest state whose interior positions are
multiplied by (1 + eps r_k), r_k fixed random numbers in [-1, 1], for eps = 0 (control), 1e-10 and 1e-8, rows written to a private rows file (Q1416_ROWSFILE), never to q1416_rows.json.
Criteria written before the run: the control (eps = 0) reproduces the stored row 110 in PN and Re to 1e-12, else VOID. Reported: Delta PN and Delta Re against the stored row for each eps. INSENSITIVE if both differences are below 1e-12 at eps = 1e-10
(the re-solve removes the starting error); otherwise the gradient g = Delta / eps is reported and the implied change for a state error of 1e-20 and 1e-28 is g times that error.
Usage: py q1485_row_state_sensitivity.py (from small_gap_q206/; runs the three rows in parallel, about 5 minutes).
"""

import json, os, glob, subprocess, sys, shutil
import numpy as np
from mpmath import mp, mpf

sys.argv = [sys.argv[0]]
import q1416_stage_row as R

mp.dps = 40
K = 110
a0 = R.event_A(K - 1)
a1 = R.event_A(K)
At = a0 + mpf("0.97") * (a1 - a0)
fs = sorted(
    glob.glob(os.path.join("fw_cap_par_chain110", f"results_q1411_gauss_K{K}_A*.json")),
    key=lambda f: abs(float(f.split("_A")[1][:-5]) - float(At)),
)
src = fs[0]
d = json.load(open(src))
d = d[-1] if isinstance(d, list) else d
rng = np.random.default_rng(11)
r = rng.uniform(-1, 1, len(d["u"]))
stored = [x for x in json.load(open("q1416_rows.json")) if x["K"] == K][0]
procs = {}
for eps in (0.0, 1e-10, 1e-8):
    tag = f"eps{eps:.0e}".replace("+", "")
    sd = f"tmp_q1485_states_{tag}"
    shutil.rmtree(sd, ignore_errors=True)
    os.makedirs(sd)
    dd = dict(d)
    dd["u"] = [str(mpf(u) * (1 + mpf(eps) * mpf(float(ri)))) for u, ri in zip(d["u"], r)]
    json.dump(dd, open(os.path.join(sd, os.path.basename(src)), "w"))
    rows = f"tmp_q1485_rows_{tag}.json"
    if os.path.exists(rows):
        os.remove(rows)
    env = dict(os.environ, Q1416_STATES=sd, Q1416_DUMP=f"tmp_q1485_dump_{tag}.json", Q1416_ROWSFILE=rows)
    procs[eps] = (
        subprocess.Popen(
            [sys.executable, "q1416_stage_row.py", str(K), str(K - 1)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ),
        rows,
    )
res = {}
for eps, (p, rows) in procs.items():
    out, _ = p.communicate()
    try:
        row = json.load(open(rows))[0]
        res[eps] = row
    except Exception:
        print(f"eps {eps}: row failed: {out[-200:]}")
print(
    f"stored row K = {K}: PN {stored['PN']:.15f} Re {stored['Re']:.15f}; state file {os.path.basename(src)}"
)
for eps, row in res.items():
    dpn = row["PN"] - stored["PN"]
    dre = row["Re"] - stored["Re"]
    print(
        f"eps {eps:.0e}: PN {row['PN']:.15f} Re {row['Re']:.15f} control_ok {row['control_ok']}; Delta PN {dpn:+.2e}, Delta Re {dre:+.2e}"
        + (f"; gradient PN {dpn / eps:+.2e}, Re {dre / eps:+.2e}" if eps > 0 else "")
    )
c = res.get(0.0)
if c is None or abs(c["PN"] - stored["PN"]) > 1e-12 or abs(c["Re"] - stored["Re"]) > 1e-12:
    print("READING q1485: VOID (control eps = 0 does not reproduce the stored row to 1e-12)")
else:
    e = res.get(1e-10)
    ins = e is not None and abs(e["PN"] - stored["PN"]) < 1e-12 and abs(e["Re"] - stored["Re"]) < 1e-12
    print(
        "READING q1485: "
        + (
            "INSENSITIVE (the re-solve removes a starting error of 1e-10 in the positions)"
            if ins
            else "SENSITIVE or not decided: see the gradients above"
        )
    )
