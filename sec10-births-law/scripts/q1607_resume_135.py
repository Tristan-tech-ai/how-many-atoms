"""q1607 (10 Oct 2026, LAUNCHER, no verdict word): resume the capacity walk of the registered block B131 from the last stored state of the odd family K = 135 in the directory of chain 134 (results_q1411_gauss_K135_A76.529022.json, the end of the chain 134 at A_END = pred(134) + fwd), with the same walker (q1411_fast_walk.py CLI) and a larger A_END = 77.40, which lies 0.011 above the registered prediction of K_b 136 (77.388989). The output directory fw_cap_par_chain135c is read by the glob of cap_cycle.sh. Blind rule: the log is never read; only the existence of event_Kb135.json and event_Kb136.json in the directory is looked at, at the chain checks (py undelivered_events.py), and the gate q1527 judges the events.
Prediction written before the start (the gate q1527 has the bars; these are about the process):
 P1 the walk reaches the split K_b 135 and writes event_Kb135.json within 90 minutes of the start;
 P2 the even family 136 is started after the split within the retry limit Q1411_SPLIT_FIRST_FAILS = 8, and the walk reaches the birth K_b 136 (event_Kb136.json) within a further 120 minutes;
 P3 the A_event of K_b 135 passes the bar of q1527 (spread 1.5e-8 at K_b 135), which is judged by the gate only.
Usage: py q1607_resume_135.py (starts one detached BelowNormal process and prints its PID)."""

import os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "py"
state = os.path.join(HERE, "fw_cap_par_chain134", "results_q1411_gauss_K135_A76.529022.json")
assert os.path.exists(state)
log = open(os.path.join(HERE, "cap_logs", "capc_resume_135c.log"), "wb")
DETACHED = 0x00000008 | 0x00000200 | 0x08000000
BELOW = 0x00004000
env = dict(os.environ, Q1411_SPLIT_FIRST_FAILS="8", PYTHONIOENCODING="utf-8")
p = subprocess.Popen(
    [PY, "q1411_fast_walk.py", state, "fw_cap_par_chain135c", "77.40"],
    cwd=HERE,
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    creationflags=DETACHED | BELOW,
)
print("started, launcher PID", p.pid)
