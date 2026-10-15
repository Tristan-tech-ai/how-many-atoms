"""q1483 (9 Oct 2026, helper, no verdict): rules R1 to R3 of amendment 2186 by script. For every event file of fw_gpu_cap (K_b >= 115 only; the folder fw_gpu_cap2 is handled by a second run) it checks the verify record of the walker (the record whose bracket matches the event file): sign_ok of both bracket ends True and zero failed Newton calls
in the whole run; then copies the file to fw_cap_par when no event for that K_b is present there (R1), and for every K_b that has both a GPU file and a chain file it prints ONLY the difference of A_event (R2, bar 1e-10). Amplitudes are never printed.
Usage: py q1483_gpu_events_transfer.py (from small_gap_q206/)."""

import json, os, glob, shutil
from mpmath import mp, mpf

mp.dps = 40
import subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, os.environ.get("Q1483_GPU_DIR", "fw_gpu_cap"))
PAR = os.path.join(
    HERE, "fw_cap_par"
)  # 10 Oct 10:5x: the variable selects the folder; without it the script also runs itself on fw_gpu_cap2 (GPU walk from K_b 138) when that folder exists
if "Q1483_GPU_DIR" not in os.environ and os.path.isdir(os.path.join(HERE, "fw_gpu_cap2")):
    print("--- folder fw_gpu_cap2 ---")
    print(
        subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, Q1483_GPU_DIR="fw_gpu_cap2"),
            capture_output=True,
            text=True,
        ).stdout,
        end="",
    )
    print("--- folder fw_gpu_cap ---")
VF = sorted(
    glob.glob(os.path.join(G, "verify*.json"))
)  # one file per walker run (a run that is restarted after a crash writes a new verify.json; resume_research.py renames the old one to verify_run<N>.json)
if not VF:
    print("no verify.json yet: nothing transferred")
    raise SystemExit(0)
RUNS = []
for vf in VF:
    try:
        v = json.load(open(vf))
        RUNS.append((v.get("newton_fail", None), v.get("events", [])))
    except Exception as e:
        print("verify file unreadable (skipped):", os.path.basename(vf), repr(e)[:60])


def key(s):
    return str(s).replace(".", "")[:17]


for f in sorted(glob.glob(os.path.join(G, "event_Kb*.json")), key=lambda x: int(os.path.basename(x)[8:-5])):
    kb = int(os.path.basename(f)[8:-5])
    if kb < 115:
        continue
    ev = json.load(open(f))
    rec = []
    fails = None
    for fl, rs in RUNS:
        m = [r for r in rs if key(r.get("A_lo", "")) == key(ev["A_lo"])]
        if m:
            rec = m
            fails = fl
            break
    ok = (fails == 0) and len(rec) >= 1 and bool(rec[0].get("sign_ok"))
    dst = os.path.join(PAR, os.path.basename(f))
    chains = [c for c in glob.glob(os.path.join(HERE, "fw_cap_par_chain*", os.path.basename(f)))]
    for c in chains:
        d = abs(mpf(str(json.load(open(c))["A_event"])) - mpf(str(ev["A_event"])))
        print(
            f"K_b {kb}: GPU route minus chain {os.path.basename(os.path.dirname(c))}: difference {mp.nstr(d, 4)} (bar 1e-10) {'OK' if d < mpf('1e-10') else 'EXCEEDS THE BAR'}"
        )
    if not ok:
        print(
            f"K_b {kb}: GPU event NOT usable (newton fails {fails}, matching verify records {len(rec)}, sign_ok {rec[0].get('sign_ok') if len(rec) >= 1 else None})"
        )
        continue
    if os.path.exists(dst):
        print(f"K_b {kb}: fw_cap_par already has the event; GPU file kept in fw_gpu_cap")
    elif chains:
        print(f"K_b {kb}: a chain event exists; not copied (R1)")
    else:
        shutil.copy(f, dst)
        print(f"K_b {kb}: copied from fw_gpu_cap to fw_cap_par (R1)")
