"""q1149_postfamily_watcher.py (7 Oct 2026): tooling, no verdict. Polls every 30 s. When the event file that ends the family K (K = 76..79) exists, parses as JSON and has not changed for 20 s, and the family list exists,
it runs `bash post_family.sh K` once (output scratch/post_family_K<K>.log, marker post_family_K<K>.done). When K = 79 is done it runs `bash final_verdicts.sh` once
(output scratch/final_verdicts_run.log). It never writes an amendment; the outputs are read by hand afterwards. Stops after the final run or after 12 hours.
Usage: py q1149_postfamily_watcher.py"""

import json, os, subprocess, sys, time

A = "scratch"
T0 = time.time()
BASH = "C:/Program Files/Git/usr/bin/bash.exe"  # repair of 7 Oct 20:30 (amendment 1906): plain "bash" resolved to the WSL launcher in System32


def event_file(K):
    return f"{A}/tower_K{K + 1}_As.json" if K % 2 == 1 else f"{A}/tower_K{K + 1}_birth_out.json"


def settled(p):
    if not os.path.exists(p):
        return False
    try:
        json.load(open(p))
    except Exception:
        return False
    return time.time() - os.path.getmtime(p) >= 20


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


log("watcher started")
while time.time() - T0 < 12 * 3600:
    for K in (76, 77, 78, 79):
        done = f"{A}/post_family_K{K}.done"
        if os.path.exists(done):
            continue
        if settled(event_file(K)) and os.path.exists(f"{A}/tower_family_list_K{K}.txt"):
            log(f"family K {K}: event file settled, running post_family.sh")
            with open(f"{A}/post_family_K{K}.log", "w", encoding="utf-8") as fh:
                rc = subprocess.run(
                    [BASH, "post_family.sh", str(K)],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                ).returncode
            open(done, "w").write(f"exit {rc} at {time.strftime('%H:%M:%S')}\n")
            log(f"family K {K}: post_family.sh exit {rc}")
    if os.path.exists(f"{A}/post_family_K79.done") and not os.path.exists(f"{A}/final_verdicts.done"):
        log("K 79 done: running final_verdicts.sh once")
        with open(f"{A}/final_verdicts_run.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(
                [BASH, "final_verdicts.sh"],
                stdout=fh,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            ).returncode
        open(f"{A}/final_verdicts.done", "w").write(f"exit {rc} at {time.strftime('%H:%M:%S')}\n")
        log(f"final_verdicts.sh exit {rc}")
        break
    time.sleep(30)
log("watcher ends")
