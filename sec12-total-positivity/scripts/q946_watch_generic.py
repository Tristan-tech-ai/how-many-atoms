"""q946 (7 Oct 2026): generic text copy of q934 (and q927b): runs q924b_cap_structure.py (registered in 1701/1704) on every saved state results_q220_gauss_K<K>_A<A>.json of a directory as soon as it appears, up to 3 processes at a time, until the marker line appears in the trace log and nothing is pending.
Output per state: scratch/q924b_K<K>_A<A4>_out.txt (A4 = the four decimals of the file name).
Usage: py q946_watch_generic.py DIR K TRACE_LOG MARKER"""

import glob, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
DIR, K, LOG, MARK = sys.argv[1:5]
ARCH = "scratch"
running = {}


def done_trace():
    try:
        return MARK in open(LOG, encoding="utf-8", errors="replace").read()
    except Exception:
        return False


def out_of(f):
    a4 = re.search(r"_A([0-9.]+)\.json", f).group(1)
    return a4, f"{ARCH}/q924b_K{K}_A{a4}_out.txt"


while True:
    files = sorted(glob.glob(os.path.join(DIR, f"results_q220_gauss_K{K}_A*.json")))
    for f in files:
        a4, out = out_of(f)
        if os.path.exists(out) or a4 in running:
            continue
        if len([1 for p in running.values() if p.poll() is None]) >= 3:
            break
        running[a4] = subprocess.Popen(
            [sys.executable, "q924b_cap_structure.py", f],
            stdout=open(out, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
    pending = [f for f in files if not os.path.exists(out_of(f)[1])]
    if done_trace() and not pending and all(p.poll() is not None for p in running.values()):
        break
    time.sleep(10)
print("watcher done", flush=True)
