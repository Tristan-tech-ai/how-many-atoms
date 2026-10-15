"""build_count_law_data.py (15 Oct 2026, task FACEFIG): one data file for the count-law figure of Section III-A (figs/make_fig_law.py),
so that the figure can be redrawn from the public material. Reads only machine outputs and code:
  events  K_b = 3..59  centre_scripts/c60_*_K<K_b>.json (A_t; 128-bit full-chain relocation; no file for K_b 54, 56, 58)
          K_b = 61..82 small_gap_q206/q1717_res.json (61, 62) and q1712_res.json (63..82), field "new" (arb re-measurement)
          K_b = 83..   small_gap_q206/fw_prod/event_Kb<K_b>.json, field A_event
  states  every results_q2*_gauss_K*_A*.json with verdict OPTIMAL (the selection of small_gap_q206/q1828: folders of archives, rebuilds
          and copies skipped, one row per distinct A and K)
  beta/2  the constant B2 of small_gap_q206/q1828_stored_states_count_law_reading.py
Writes gate_truth/count_law_data.json: {"beta_half", "K_b_max", "events": [[K_b, A, source]], "states": [[A, K, source]]}; sources are
paths relative to temporal_within. Prints the counts. Usage (from temporal_within): py gate_truth/build_count_law_data.py"""
import glob
import json
import os
import re

TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(p):
    return os.path.relpath(p, TW).replace(os.sep, "/")


def inner(s):
    s = str(s)
    return s[len("np.float64("):-1] if s.startswith("np.float64(") else s


q1828 = os.path.join(TW, "small_gap_q206", "q1828_stored_states_count_law_reading.py")
B2 = float(re.search(r"^B2 = ([0-9.]+)", open(q1828, encoding="utf-8").read(), re.M).group(1))
ev = {}
for f in sorted(glob.glob(os.path.join(TW, "centre_scripts", "c60_*K*.json"))):
    m = re.search(r"_K(\d+)\.json$", f)
    if not m or int(m.group(1)) >= 60 or int(m.group(1)) in ev:
        continue
    d = json.load(open(f))
    if isinstance(d, dict) and "A_t" in d:
        ev[int(m.group(1))] = (float(d["A_t"]), rel(f))
for name in ("q1717_res.json", "q1712_res.json"):
    p = os.path.join(TW, "small_gap_q206", name)
    for k, v in json.load(open(p)).items():
        if v.get("status") == "ok":
            ev[int(k)] = (float(v["new"]), rel(p))
for f in glob.glob(os.path.join(TW, "small_gap_q206", "fw_prod", "event_Kb*.json")):
    d = json.load(open(f))
    ev[int(d["K_b"])] = (float(d["A_event"]), rel(f))
events = [[k, ev[k][0], ev[k][1]] for k in sorted(ev)]
assert all(a[1] < b[1] for a, b in zip(events, events[1:])), "event amplitudes must increase with K_b"
seen, states = set(), []
for f in glob.glob(os.path.join(TW, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
    low = f.replace("\\", "/")
    if any(s in low for s in ("/_archive/", "/_repo_rebuild/", "/how-many-atoms", "/scratchpad_archive", "/gates/")):
        continue
    try:
        d = json.load(open(f, encoding="utf-8"))
        if not isinstance(d, dict) or d.get("verdict") != "OPTIMAL":
            continue
        A, K = float(inner(d["A"])), int(float(inner(d["K"])))
    except Exception:
        continue
    if (str(d["A"]), K) in seen:
        continue
    seen.add((str(d["A"]), K))
    states.append([A, K, rel(f)])
states.sort()
out = {"beta_half": B2, "K_b_max": events[-1][0], "events": events, "states": states}
json.dump(out, open(os.path.join(TW, "gate_truth", "count_law_data.json"), "w", encoding="utf-8", newline="\n"), indent=0)
print("events %d (K_b %d to %d, missing %s); states %d (A >= 29.85: %d)" % (
    len(events), events[0][0], events[-1][0], [k for k in range(events[0][0], events[-1][0] + 1) if k not in ev], len(states),
    sum(1 for s in states if s[0] >= 29.85)))
