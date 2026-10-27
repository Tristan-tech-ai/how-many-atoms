"""Position of each of the 94 states of the computer-assisted proof within its family (Section VI of paper C).
A family with K points runs between the amplitudes of the events K_b = K - 1 and K_b = K of count_law_data.json (field "events":
[K_b, A, source]). A state at amplitude A of family K has the relative position (A - A_start)/(A_end - A_start); a value above 1
marks a stationary state beyond the end of its family. Writes certified_positions.json next to this script.
Usage: python certified_positions.py"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layout import find, state_names  # noqa: E402

ev = {int(k): float(a) for k, a, *_ in json.load(open(find("count_law_data.json")))["events"]}
rows = []
for name in state_names():
    m = re.search(r"_K(\d+)_A([0-9.]+?)(?:\.json|\.txt)?$", name)
    if not m:
        continue
    K, A = int(m.group(1)), float(m.group(2).rstrip("."))
    a0, a1 = ev.get(K - 1), ev.get(K)
    rel = (A - a0) / (a1 - a0) if a0 is not None and a1 is not None else None
    rows.append({"K": K, "A": A, "start": a0, "end": a1, "rel": rel, "dist_to_end": (a1 - A) if a1 is not None else None})
opt = [r for r in rows if r["rel"] is not None and r["rel"] <= 1]
beyond = [r for r in rows if r["rel"] is not None and r["rel"] > 1]
out = {"states": len(rows), "inside_family": len(opt), "beyond_end": [(r["K"], r["A"], round(r["rel"], 4)) for r in beyond],
       "rel_min": min(r["rel"] for r in opt), "rel_max": max(r["rel"] for r in opt),
       "smallest_distance_to_end": min(r["dist_to_end"] for r in opt),
       "states_in_last_tenth": sum(1 for r in opt if r["rel"] >= 0.9), "rows": rows}
json.dump(out, open(os.path.join(HERE, "certified_positions.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
