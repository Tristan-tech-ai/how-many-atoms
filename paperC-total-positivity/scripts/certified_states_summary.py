"""Summary of the 94 states of the computer-assisted proof (Section VI of paper C): the counts of states per number of mass points K,
the range of the number m of positive interior points (m = (K - 2)//2 for even K, (K - 3)//2 for odd K) and the range of A.
The states are read from the names of their output files, which carry K and A (sec12-total-positivity/logs/q1882_states__*.txt).
Writes certified_states.json next to this script. Usage: python certified_states_summary.py"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layout import state_names  # noqa: E402

rows = []
for f in state_names():
    m = re.search(r"_K(\d+)_A([0-9.]+?)(?:\.json|\.txt)?$", f)
    if m:
        rows.append((int(m.group(1)), float(m.group(2).rstrip(".")), f))
cnt = collections.Counter(k for k, a, f in rows)
mpos = sorted({(k - 3) // 2 if k % 2 else (k - 2) // 2 for k, a, f in rows})
out = {"files": len(rows), "K_counts": dict(sorted(cnt.items())), "positive_interior_points_min": mpos[0],
       "positive_interior_points_max": mpos[-1], "A_min": min(a for k, a, f in rows), "A_max": max(a for k, a, f in rows)}
json.dump(out, open(os.path.join(HERE, "certified_states.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
