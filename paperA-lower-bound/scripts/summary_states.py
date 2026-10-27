"""Summary of the stored optima and of the count changes used in Section VI and Fig. 5 of the paper: the number of states
and their verdicts, the largest Kuhn-Tucker residual, the range of K - 0.4002 (A + beta/2)^(4/3), and the ratio of the counts to the
proved bound. Reads count_law_data.json and every state file it lists; writes numbers_states.json next to this script.
Usage: python summary_states.py"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
import sys  # noqa: E402
sys.path.insert(0, HERE)
from layout import find, state  # noqa: E402
D = json.load(open(find("count_law_data.json"), encoding="utf-8"))
B2 = D["beta_half"]
C_LAW = math.floor(1.5 * (3 / (16 * math.pi ** 2)) ** (1 / 3) * 1e4 + 0.5) / 1e4
lam1 = 1.1648
rho1 = (math.e - 1) * lam1 / (1 - math.exp(-lam1))
C_LOW = math.floor(0.25 * (4 * rho1) ** (-1 / 6) * 1e4) / 1e4
res, verdicts, missing = [], {}, 0
for A, K, f in D["states"]:
    p = state(f)
    if not os.path.exists(p):
        missing += 1; continue
    d = json.load(open(p, encoding="utf-8"))
    res.append(float(d["residual"])); verdicts[d["verdict"]] = verdicts.get(d["verdict"], 0) + 1
st = [(s[0], s[1]) for s in D["states"] if s[0] >= 29.85]
diff = [K - C_LAW * (A + B2) ** (4 / 3) for A, K in st]
ratio = [(e[0] + 0.5) / (C_LOW * e[1] ** (4 / 3)) for e in D["events"]]
out = {"n_states": len(D["states"]), "n_missing": missing, "verdicts": verdicts, "max_residual": max(res),
       "n_states_ge_29.85": len(st), "diff_min": min(diff), "diff_max": max(diff),
       "n_events": len(D["events"]), "Kb_min": min(e[0] for e in D["events"]), "Kb_max": max(e[0] for e in D["events"]),
       "ratio_min": min(ratio), "ratio_max": max(ratio), "ratio_last": ratio[-1], "beta_half": B2,
       "C_LAW": C_LAW, "C_LOW": C_LOW, "C_LAW_over_C_LOW": C_LAW / C_LOW, "C_LOW_over_C_LAW_percent": 100 * C_LOW / C_LAW}
json.dump(out, open(os.path.join(HERE, "numbers_states.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out))
