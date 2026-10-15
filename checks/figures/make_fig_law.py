"""Fig. of the merged paper (Section III-A): the count law of the capacity on an interval.
(a) The amplitudes A at which the number of atoms changes from K_b to K_b + 1 (log-log), with the numerical law 0.4002 (A + beta/2)^(4/3)
    and the proved asymptotic lower bound 0.1660 A^(4/3) of Theorem 1.
(b) K - 0.4002 (A + beta/2)^(4/3) at every stored optimal state with A >= 29.85 (the states of reading 276, q1828).
The data come from gate_truth/count_law_data.json, written by gate_truth/build_count_law_data.py from the machine outputs (event files of
the tower and its re-measurement, the stored optimal states, and beta/2 of q1828); every row names its source file.
0.4002 = (3/2)(3/(16 pi^2))^(1/3) rounded to four digits as printed; 0.1660 = (1/4)(4 rho_1)^(-1/6) cut to four digits,
rho_1 = (e - 1) lambda_1 / (1 - e^(-lambda_1)), lambda_1 = 1.1648 (Theorem 1). Writes fig_law.pdf and .png next to this script."""
import json
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from figstyle import apply, OK, WIDTH


D = json.load(open(os.path.join(TW, "gate_truth", "count_law_data.json"), encoding="utf-8"))
B2 = D["beta_half"]
C_LAW = math.floor(1.5 * (3 / (16 * math.pi ** 2)) ** (1 / 3) * 1e4 + 0.5) / 1e4
lam1 = 1.1648
rho1 = (math.e - 1) * lam1 / (1 - math.exp(-lam1))
C_LOW = math.floor(0.25 * (4 * rho1) ** (-1 / 6) * 1e4) / 1e4
Kb = [e[0] for e in D["events"]]
Aev = [e[1] for e in D["events"]]
ev = set(Kb)
st = [(s[0], s[1]) for s in D["states"] if s[0] >= 29.85]
res = [K - C_LAW * (A + B2) ** (4 / 3) for A, K in st]

apply()
fig, ax = plt.subplots(2, 1, figsize=(WIDTH, 3.6), gridspec_kw={"height_ratios": [2.1, 1]})
xs = [Aev[0] * (Aev[-1] / Aev[0]) ** (i / 200) for i in range(201)]
ax[0].loglog(Aev, [k + 0.5 for k in Kb], "o", ms=2.2, color=OK["blue"], label="count changes $K_b \\to K_b+1$")
ax[0].loglog(xs, [C_LAW * (x + B2) ** (4 / 3) for x in xs], "-", lw=0.9, color=OK["black"], label="$%.4f\\,(A+\\beta/2)^{4/3}$" % C_LAW)
ax[0].loglog(xs, [C_LOW * x ** (4 / 3) for x in xs], "--", lw=0.9, color=OK["vermillion"], label="$%.4f\\,A^{4/3}$ (Theorem 1)" % C_LOW)
ax[0].set_xlabel("half-length $A$")
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator
for axis, ticks in ((ax[0].xaxis, [2, 5, 10, 20, 50]), (ax[0].yaxis, [1, 2, 5, 10, 20, 50, 100])):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(ScalarFormatter()); axis.set_minor_locator(NullLocator())
ax[0].set_ylabel("number of atoms $K$")
ax[0].legend(loc="lower right")
ax[0].text(0.03, 0.95, "(a)", transform=ax[0].transAxes, ha="left", va="top")
ax[1].plot([a for a, k in st], res, ".", ms=1.5, color=OK["blue"], label="stored optimal states")
ax[1].axhline(0.0, color="0.6", lw=0.5)
ax[1].set_xlabel("half-length $A$")
ax[1].set_ylabel("$K-%.4f\\,(A+\\beta/2)^{4/3}$" % C_LAW)
ax[1].set_ylim(-1.0, 1.0)
ax[1].text(0.03, 0.92, "(b)", transform=ax[1].transAxes, ha="left", va="top")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_law.pdf")); fig.savefig(os.path.join(HERE, "fig_law.png"), dpi=200)
print("fig_law written: events %d (K_b %d to %d, missing %s); states %d, residual %.3f to %.3f; constants %.4f %.4f beta/2 %.8f"
      % (len(Kb), Kb[0], Kb[-1], [k for k in range(Kb[0], Kb[-1] + 1) if k not in ev], len(st), min(res), max(res), C_LAW, C_LOW, B2))
