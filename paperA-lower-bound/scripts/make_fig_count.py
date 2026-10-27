"""Fig. 5 of the paper: the computed counts against the proved lower bound. Dots: the amplitudes A at which the computed
number of mass points changes from K_b to K_b + 1, drawn at K = K_b + 1/2 on log-log axes. Solid line: 0.4002 (A + beta/2)^(4/3),
with beta/2 from the same file. Dashed line: the asymptotic lower bound 0.1660 A^(4/3) of Theorem 1. Data: count_law_data.json.
0.4002 = (3/2)(3/(16 pi^2))^(1/3) rounded to four digits; 0.1660 = (1/4)(4 rho_1)^(-1/6) cut to four digits.
Writes fig_count.pdf and fig_count.png next to itself and prints the numbers of the caption. Usage: python make_fig_count.py"""
import json
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH

D = json.load(open(find("count_law_data.json"), encoding="utf-8"))
B2 = D["beta_half"]
C_LAW = math.floor(1.5 * (3 / (16 * math.pi ** 2)) ** (1 / 3) * 1e4 + 0.5) / 1e4
lam1 = 1.1648
rho1 = (math.e - 1) * lam1 / (1 - math.exp(-lam1))
C_LOW = math.floor(0.25 * (4 * rho1) ** (-1 / 6) * 1e4) / 1e4
Kb = [e[0] for e in D["events"]]
Aev = [e[1] for e in D["events"]]
ratio = [(k + 0.5) / (C_LOW * a ** (4 / 3)) for k, a in zip(Kb, Aev)]

apply()
fig, ax = plt.subplots(1, 1, figsize=(WIDTH, 2.5))
xs = [Aev[0] * (Aev[-1] / Aev[0]) ** (i / 200) for i in range(201)]
# direct labels instead of a legend box; the band between the two curves shows the room left between them
law = [C_LAW * (x + B2) ** (4 / 3) for x in xs]
low = [C_LOW * x ** (4 / 3) for x in xs]
ax.fill_between(xs, low, law, color="#E9EDF2", lw=0)
ax.loglog(xs, law, "-", lw=0.9, color="#1F2A36")
ax.loglog(xs, low, "--", lw=1.0, color=OK["vermillion"])
ax.loglog(Aev, [k + 0.5 for k in Kb], "o", ms=2.0, color=OK["blue"], mec="none")
ax.text(3.0, 60, "computed counts (dots)", fontsize=7.5, color=OK["blue"], ha="left")
ax.text(3.0, 40, "line: $%.4f\\,(A+\\beta/2)^{4/3}$" % C_LAW, fontsize=7.5, color="#1F2A36", ha="left")
ax.text(20, 8.5, "proved: $%.4f\\,A^{4/3}$" % C_LOW, fontsize=7.5, color=OK["vermillion"], ha="left", va="top")
for axis, ticks in ((ax.xaxis, [2, 5, 10, 20, 50]), (ax.yaxis, [2, 5, 10, 20, 50, 100])):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(ScalarFormatter()); axis.set_minor_locator(NullLocator())
ax.set_xlabel("amplitude $A$")
ax.set_ylabel("number of mass points $K$")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_count.pdf")); fig.savefig(os.path.join(HERE, "fig_count.png"), dpi=200)
print("fig_count written: events %d, K_b %d to %d, A %.6f to %.6f; constants %.4f %.4f; beta/2 %.11f; ratio (K_b+1/2)/(%.4f A^(4/3)) %.3f to %.3f (last %.3f)"
      % (len(Kb), Kb[0], Kb[-1], Aev[0], Aev[-1], C_LAW, C_LOW, B2, C_LOW, min(ratio), max(ratio), ratio[-1]))
