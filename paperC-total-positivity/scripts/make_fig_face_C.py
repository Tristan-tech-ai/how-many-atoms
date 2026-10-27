"""Fig. 1 of the paper: the support of the capacity-achieving input against the amplitude A from 1.01 to 15.797, with the
count K(A) below. Every change happens at the origin: a center point is born, or the center point splits into a pair, and the points
then move outward. Data: q1382_states_to15p8.json (2504 optimal states of a continuation in A in double precision, each with its
Kuhn-Tucker residual; fields K, A, u = positive interior positions, w = their masses followed by the mass of each end point,
wc = center mass; events [A, kind, K after the event]). Checked numerically only. Writes fig_face_C.pdf, fig_face_C.png and
fig_face_C_numbers.json next to itself. Usage: python make_fig_face_C.py"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH

INK = "#1F2A36"
d = json.load(open(find("q1382_states_to15p8.json"), encoding="utf-8"))
S = sorted(d["states"], key=lambda s: s["A"])
families = []
for s in S:
    if not families or families[-1][0]["K"] != s["K"]:
        families.append([])
    families[-1].append(s)
ok = all(2 * len(s["u"]) + 2 + (1 if s["centre"] else 0) == s["K"] for s in S)
apply()
fig, (ax, ak) = plt.subplots(2, 1, figsize=(WIDTH, 3.1), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
Amax = S[-1]["A"]
ax.fill_between([0, Amax], [0, -Amax], [0, Amax], color="#F3F5F8", lw=0)
ax.plot([0, Amax], [0, Amax], color=INK, lw=1.0); ax.plot([0, Amax], [0, -Amax], color=INK, lw=1.0)
for fam in families:
    A = [s["A"] for s in fam]
    n = len(fam[0]["u"])
    for j in range(n):
        for sg in (1, -1):
            ax.plot(A, [sg * s["u"][j] for s in fam], color=OK["blue"], lw=0.9)
    if fam[0]["centre"]:
        ax.plot(A, [0.0] * len(A), color=OK["vermillion"], lw=1.3)
for a, kind, K in d["events"]:
    if kind == "birth":
        ax.plot([a], [0], "o", ms=3.0, mfc="white", mec=OK["vermillion"], mew=0.9, zorder=5)
    else:
        ax.plot([a], [0], "o", ms=3.0, color=INK, mec="none", zorder=5)
ax.set_ylim(-Amax * 1.02, Amax * 1.02); ax.set_ylabel("position of a mass point")
ax.text(2.2, 9.8, "end points $\\pm A$", fontsize=7.5, color=INK, rotation=0)
ax.annotate("center point born", xy=(4.03, 0), xytext=(1.2, -6.5), fontsize=7.5, color=OK["vermillion"],
            arrowprops=dict(arrowstyle="-", lw=0.5, color=OK["vermillion"]))
ax.annotate("center point splits", xy=(5.06, 0), xytext=(5.6, -9.2), fontsize=7.5, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
ks = [s["K"] for s in S]; As = [s["A"] for s in S]
ak.step(As, ks, where="post", color=INK, lw=1.0)
ak.set_ylim(0, 20); ak.set_yticks([2, 10, 18]); ak.set_ylabel("$K(A)$")
ak.set_xlim(0, Amax); ak.set_xlabel("amplitude $A$")
fig.savefig(os.path.join(HERE, "fig_face_C.pdf")); fig.savefig(os.path.join(HERE, "fig_face_C.png"), dpi=220)
nums = {"n_states": len(S), "A_min": S[0]["A"], "A_max": Amax, "K_min": min(ks), "K_max": max(ks), "n_events": len(d["events"]),
        "events": d["events"], "count_consistent": ok, "max_residual": max(s["res"] for s in S),
        "K_nondecreasing_on_grid": all(b >= a for a, b in zip(ks, ks[1:]))}
json.dump(nums, open(os.path.join(HERE, "fig_face_C_numbers.json"), "w"), indent=1)
print({k: v for k, v in nums.items() if k != "events"})
