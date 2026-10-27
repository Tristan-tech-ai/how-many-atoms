"""Fig. 3 of the paper: the Gram matrix N of Theorem 1 and E'NE' at one computed optimum (the last state of the family with
K = 18 in q1382_states_to15p8.json, A = 15.797). N is the Gram matrix of the derivative functions psi_p = w_p d/du[phi(y-u)+phi(y+u)]
at u_p, and psi_A for the end point, after removing their projection on the span of the mass directions chi_k = phi_k - phi_A (and
chi_0 = phi_0 - phi_A/2 for odd K), in the inner product int f g / Q (trapezoid rule, step 0.002). E' = diag((-1)^c(p)), with c(p)
the number of nodes at or below the node of p. Both matrices are drawn after the diagonal scaling N_ij / sqrt(N_ii N_jj), which
changes the sign of no minor. Double precision, checked numerically. The script first reproduces N of the worked example
(example_k4.json) to 1e-6 and stops if it does not. Writes fig_tp.pdf, fig_tp.png and fig_tp_numbers.json.
Usage: python make_fig_tp.py"""
import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH


def phi(t):
    return np.exp(-t * t / 2) / math.sqrt(2 * math.pi)


def gram_N(s):
    A = s["A"]; u = s["u"]; m = len(u); w = s["w"][:m]; we = s["w"][-1]; wc = s["wc"] if s["centre"] else None
    y = np.arange(-A - 12, A + 12 + 1e-9, 0.002); dy = 0.002
    Q = sum(wj * (phi(y - uj) + phi(y + uj)) for uj, wj in zip(u, w)) + we * (phi(y - A) + phi(y + A)) + (wc * phi(y) if wc else 0)
    dphi = lambda c: (y - c) * phi(y - c) - (y + c) * phi(y + c)
    psi = [wj * dphi(uj) for uj, wj in zip(u, w)] + [we * dphi(A)]
    phA = phi(y - A) + phi(y + A)
    chi = [phi(y - uj) + phi(y + uj) - phA for uj in u] + ([phi(y) - phA / 2] if wc else [])
    ip = lambda f, g: float(np.sum(f * g / Q) * dy)
    G = np.array([[ip(a, b) for b in psi] for a in psi])
    B = np.array([[ip(a, b) for b in chi] for a in psi])
    H = np.array([[ip(a, b) for b in chi] for a in chi])
    N = G - B @ np.linalg.solve(H, B.T)
    odd = wc is not None
    cnt = [p + 1 + (1 if odd else 0) for p in range(m)]
    cnt.append(cnt[-1] + 1)
    E = np.diag([(-1.0) ** c for c in cnt])
    return N, E @ N @ E


S = json.load(open(find("q1382_states_to15p8.json"), encoding="utf-8"))["states"]
ref = json.load(open(os.path.join(HERE, "example_k4.json")))
s4 = min((s for s in S if s["K"] == 4), key=lambda s: abs(s["A"] - 3.5))
N4, _ = gram_N(s4)
ctrl = float(np.max(np.abs(N4 - np.array(ref["N"])))) if abs(s4["A"] - 3.5) < 1e-9 else None
print(f"control at A = {s4['A']}: max |N - N(example_k4)| = {ctrl}")
if ctrl is None or ctrl > 1e-6:
    print("VOID: control failed"); sys.exit(2)
s = max((s for s in S if s["K"] == 18), key=lambda s: s["A"])
N, M = gram_N(s)
dn = np.sqrt(np.diag(N))
Nn, Mn = N / np.outer(dn, dn), M / np.outer(dn, dn)
print(f"state K {s['K']} A {s['A']}: N negative entries {int((N < 0).sum())} of {N.size}; E'NE' smallest entry {M.min():.4e}; "
      f"all entries of E'NE' positive {bool((M > 0).all())}")
cm = LinearSegmentedColormap.from_list("bwv", [OK["blue"], "#FFFFFF", OK["vermillion"]])
apply()
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 1.95), gridspec_kw={"wspace": 0.12})
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
labels = [f"$u_{j + 1}$" for j in range(len(s["u"]))] + ["$A$"]
for a, X, t in ((axs[0], Nn, "Gram matrix $N$"), (axs[1], Mn, "$E'NE'$: all entries positive")):
    a.imshow(X, cmap=cm, norm=norm)
    a.set_title(t, fontsize=8)
    a.set_xticks(range(len(labels))); a.set_yticks(range(len(labels)))
    a.set_xticklabels(labels, fontsize=6.5); a.set_yticklabels(labels, fontsize=6.5)
    a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
axs[1].set_yticklabels([])
fig.savefig(os.path.join(HERE, "fig_tp.pdf")); fig.savefig(os.path.join(HERE, "fig_tp.png"), dpi=220)
out = {"control_max_diff": ctrl, "control_A": s4["A"], "K": s["K"], "A": s["A"], "m": len(s["u"]), "order": int(N.shape[0]),
       "N_negative_entries": int((N < 0).sum()), "entries": int(N.size), "EpNEp_all_positive": bool((M > 0).all()),
       "EpNEp_min_scaled": float(Mn.min())}
json.dump(out, open(os.path.join(HERE, "fig_tp_numbers.json"), "w"), indent=1)
print(out)
