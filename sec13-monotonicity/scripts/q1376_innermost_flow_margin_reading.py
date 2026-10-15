"""q1376 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1376_innermost_flow_margin_reading.py from temporal_within/centre_scripts/): where in the family is the flow of the innermost mass smallest relative to the scale S = 1 / (2 (A + delta))? For every consecutive same-family pair of the stored capacity states (K >= 60, 1e-6 < A2 - A1 < 0.2) the margin m1 = (dp_1/dA) / S, p_1 the mass of the innermost right-half atom (the centre atom if there is one, otherwise the first interior atom), is binned by the lattice phase of the family: for K even phi = u1 / ((u3 - u1) / 2); for K odd the relaxation variable r = p_c / p_ref with p_ref the mass of the first interior atom (T27: the centre mass relaxes to the lattice value and the atom splits at r = 1). Printed: m1 by phase bin (median and minimum), the phase of the overall minimum, and the dependence of the minimum on K (K = 60, 70, 80). Expected before the run: unknown; the minimum of the margin may sit at the start of the family (just after the split or the birth) or at its end."""

import glob, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DELTA = 1.990803974


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


states = []
for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
    try:
        j = json.load(open(f))
        res = float(numstr(j["residual"]))
        K = int(j["K"])
        A = float(numstr(j["A"]))
    except Exception:
        continue
    if j.get("verdict") == "OPTIMAL" and res < 1e-20 and K >= 60:
        u = [float(numstr(x)) for x in j["u"]]
        w = [float(numstr(x)) for x in j["w"]]
        we = float(numstr(j["we"]))
        wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
        p = np.array(([wc] if wc is not None else []) + w + [we])
        ph = (wc / w[0]) if wc is not None else u[0] / ((u[2] - u[0]) / 2)
        states.append((K, wc is not None, A, p, ph))
states.sort(key=lambda t: (t[0], t[1], t[2]))
rows = []
for s1, s2 in zip(states[:-1], states[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
        dA = s2[2] - s1[2]
        Am = (s1[2] + s2[2]) / 2
        m1 = (s2[3][0] - s1[3][0]) / dA * 2 * (Am + DELTA)
        rows.append((s1[0], s1[1], (s1[4] + s2[4]) / 2, m1))
r = np.array([(a, b, c, d) for a, b, c, d in rows], dtype=float)
for centre, name, edges in (
    (0, "even K (phase phi = u1 / local spacing)", np.linspace(0, 0.5, 11)),
    (1, "odd K (relaxation r = p_c / p_first interior)", None),
):
    m = r[r[:, 1] == centre]
    if edges is None:
        edges = np.quantile(m[:, 2], np.linspace(0, 1, 11))
    print(
        f"{name}: pairs {len(m)}; margin dp_1/S: overall min {m[:, 3].min():.4f} at phase {m[np.argmin(m[:, 3]), 2]:.3f} (K {int(m[np.argmin(m[:, 3]), 0])}); median {np.median(m[:, 3]):.4f}",
        flush=True,
    )
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = m[(m[:, 2] >= lo) & (m[:, 2] <= hi)]
        if len(b):
            print(
                f"   phase {lo:.3f}-{hi:.3f}: pairs {len(b):4d} min {b[:, 3].min():+.4f} median {np.median(b[:, 3]):+.4f} max {b[:, 3].max():+.4f}",
                flush=True,
            )
for K in (60, 70, 80):
    m = r[(r[:, 0] == K) | (r[:, 0] == K + 1)]
    print(f"K {K},{K + 1}: smallest margin {m[:, 3].min():.4f}, median {np.median(m[:, 3]):.4f}")
print("READING q1376 (exploratory, no verdict)")
# part 2: the smallest margin of each family (it sits at the end of the family) against K, with an extrapolation c + a/K + b/K^2 for the even-K and the odd-K families separately
for centre, name in ((0, "even K"), (1, "odd K")):
    Ks = sorted(set(int(k) for k in r[r[:, 1] == centre][:, 0]))
    mins = []
    for K in Ks:
        m = r[(r[:, 0] == K) & (r[:, 1] == centre)]
        mins.append(m[:, 3].min())
    x = np.array(Ks, dtype=float)
    y = np.array(mins)
    c3 = np.polyfit(1 / x, y, 2)[::-1]
    c2 = np.polyfit(1 / x, y, 1)[::-1]
    print(
        f"{name}: families {len(Ks)} (K {Ks[0]}..{Ks[-1]}); smallest margin per family: "
        + ", ".join(f"{k}: {v:.4f}" for k, v in zip(Ks[::3], mins[::3])),
        flush=True,
    )
    print(
        f"   fit c + a/K: c = {c2[0]:.4f}, a = {c2[1]:.3f}, rms {np.sqrt(np.mean((np.polyval(c2[::-1], 1 / x) - y) ** 2)):.1e}; fit c + a/K + b/K^2: c = {c3[0]:.4f}, a = {c3[1]:.3f}, b = {c3[2]:.2f}, rms {np.sqrt(np.mean((np.polyval(c3[::-1], 1 / x) - y) ** 2)):.1e}",
        flush=True,
    )
print("READING q1376b (exploratory, no verdict)")
