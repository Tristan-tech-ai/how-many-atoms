"""q1611 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 65): the births law with the self-consistent edge pair of q1610.
q1610 iterated the edge pair on the exact clean block N = 90 (atoms k = 30..60 against the formal law through c20) and found the pair (nu0, d0) = (-0.093773648256, +0.001159913641). It differs from the Aitken limit pair of reading 74 (-0.093773592, +0.001159849) by (-5.626e-08, +6.464e-08). The births law of q1564 (law through c20 at the centre, offs(nu, d0) over the 48 measured births K_b 83..130) gave for the limit pair a mean offset of -2.013e-09, a spread of 2.30e-08 and a largest |offset| of 1.241e-08.
Estimate written before the run: a shift of the pair moves the births by about delta A = -delta N / (dN/dA) with delta N = -delta nu - (dN/dA) delta d0 and dN/dA about 1.15 (from the table of K against A). With (delta nu, delta d0) = (-5.626e-08, +6.464e-08) this gives delta N = -1.8e-08 and delta A = +1.5e-08. The scale checks against q1564: the pair (6, 7, 8) differs from the pair (8, 9, 10) by (-1.42e-07, +1.78e-07), the same formula gives +5.5e-08 and the measured mean offset moved by +4.6e-08, so the estimate is taken at 0.85 of the formula, +1.3e-08.
Predictions written before the run:
 P1 the mean offset of the births law with the q1610 pair lies between +5e-09 and +1.8e-08;
 P2 the spread (largest minus smallest offset) with the q1610 pair is within 20 percent of 2.30e-08 (between 1.84e-08 and 2.76e-08);
 P3 the largest |offset| with the q1610 pair lies between 1.5e-08 and 3.5e-08, that is, above the 1.241e-08 of the limit pair (the block pair does not reproduce the births better than the births-fitted limit pair);
 P4 positive control: the offsets for the limit pair (8, 9, 10) computed in this script reproduce the q1564 numbers (mean -2.013e-09, largest 1.241e-08) to 1e-11 in the mean and the largest value.
Usage: py q1611_births_new_pair.py (about 60 s; executes q1564 silently for its functions and the limit pair).
"""

import io, contextlib, os, json
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(H, "q1564_orders_9_10.py"), "__name__": "q1564ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(H, "q1564_orders_9_10.py"), encoding="utf-8").read(), ns)
offs, lim, Ks = ns["offs"], ns["lim"], ns["Ks"]
new = json.load(open(os.path.join(H, "q1610_res.json")))["final_pair"]
nu_new, d0_new = float(new[0]), float(new[1])
lp = lim[(8, 9, 10)]
print(
    f"limit pair (8, 9, 10): ({float(lp[0]):+.9f}, {float(lp[1]):+.9f}); q1610 pair: ({nu_new:+.9f}, {d0_new:+.9f}); difference ({nu_new - float(lp[0]):+.3e}, {d0_new - float(lp[1]):+.3e})"
)
r_lim = offs(float(lp[0]), float(lp[1]))
r_new = offs(nu_new, d0_new)
r_nu = offs(float(lp[0]) + 1e-7, float(lp[1]))
r_d0 = offs(float(lp[0]), float(lp[1]) + 1e-7)
print(
    f"sensitivity of the mean offset: d/d nu0 {(r_nu.mean() - r_lim.mean()) / 1e-7:+.3f}, d/d d0 {(r_d0.mean() - r_lim.mean()) / 1e-7:+.3f}"
)
for name, r in (("limit pair", r_lim), ("q1610 pair", r_new)):
    k = Ks[int(np.abs(r).argmax())]
    print(
        f"{name}: mean offset {r.mean():+.3e}, spread {r.max() - r.min():.3e}, largest |.| {np.abs(r).max():.3e} at K_b {k}, {len(r)} births"
    )
p1 = 5e-9 <= r_new.mean() <= 1.8e-8
sp = r_new.max() - r_new.min()
p2 = 1.84e-8 <= sp <= 2.76e-8
mx = np.abs(r_new).max()
p3 = 1.5e-8 <= mx <= 3.5e-8
p4 = abs(r_lim.mean() - (-2.013e-9)) < 1e-11 and abs(np.abs(r_lim).max() - 1.241e-8) < 1e-11
lab = lambda p: "holds" if p else "does not hold"
print(
    f"P1 mean offset in [+5e-09, +1.8e-08]: {lab(p1)}; P2 spread within 20 percent of 2.30e-08: {lab(p2)}; P3 largest |offset| in [1.5e-08, 3.5e-08]: {lab(p3)}; P4 control reproduces q1564: {lab(p4)}"
)
print("READING q1611 done (labels by the script; no verdict word)")
