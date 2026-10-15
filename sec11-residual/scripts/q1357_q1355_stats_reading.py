"""q1357 (8 Oct 2026, EXPLORATORY reading after q1355, amendment 2123; no item, no verdict): size and structure of dR/Da = (R / Da) - prediction in the q1355 events, per exponent and for the two groups K_b 14-32 (new chains) and K_b 33-45 (events of q1345): number located, mean, rms, largest |dR/Da| (degree 17 and degree 19) and the K_b of the largest; the slope of dR/Da against K_b; and the same restricted to R / Da > 0.2 as a relative misfit.
Usage: py q1357_q1355_stats_reading.py  (from temporal_within/centre_scripts/, reads q1355_out.txt)"""

import re
import numpy as np

rows = {}
for l in open("q1355_out.txt", encoding="utf-8"):
    m = re.match(
        r"k ([\d.]+) K_b (\d+): v\^2 ([\d.]+), R/Da ([+-][\d.]+), predicted ([+-][\d.]+), dR/Da ([+-][\d.]+) \(degree 19 ([+-][\d.]+)\)",
        l,
    )
    if m:
        rows.setdefault(float(m.group(1)), []).append(
            (
                int(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
                float(m.group(6)),
                float(m.group(7)),
            )
        )
tot = 0
for k, r in rows.items():
    for lab, sel in (
        ("K_b 14-32", [x for x in r if x[0] <= 32]),
        ("K_b 33-45", [x for x in r if x[0] >= 33]),
        ("all", r),
    ):
        d = np.array([x[4] for x in sel])
        d19 = np.array([x[5] for x in sel])
        Kb = np.array([x[0] for x in sel])
        i = int(np.abs(d).argmax())
        rel = np.array([abs(x[4]) / abs(x[2]) for x in sel if abs(x[2]) > 0.2])
        print(
            f"k {k} {lab:9s}: located {len(sel):2d}; mean {d.mean():+.5f}, rms {np.sqrt(np.mean(d ** 2)):.5f}, largest |d| {np.abs(d).max():.4f} at K_b {Kb[i]} (degree 19: {np.abs(d19).max():.4f}); slope {np.polyfit(Kb, d, 1)[0]:+.6f} per K_b; largest relative (R/Da > 0.2) {rel.max() if len(rel) else float('nan'):.4f}"
        )
    tot += len(r)
allv = np.array([x[4] for r in rows.values() for x in r])
print(
    f"all {tot} located events: rms {np.sqrt(np.mean(allv ** 2)):.5f}, largest |dR/Da| {np.abs(allv).max():.4f}; bar 0.006 is {0.006 / np.abs(allv).max():.1f} times the largest"
)
print("READING q1357 (exploratory, no verdict)")
