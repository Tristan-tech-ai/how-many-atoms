"""q1446 reading numbers (9 Oct 2026, READING, no verdict): the post hoc fits that led to the gate q1446, from the training snapshot only."""

import json, os, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
tj = json.load(open(os.path.join(HERE, "q1446_training_rows.json")))
tab = json.load(open(os.path.join(HERE, "q1446_margin_table.json")))
rows = tj["rows"]
print(
    "training rows",
    len(rows),
    "K",
    rows[0]["K"],
    "to",
    rows[-1]["K"],
    "sha256",
    tj["sha256"][:16],
    "| table sha256",
    tab["sha256"][:16],
)


def seq(par, q):
    return sorted((r["K"], r[q]) for r in rows if r["parity"] == par)


def fit(s, p):
    K = np.array([k for k, _ in s], float)
    y = np.array([v for _, v in s])
    M = np.vstack([np.ones_like(K), K ** (-p)]).T
    return np.linalg.lstsq(M, y, rcond=None)[0]


print("free exponent, y = L + a K^-p")
for par in ("even", "odd"):
    for q in ("PN", "Re"):
        s = seq(par, q)
        K = np.array([k for k, _ in s], float)
        y = np.array([v for _, v in s])
        r = least_squares(lambda z: z[0] + z[1] * K ** (-z[2]) - y, [1.04, 2.0, 0.7])
        print(
            f"  {par} {q}: n={len(s)} K {int(K[0])}..{int(K[-1])} L={r.x[0]:.4f} p={r.x[2]:.3f} rms={np.sqrt(np.mean(r.fun ** 2)):.1e}"
        )
print("hold-out (train K<=96, test K 97..108), largest absolute error")
for par in ("even", "odd"):
    for q in ("PN", "Re"):
        s = seq(par, q)
        tr = [(k, v) for k, v in s if k <= 96]
        te = [(k, v) for k, v in s if k >= 97]
        print(
            f"  {par} {q}: n_train={len(tr)} n_test={len(te)} "
            + " ".join(
                f"p={p:.2f}: {max(abs(fit(tr, p)[0] + fit(tr, p)[1] * k ** (-p) - v) for k, v in te):.1e}"
                for p in (1 / 3, 2 / 3, 1.0, 4 / 3)
            )
        )
print("fixed exponent 2/3 on all training rows: limit L")
for par in ("even", "odd"):
    print("  " + " ".join(f"{q}: L={fit(seq(par, q), 2 / 3)[0]:.4f}" for q in ("PN", "Re")))
print(
    "bars (2 x hold-out error of the 2/3 fit):",
    {par: {q: float("%.3g" % tab["bar"][par][q]) for q in ("PN", "Re")} for par in ("even", "odd")},
)
print(
    "model difference |2/3 - null| at the target rows (units 1e-5), PN and Re:",
    {
        K: [round(abs(tab["pred"][str(K)][q] - tab["null"][str(K)][q]) * 1e5, 1) for q in ("PN", "Re")]
        for K in range(109, 115)
    },
)
print(
    "predictions 2/3:", {K: [round(tab["pred"][str(K)][q], 6) for q in ("PN", "Re")] for K in range(109, 115)}
)
r108 = [r for r in rows if r["K"] == 108][0]
print("row K = 108 (training): PN %.5f Re %.5f" % (r108["PN"], r108["Re"]))
