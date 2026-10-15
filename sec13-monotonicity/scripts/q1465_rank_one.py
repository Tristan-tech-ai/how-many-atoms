"""q1465 (9 Oct 2026, READING, no verdict): is the K dependence of the six edge fractions ONE-DIMENSIONAL? From q1463_terms.json (rows K = 83..114, parities apart) form the matrix F[j, K] of the six fractions f_j = t_j / sum of mass-type t (t_j = x_j bd_j), subtract the mean over K, and print the singular values (relative to the first) and the direction alpha (first left singular vector, scaled so that the mass-type entries have the sign pattern of fraction 5 positive). Also: the same for the log of the five normalised response entries, and the comparison of the scalar factor phi(K) of the even and odd rows after mapping with a shared direction. Usage: py q1465_rank_one.py (from small_gap_q206/)."""

import json, numpy as np

d = json.load(open("q1463_terms.json"))
ks = sorted(int(k) for k in d)


def mats(par):
    sel = [k for k in ks if d[str(k)]["parity"] == par]
    X = np.array([d[str(k)]["x_six"] for k in sel])
    BD = np.array([d[str(k)]["bd_six"] for k in sel])
    T = X * BD
    F = T / T[:, 3:].sum(axis=1)[:, None]
    return sel, F, np.log(np.abs(X[:, [0, 1, 3, 4, 5]]))


res = {}
for par in ("even", "odd"):
    sel, F, LX = mats(par)
    res[par] = (sel, F, LX)
    for name, M in (("six edge fractions", F), ("ln |x| of the five free entries", LX)):
        C = M - M.mean(axis=0)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        print(
            f"== {par}: {name}: n = {len(sel)} rows; singular values relative to the first: "
            + ", ".join(f"{s / S[0]:.2e}" for s in S[:5])
        )
        if name.startswith("six"):
            al = Vt[0] * np.sign(Vt[0][5])
            print("   direction alpha (unit vector over the six fractions):", np.round(al, 5).tolist())
# shared direction: project the odd rows on the even direction
selE, FE, _ = res["even"]
selO, FO, _ = res["odd"]
CE = FE - FE.mean(axis=0)
UE, SE, VE = np.linalg.svd(CE, full_matrices=False)
aE = VE[0]
CO = FO - FO.mean(axis=0)
UO, SO, VO = np.linalg.svd(CO, full_matrices=False)
aO = VO[0]
print("cosine between the even and the odd direction:", float(abs(aE @ aO)))
resid = CO - np.outer(CO @ aE, aE)
print(
    "relative residual of the odd rows outside the even direction:",
    float(np.linalg.norm(resid) / np.linalg.norm(CO)),
)
