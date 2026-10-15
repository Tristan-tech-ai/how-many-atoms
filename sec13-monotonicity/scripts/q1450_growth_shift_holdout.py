"""q1450 (9 Oct 2026, READING, no verdict; track B step B3d): hold-out test of the growth-rate-shift model of reading 4.
Model for the six normalised response entries x_j (q1448_xbd.json, index 0..5 = position-type atoms a = 2, 1, 0 and mass-type atoms a = 2, 1, 0 counted from the edge; x_2 = 1 by normalisation): x_j(K) = X_j exp(-s(K) a_j), s(K) = c K^(-2/3). Five free amplitudes X_j and c, fitted per parity to log|x_j| on the rows K <= 96, used to predict the rows K = 97 to 108 (even: 97..108 step 2, odd: 97..107). The terms are t_j = x_j bd_j with the K-independent bd_j of q1448; the edge fractions are t_j / sum of the mass-type t_j, and R_e = sum of the mass-type products over minus the sum of the position-type products. Baselines on the same training rows: (B) direct two-parameter fit R_e = L + a K^(-2/3); (C) a K^(-2/3) fit of each fraction. No row above K = 108 is used. Usage: py q1450_growth_shift_holdout.py (from small_gap_q206/).
"""

import json, numpy as np
from scipy.optimize import least_squares

e = json.load(open("q1448_xbd.json"))
ks = sorted(int(k) for k in e)
A = np.array([2, 1, 0, 2, 1, 0])


def Re_of(x, bd):
    t = x * bd
    return (
        t[3:].sum() / -t[:3].sum()
    )  # signs: position-type x > 0 and mass-type x < 0 with bd > 0 (checked below)


for par in ("even", "odd"):
    sel = [k for k in ks if e[str(k)]["parity"] == par]
    K = np.array(sel, float)
    X = np.array([e[str(k)]["x_six"] for k in sel])
    BD = np.array([e[str(k)]["bd_six"] for k in sel])
    tr = K <= 96
    te = K >= 97
    free = [0, 1, 3, 4, 5]

    def model(q, Kv):
        c = q[5]
        out = np.ones((len(Kv), 6))
        s = c * Kv ** (-2 / 3)
        for i, j in enumerate(free):
            out[:, j] = q[i] * np.exp(-s * A[j])
        out[:, 2] = 1.0
        return out

    sign = np.sign(X[0])
    q0 = np.concatenate([np.abs(X[0, free]) * 1.0, [1.66]])

    def res(q):
        return (np.log(model(q, K[tr])) - np.log(np.abs(X[tr]))).ravel()

    r = least_squares(res, q0, xtol=1e-14, ftol=1e-14)
    q = r.x
    P = model(q, K) * sign
    Re_true = np.array([Re_of(X[i], BD[i]) for i in range(len(K))])
    Re_mod = np.array([Re_of(P[i], BD[i]) for i in range(len(K))])
    # baseline B: direct fit of R_e
    M = np.vstack([np.ones_like(K), K ** (-2 / 3)]).T
    cB = np.linalg.lstsq(M[tr], Re_true[tr], rcond=None)[0]
    Re_B = M @ cB
    # baseline C: per-term fraction fits
    T = X * BD
    F = T / T[:, 3:].sum(axis=1)[:, None]
    Fp = np.zeros_like(F)
    for j in range(6):
        cj = np.linalg.lstsq(M[tr], F[tr, j], rcond=None)[0]
        Fp[:, j] = M @ cj
    Re_C = Fp[:, 3:].sum(axis=1) / -Fp[:, :3].sum(axis=1)
    print(
        f"== {par}: rows {sel[0]}..{sel[-1]}, train K<=96 n={tr.sum()}, test K 97.. n={te.sum()}; c = {q[5]:.3f}, s(108) = {q[5]*108**(-2/3):.4f}; sign check x position-type > 0: {bool((X[:, :3] > 0).all())}, mass-type < 0: {bool((X[:, 3:] < 0).all())}"
    )
    print(
        "   amplitudes X_0, X_1, Y_3, Y_4, Y_5:",
        ["%.4f" % v for v in q[:5]],
        " (limits of the p=2/3 fits of reading 4 for atoms 1, 2: 8.2191, 44.967 at K to infinity)",
    )
    print("   fit rms of log x on the training rows: %.1e" % np.sqrt(np.mean(r.fun**2)))
    print(
        "   hold-out max |error| of x_j / x_j:",
        ["%.1e" % np.max(np.abs(P[te, j] / X[te, j] - 1)) for j in range(6)],
    )
    print(
        "   hold-out max |R_e error|: model %.1e | (B) direct p=2/3 %.1e | (C) per-fraction p=2/3 %.1e"
        % (
            np.max(np.abs(Re_mod[te] - Re_true[te])),
            np.max(np.abs(Re_B[te] - Re_true[te])),
            np.max(np.abs(Re_C[te] - Re_true[te])),
        )
    )
    # extrapolated limit of the model
    Lx = q[:5]
    xinf = np.ones(6)
    for i, j in enumerate(free):
        xinf[j] = Lx[i]
    xinf = xinf * sign[None, :][0] if False else xinf * sign
    print(
        "   model limit K -> infinity: x_inf =",
        ["%+.4f" % v for v in xinf],
        " R_e(inf) = %.5f" % Re_of(xinf, BD[0]),
    )
