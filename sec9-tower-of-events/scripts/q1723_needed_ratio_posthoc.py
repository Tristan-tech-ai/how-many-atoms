"""q1723 (10 Oct 2026, READING, POST HOC, no predictions; follows q1722): the ratio rho_need = x / (1 + x), x = (A - a_c) / (a_c - a_b), that would make the Aitken extrapolation hit the re-measured amplitude A exactly at each lower birth K_b 61..82, for the triples (12, 13, 14) and (13, 14, 15), against the ratio rho_model = (a_c - a_b) / (a_b - a_a) of the model predictions themselves. Written after the outputs of q1720 and q1722 were read, so nothing here is a test."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}


def parse(fn, ns):
    t = open(os.path.join(H, fn), encoding="utf-8").read()
    out = {}
    for n in ns:
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, t, re.M)
        out[n] = float(
            mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30))
        )
    return out


f = parse("q1557_stdout_ORD30.txt", (26, 28, 30))
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28], f[30]]


def setlaw(NG):
    B.F = FALL[: NG + 1]
    B.NG = NG
    B.g = [1.0]
    for n in range(1, NG + 1):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


R = json.load(open(os.path.join(H, "q1712_res.json")))
R.update(json.load(open(os.path.join(H, "q1717_res.json"))))
NEW = {int(k): float(v["new"]) for k, v in R.items() if v.get("status") == "ok"}


def meas(K):
    return B.measured(K) if K >= 83 else NEW[K]


ex = [E[n] for n in range(6, 19)]
h = 1e-7
Kin = [K for K in range(83, 131) if B.measured(K) is not None]
Klow = list(range(61, 83))
mf = np.array([B.measured(K) for K in Kin])


def run(triple):
    pred = {}
    for NG in triple:
        setlaw(NG)
        nu, d0 = -0.0937736508347, 0.0011599160063
        for rnd in range(4):
            p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Kin])
            M = np.stack(
                [
                    [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Kin],
                    [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Kin],
                ],
                1,
            )
            sol, *_ = np.linalg.lstsq(M, -(p0 - mf), rcond=None)
            nu, d0 = nu + sol[0], d0 + sol[1]
        pred[NG] = {K: B.a_pred(K, nu, d0, ex) for K in Klow + Kin}
    a, b, c = triple
    rb = float(np.median([(pred[c][K] - pred[b][K]) / (pred[b][K] - pred[a][K]) for K in Klow]))
    inf = {K: pred[c][K] + (pred[c][K] - pred[b][K]) * rb / (1 - rb) for K in Klow + Kin}
    low = max(abs(inf[K] - meas(K)) for K in Klow)
    rms = float(np.sqrt(np.mean([(inf[K] - meas(K)) ** 2 for K in Kin])))
    return rb, low, rms, inf, pred


import numpy as np

for triple in ((12, 13, 14), (13, 14, 15)):
    rb, low, rms, inf, pred = run(triple)
    a, b, c = triple
    need = []
    model = []
    for K in Klow:
        x = (meas(K) - pred[c][K]) / (pred[c][K] - pred[b][K])
        need.append(x / (1 + x))
        model.append((pred[c][K] - pred[b][K]) / (pred[b][K] - pred[a][K]))
    need = np.array(need)
    model = np.array(model)
    print(
        "triple %s: rho_model median %.4f (range %.3f to %.3f); rho_need median %.4f (range %.3f to %.3f); rho_need at K_b 61, 70, 82: %.3f, %.3f, %.3f"
        % (
            triple,
            np.median(model),
            model.min(),
            model.max(),
            np.median(need),
            need.min(),
            need.max(),
            need[0],
            need[9],
            need[-1],
        )
    )
print("READING q1723 done (post hoc; no labels)")
