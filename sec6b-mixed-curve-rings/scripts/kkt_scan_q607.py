"""kkt_scan_q607.py (14 Oct 2026, task KKTC, reading, no verdict): scan of the NPMLE optimality function for the stored interval states of
the extended test (centre_scripts/q607_u1207_K*.json, amendment 2040) that hold every atom.

Function (q607 docstring): c(y) = int_{-a}^{a} phi(x - y) / P(x) dx with P(x) = sum_j w_j phi(x - p_j), phi(t) = exp(-t^2/2), the input
symmetric (+-p_j, and a centre atom when K is odd); optimality needs c(y) <= C on [-a, a] with equality at the atoms. C is taken as the
mean of c at the atoms (the stored masses are unnormalised; the conditions are invariant under the scaling). The stored records of odd K
do not hold the centre mass (u[2 NF] is not saved by c35_chain_k.py / q607); since 14 Oct (task KKTO) odd K is scanned from the full
state written by gate_truth/resolve_q607_odd.py (gate_truth/kkt_scans/q607_u1207_K<K>_full.json), centre atom included.
Numerics: float64, Gauss-Legendre panels of width 0.25 with 24 nodes on [-a, a]; y on a grid of step 0.005 on [0, a] (symmetry) plus
every atom. Prints per file the largest c(y)/C - 1 over the grid and the spread of c/C - 1 over the atoms (a check of the quadrature),
and writes gate_truth/kkt_scans/<file>.json with key scanmax = [y, value] for paper_unified/gate_kkt_margin.py.
The record used is the last solved record before the transition (largest a below a_t), as in c20_refit.py.
Usage (from temporal_within): py gate_truth/kkt_scan_q607.py K [K ...]"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "gate_truth", "kkt_scans")


def gl_nodes(a, width=0.25, n=24):
    t, w = np.polynomial.legendre.leggauss(n)
    edges = np.arange(-a, a + 1e-12, width)
    if edges[-1] < a:
        edges = np.append(edges, a)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * w)
    return np.concatenate(xs), np.concatenate(ws)


def scan(path):
    d = json.load(open(path))
    K = d["K"]
    mc = None
    if K % 2 == 1:  # odd K: the full state re-solved by resolve_q607_odd.py (with the centre mass)
        full = os.path.join(OUT, os.path.basename(path).replace(".json", "_full.json"))
        if not os.path.exists(full):
            return None, "odd K: the stored record has no centre mass"
        r = json.load(open(full))
        mc = float(r["m0"])
    else:
        at = d.get("a_t")
        recs = [h for h in d["hist"] if "p" in h and (at is None or h["a"] < at)]
        r = max(recs, key=lambda h: h["a"])
    a = float(r["a"])
    p = np.array(r["p"], float)
    m = np.array(r["m"], float)
    if len(p) != K // 2:
        return None, "record holds %d atoms per side, K %d" % (len(p), K)
    P_atoms = np.concatenate([p, -p] + ([np.array([0.0])] if mc is not None else []))
    W = np.concatenate([m, m] + ([np.array([mc])] if mc is not None else []))
    if mc is not None:
        p = np.concatenate([p, [0.0]])
    x, wx = gl_nodes(a)
    P = (W[None, :] * np.exp(-0.5 * (x[:, None] - P_atoms[None, :]) ** 2)).sum(1)
    g = wx / P

    def c(y):
        y = np.atleast_1d(y)
        out = np.empty(len(y))
        for i in range(0, len(y), 2000):
            yy = y[i : i + 2000]
            out[i : i + 2000] = (np.exp(-0.5 * (x[None, :] - yy[:, None]) ** 2) * g[None, :]).sum(1)
        return out

    ca = c(p)
    C = ca.mean()
    spread = float(np.max(np.abs(ca / C - 1)))
    ys = np.concatenate([np.arange(0.0, a + 1e-12, 0.005), p])
    v = c(ys) / C - 1
    k = int(np.argmax(v))
    return (float(ys[k]), float(v[k]), spread, a), None


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    for k in argv:
        f = "centre_scripts/q607_u1207_K%s.json" % k
        res, why = scan(os.path.join(ROOT, f))
        if res is None:
            print("K %s: not scanned (%s)" % (k, why))
            continue
        y, v, spread, a = res
        print(
            "K %s (a %.4f): largest c/C - 1 on [0, a] %+.3e at y %.3f; spread of c/C - 1 over the atoms %.1e"
            % (k, a, v, y, spread)
        )
        json.dump(
            {
                "file": f,
                "scanmax": [y, v],
                "atom_spread": spread,
                "a": a,
                "by": "gate_truth/kkt_scan_q607.py",
            },
            open(os.path.join(OUT, os.path.basename(f)), "w"),
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
