"""q1424 prediction table (9 Oct 2026; NO measured amplitude is read): the derived-series births law of q1209 (order 12, coefficients d22_out_12.json, central difference H = 1e-3, cubic spline of the counting function, atoms with depth <= 90, law fitted on the atoms of depth 4 to 30) evaluated for capacity K_b 81 to 114 on five converged stages of the c50c chain (centre_scripts/c50_capc_D140.json): D = 98, 104, 110, 116, 122. The stage D = 110 is the registered stage of amendment 1968 (control: it reproduces q1209_predictions.json). Output q1424_stage_table.json with the sha256 of its content. Usage: py centre_scripts/q1424_stage_predict.py (from temporal_within/)."""

import hashlib, json, os, re, sys

sys.path.insert(0, ".")
os.environ["E_FILE"] = "d22_out_12.json"
import numpy as np
from scipy.optimize import brentq, least_squares
import q1183_births_residual_halfline_depth_reading as Q

pi = np.pi
H = 1e-3
E = {
    int(k[1:]): float(v)
    for k, v in json.load(open("centre_scripts/d22_out_12.json")).items()
    if re.fullmatch(r"e\d+", k) and k != "e1"
}


def delta(rho, nmax=12):
    v = 4 * pi * rho
    return sum(E[n] / v ** (2 * n) for n in range(2, nmax + 1))


def build(D):
    sc = [s for s in json.load(open("centre_scripts/c50_capc_D140.json"))["stages"] if s["D"] == D][0]
    assert sc["conv"], ("stage not converged", D)
    dc = np.array(sc["d"][: 1 + sc["nfree"]])
    jj = np.arange(1, len(dc) + 1) - 0.5
    mm = (dc >= 4) & (dc <= 30)
    law = tuple(
        least_squares(
            lambda q: np.array([Q.N_of(x, q[0], q[1]) for x in dc[mm]]) - jj[mm], [0.0558, 0.5482]
        ).x
    )
    N, _ = Q.spline(dc[dc <= 90.0], law)
    return lambda K: brentq(
        lambda a: 2 * N(a) - K - 1 / 6 - delta((N(a + H) - N(a - H)) / (2 * H)), 30, 140, xtol=1e-14
    )


def main():
    out = {"stages": {}}
    for D in (98.0, 104.0, 110.0, 116.0, 122.0):
        f = build(D)
        out["stages"][str(int(D))] = {str(K): f(K) for K in range(81, 115)}
        print("stage", int(D), "done", flush=True)
    reg = json.load(open("centre_scripts/q1209_predictions.json"))["capacity"]
    dev = max(abs(out["stages"]["110"][str(K)] - reg[str(K)]) for K in range(81, 115))
    print("control: stage 110 against the registered table q1209, largest difference", dev)
    if not dev < 1e-12:
        print("VOID: stage 110 does not reproduce the registered table")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("centre_scripts/q1424_stage_table.json", "w"), indent=1)
    print("sha256", sh)


if __name__ == "__main__":
    main()
