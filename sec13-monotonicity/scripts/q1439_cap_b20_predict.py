"""q1439 prediction table (9 Oct 2026; NO capacity amplitude above K_b 101 is read or exists at this point): the derived-series births law of q1209 (order 12, d22_out_12.json, H = 1e-3, cubic spline, atoms of depth <= 90, law fitted on 4 to 30) for the capacity events K_b 82 to 114 on the stages D = 122, 128 and 134 of the capacity half-line chain with the semi-free buffer B1 = 20 (centre_scripts/c50_capc_D140_B20.json, derived-law buffer, 192 bits). Control: the same code on the stage D = 110 of c50_capc_D140.json (B1 = 14) reproduces the registered table q1209_predictions.json to 1e-12. Output q1439_cap_b20_table.json (stages and the spread max(|122 - 134|, |128 - 134|) per K_b, sha256). Usage: py q1439_cap_b20_predict.py (from temporal_within/centre_scripts/ ... run as py centre_scripts/q1439_cap_b20_predict.py from temporal_within/)."""

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


def build(fname, D):
    sc = [s for s in json.load(open("centre_scripts/" + fname))["stages"] if s["D"] == D][0]
    assert sc["conv"], ("stage not converged", fname, D)
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
        lambda a: 2 * N(a) - K - 1 / 6 - delta((N(a + H) - N(a - H)) / (2 * H)), 30, 85, xtol=1e-14
    )


def main():
    reg = json.load(open("centre_scripts/q1209_predictions.json"))["capacity"]
    f = build("c50_capc_D140.json", 110.0)
    dev = max(abs(f(K) - reg[str(K)]) for K in range(82, 115))
    print("control: stage 110 (B1 = 14) against the registered table q1209, largest difference", dev)
    if not dev < 1e-12:
        print("VOID: control failed")
        sys.exit(2)
    out = {"stages": {}}
    for D in (122.0, 128.0, 134.0):
        f = build("c50_capc_D140_B20.json", D)
        out["stages"][str(int(D))] = {str(K): f(K) for K in range(82, 115)}
        print("stage", int(D), "done", flush=True)
    st = out["stages"]
    out["spread"] = {
        str(K): max(abs(st["122"][str(K)] - st["134"][str(K)]), abs(st["128"][str(K)] - st["134"][str(K)]))
        for K in range(82, 115)
    }
    print(
        "spread at K_b 82, 101, 102, 108, 114:",
        [f"{out['spread'][str(K)]:.2e}" for K in (82, 101, 102, 108, 114)],
    )
    if not out["spread"]["101"] < 5e-11:
        print("VOID: spread control failed")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("centre_scripts/q1439_cap_b20_table.json", "w"), indent=1)
    print("sha256", sh)


if __name__ == "__main__":
    main()
