"""q1527 prediction table (10 Oct 2026, written after the verdict VERDICT q1451: B115 PASS; B123 PASS): the pipeline of q1451_cap_b20_predict.py unchanged (derived-series births law of q1209, order 12, stages D = 122, 128, 134 of the chain with the buffer B1 = 20, spread = max(|122 - 134|, |128 - 134|)), for the capacity events K_b 131 to 138 (block B131). Controls: stage 110 of the B1 = 14 chain reproduces the registered table q1209 for K_b 82 to 114; stage 134 reproduces the registered table q1439 for K_b 82 to 114 and the registered table q1451 for K_b 115 to 130 (differences below 1e-13); the spread at K_b 131 is within a factor 5 of the registered spread at K_b 130. No capacity amplitude above K_b 130 exists or has been read. Usage: py centre_scripts/q1527_cap_b20_predict.py (from temporal_within/; a few minutes)."""

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
    old = json.load(open("centre_scripts/q1439_cap_b20_table.json"))
    t51 = json.load(open("centre_scripts/q1451_cap_b20_table.json"))
    out = {"stages": {}}
    for D in (122.0, 128.0, 134.0):
        f = build("c50_capc_D140_B20.json", D)
        out["stages"][str(int(D))] = {str(K): f(K) for K in range(131, 139)}
        print("stage", int(D), "done", flush=True)
        if D == 134.0:
            c2 = max(abs(f(K) - old["stages"]["134"][str(K)]) for K in range(82, 115))
            print("control 2: stage 134 against the registered table q1439, largest difference", c2)
            if not c2 < 1e-13:
                print("VOID: control 2 failed")
                sys.exit(2)
            c3 = max(abs(f(K) - t51["stages"]["134"][str(K)]) for K in range(115, 131))
            print("control 3: stage 134 against the registered table q1451, largest difference", c3)
            if not c3 < 1e-13:
                print("VOID: control 3 failed")
                sys.exit(2)
    st = out["stages"]
    out["spread"] = {
        str(K): max(abs(st["122"][str(K)] - st["134"][str(K)]), abs(st["128"][str(K)] - st["134"][str(K)]))
        for K in range(131, 139)
    }
    print("spread at K_b 131, 134, 136, 138:", [f"{out['spread'][str(K)]:.2e}" for K in (131, 134, 136, 138)])
    if not out["spread"]["131"] < 5 * t51["spread"]["130"]:
        print("VOID: spread continuity control failed")
        sys.exit(2)
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = sh
    json.dump(out, open("centre_scripts/q1527_cap_b20_table.json", "w"), indent=1)
    print("sha256", sh)


if __name__ == "__main__":
    main()
