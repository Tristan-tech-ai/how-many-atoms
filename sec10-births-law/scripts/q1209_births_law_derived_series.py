"""q1209 (8 Oct 2026): the births law with the DERIVED centre series to higher order and no immune input: 2 N_half(a) = K_b + 1/6 + sum_{n=2}^{NMAX} e_n / v^(2n), v = 4 pi rho, rho = dN_half/da at the event depth (convention-free density, amendment 953).
The coefficients e_2 ... e_12 are those of d22_order_n_numeric.py (d22_out_12.json; e_2 to e_5 equal the symbolic values of 1 Oct to every printed digit, e_6 = -567328.677..., e_7 = -29459489.13...).
Stages: capacity c50c D = 110 (centre_scripts/c50_capc_D110.json, pipeline of q1183 with the stage atoms, cut 20, window 4 to 30), NPMLE c33_npmc_D110 (spline of q645, atoms with depth <= 90).
Modes: --read : residuals A_meas - a_pred for capacity K_b 61 to 80 and NPMLE K_b 54 to 66 (located, reading) for NMAX = 5, 6, 7, 8, 10, 12; --predict : the predictions (no measurement read) for capacity K_b 81 to 114 and NPMLE K_b 67 to 100 with NMAX = 12, with sha256.
Environment (predict): E_FILE (coefficient json, default d22_out_12.json), NPRED (order, default 12), OUT_PRED (default q1209_predictions.json).
Usage: py q1209_births_law_derived_series.py --read | --predict   (from temporal_within/centre_scripts/)"""

import hashlib, json, os, re, sys
import numpy as np
from scipy.optimize import brentq, least_squares

MODE = sys.argv[1] if len(sys.argv) > 1 else "--read"
sys.argv = ["x"]
import q1201_births_law_table_shared_centre as T
import q1205_npmle_shared_centre_pred as P5

EF = os.environ.get("E_FILE", "d22_out_12.json")
NPRED = int(os.environ.get("NPRED", "12"))
OUTP = os.environ.get("OUT_PRED", "q1209_predictions.json")
E = {int(k[1:]): float(v) for k, v in json.load(open(EF)).items() if re.fullmatch(r"e\d+", k) and k != "e1"}
pi = np.pi


def delta(rho, nmax):
    v = 4 * pi * rho
    return sum(E[n] / v ** (2 * n) for n in range(2, nmax + 1))


H = 1e-3


def pred_series(N, K, nmax, lo=30.0, hi=140.0):
    f = lambda a: 2 * N(a) - K - 1 / 6 - delta((N(a + H) - N(a - H)) / (2 * H), nmax)
    return brentq(f, lo, hi, xtol=1e-14)


os.chdir("..")
sys.path.insert(0, ".")
import q1183_births_residual_halfline_depth_reading as Q

os.chdir("centre_scripts")
# capacity N_half: spline of the c50c stage D = 110 as in q1183 (law fit window 4 to 30, knots to D - 20)
sc = [s for s in json.load(open("c50_capc_D110.json"))["stages"] if s["D"] == 110.0][0]
dc = np.array(sc["d"][: 1 + sc["nfree"]])
jj = np.arange(1, len(dc) + 1) - 0.5
mm = (dc >= 4) & (dc <= 30)
capl = tuple(
    least_squares(lambda q: np.array([Q.N_of(x, q[0], q[1]) for x in dc[mm]]) - jj[mm], [0.0558, 0.5482]).x
)
Ncap, _ = Q.spline(dc[dc <= 90.0], capl)
Nnp = P5.Nn  # NPMLE spline (q645 mkN) with its own offsets


def main():
    os.chdir("..")
    if MODE == "--read":
        print("coefficients e_n used:", {n: f"{E[n]:.6e}" for n in range(2, 13)})
        print(
            "test: capacity K_b 61 to 80 (measured amplitudes known), NPMLE K_b 54 to 66 and 50, 52 (located)"
        )
        for nmax in [int(x) for x in os.environ.get("NLIST", "5,6,7,8,10,12").split(",")]:
            os.chdir("centre_scripts")
            r = {K: None for K in range(61, 81)}
            os.chdir("..")
            r = {K: Q.measured(K) - pred_series(Ncap, K, nmax) for K in range(61, 81)}
            os.chdir("centre_scripts")
            q = json.load(open("q1205_predictions.json"))["pred"]
            rn = {}
            for K in list(range(54, 67)):
                f = f"q607_u1205_K{K}.json" if K != 62 else "q607_u1205r_K62.json"
                j = json.load(open(f))
                rn[K] = j["a_t"] - pred_series(Nnp, K, nmax, 20.0, 80.0)
            os.chdir("..")
            print(
                f"NMAX {nmax}: capacity A - a_pred (1e-9) K_b 61, 70, 79, 80: {r[61] * 1e9:+.3f}, {r[70] * 1e9:+.3f}, {r[79] * 1e9:+.3f}, {r[80] * 1e9:+.3f}; max |.| over 61-80 {max(abs(x) for x in r.values()) * 1e9:.3f}e-9; NPMLE K_b 54, 60, 66: {rn[54] * 1e9:+.3f}, {rn[60] * 1e9:+.3f}, {rn[66] * 1e9:+.3f}; max |.| over 54-66 {max(abs(x) for x in rn.values()) * 1e9:.3f}e-9"
            )
            if nmax == max(int(x) for x in os.environ.get("NLIST", "5,6,7,8,10,12").split(",")):
                print(
                    f"   per-event deviations at NMAX {nmax} (1e-9): capacity "
                    + ", ".join(f"{K} {v * 1e9:+.3f}" for K, v in r.items())
                    + "; NPMLE "
                    + ", ".join(f"{K} {v * 1e9:+.3f}" for K, v in rn.items())
                )
        print("END READING q1209")
    else:
        os.chdir("centre_scripts")
        cap = {str(K): pred_series(Ncap, K, NPRED) for K in range(81, 115)}
        npm = {str(K): pred_series(Nnp, K, NPRED, 20.0, 80.0) for K in range(67, 101)}
        js = {"capacity": cap, "npmle": npm}
        sh = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
        js["sha256"] = sh
        json.dump(js, open(OUTP, "w"), indent=1)
        print("capacity K_b 81, 90, 100, 114:", [round(cap[str(k)], 10) for k in (81, 90, 100, 114)])
        print("NPMLE K_b 67, 80, 100:", [round(npm[str(k)], 10) for k in (67, 80, 100)])
        print("sha256", sh)


if __name__ == "__main__":
    main()
