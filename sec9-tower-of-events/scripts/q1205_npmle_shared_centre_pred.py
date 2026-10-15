"""q1205 (8 Oct 2026, to be registered in amendment 1965): predictions of the NPMLE transition amplitudes K_b = 54 to 66 by the method of q645 / amendment 961 with the immune ratio curve interpolated by a cubic spline of ln(1 - r) (as q1201): 2 N_half(a) = K_b + 1/6 + delta_der(rho_c(a)) r(rho_c(a)),
N_half and the edge offsets from the NPMLE half-line c33_npmc_D110.json (last converged stage, atoms with depth <= 90, offsets fitted on 15 <= d <= 70), rho_c of the derived bulk law at d = a + OFF1 (OFF1 of the NPMLE). This is the third problem of the shared centre function: the immune ratio curve comes from the immune chain only.
Controls (registration run; no prediction beyond K_b 52 is printed): the predictions for the located NPMLE transitions K_b 50 and 52 (amendment 951: 36.4666031158 and 37.5740838506) are reproduced within 5e-10.
Output: q1205_predictions.json {K_b: {a_pred, rho_c}} and the sha256 of the table. Usage: py q1205_npmle_shared_centre_pred.py [--registration-run]  (from temporal_within/centre_scripts/)
"""

import hashlib, json, sys
import numpy as np

sys.argv_saved = list(sys.argv)
sys.argv = ["x"]
import q1201_births_law_table_shared_centre as T
from scipy.optimize import brentq

sn = [s for s in json.load(open("c33_npmc_D110.json"))["stages"] if s["conv"]][-1]
pn = np.array(sn["p"][: sn["nfree"]])
pn = pn[pn <= 90]
on = T.offs(pn, 15, 70, [-3.68, 1.40])
Nn = T.mkN(pn, (-0.22676, 0.14770))


def pred(K):
    return T.pred(K, Nn, on, 20.0, 80.0)


def main():
    located = {50: 36.4666031158, 52: 37.5740838506}
    ok = True
    for K, v in located.items():
        a = pred(K)
        d = abs(a - v)
        g = d < 5e-10
        ok = ok and g
        print(
            f"control: NPMLE K_b {K}: a_pred {a:.10f}, located {v:.10f}, deviation {(a - v) * 1e9:+.3f}e-9 -> {g}"
        )
    print("NPMLE offsets (OFF1, OFF2):", on)
    if not ok:
        print("READING q1205: VOID (control)")
        sys.exit(2)
    if "--registration-run" in sys.argv_saved:
        print("READING q1205: controls only")
        sys.exit(0)
    out = {}
    for K in range(54, 67):
        a = pred(K)
        rc = float(T.rinv(a + on[0]))
        out[str(K)] = {"a_pred": a, "rho_c": rc}
        print(f"  NPMLE K_b {K}: a_pred {a:.10f}  rho_c {rc:.4f}")
    sh = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    print("sha256", sh)
    json.dump({"pred": out, "sha256": sh}, open("q1205_predictions.json", "w"), indent=1)
    print("END READING q1205 (predictions only)")


if __name__ == "__main__":
    main()
