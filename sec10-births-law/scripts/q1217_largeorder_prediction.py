"""q1217 (8 Oct 2026, prepared under amendment 1973): blind prediction of e_19 and e_20 from the coefficients e_2 .. e_18 of the centre series (d22_out_18.json, order-18 run finished 05:29).
Model fixed before e_19 exists: r_n = e_(n+1)/e_n = alpha n + beta + gamma / n, least squares on n = 10 .. 17 (eight points). Prediction: e_19 = e_18 r_18, e_20 = e_19(pred) r_19.
Items: E1 |e_19 / e_19(pred) - 1| < 0.5 percent; E2 |e_20 / e_20(pred) - 1| < 1 percent; both read from d22_out_20.json (the run 'py d22_order_n_numeric.py 20 80', log d22_run20.txt).
--registration-run : fit and prediction (the script checks that no file d22_out_19.json or d22_out_20.json exists and that no log carries 'e_19 =').
--read             : items E1 and E2.
Usage: py q1217_largeorder_prediction.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import json, os, re, sys
import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
E = {
    int(k[1:]): float(v)
    for k, v in json.load(open("d22_out_18.json")).items()
    if k[0] == "e" and k[1:].isdigit() and k != "e1"
}
log = open("d22_run20.txt", errors="replace").read() if os.path.exists("d22_run20.txt") else ""


def fit():
    n = np.arange(10, 18)
    r = np.array([E[k + 1] / E[k] for k in n])
    A = np.vstack([n, np.ones_like(n), 1.0 / n]).T.astype(float)
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    rms = float(np.sqrt(np.mean((A @ c - r) ** 2)))
    return c, rms, r


def main():
    c, rms, r = fit()
    r18 = c[0] * 18 + c[1] + c[2] / 18
    r19 = c[0] * 19 + c[1] + c[2] / 19
    p19 = E[18] * r18
    p20 = p19 * r19
    if mode == "--registration-run":
        has = (
            os.path.exists("d22_out_19.json")
            or os.path.exists("d22_out_20.json")
            or re.search(r"e_19 = ", log) is not None
            or re.search(r"e_19 = ", open("d22_run18.txt", errors="replace").read()) is not None
        )
        print(f"control: e_19 absent from every file: {not has}")
        if has:
            print("REGISTRATION-RUN q1217: VOID (control)")
            sys.exit(2)
        print("ratios r_n = e_(n+1)/e_n, n = 10 to 17:", ", ".join(f"{x:.3f}" for x in r))
        print(
            f"fit r_n = alpha n + beta + gamma/n: alpha {c[0]:.4f}, beta {c[1]:.3f}, gamma {c[2]:.2f}, rms {rms:.4f}"
        )
        print(f"r_18 = {r18:.3f}, r_19 = {r19:.3f}; e_19 predicted {p19:.6e}; e_20 predicted {p20:.6e}")
        json.dump(
            {"c": list(map(float, c)), "e19_pred": p19, "e20_pred": p20}, open("q1217_prediction.json", "w")
        )
        print("REGISTRATION-RUN q1217: controls True")
    else:
        j = json.load(open("d22_out_20.json"))
        pr = json.load(open("q1217_prediction.json"))
        for n, bar in ((19, 0.005), (20, 0.01)):
            if f"e{n}" not in j:
                print(f"READING q1217: e{n} not in file")
                sys.exit(2)
            e = float(j[f"e{n}"])
            d = e / pr[f"e{n}_pred"] - 1
            print(
                f"e_{n} = {e:.6e}; predicted {pr[f'e{n}_pred']:.6e}; ratio - 1 = {d * 100:+.3f} percent; item E{n - 18} (< {bar * 100:.1f} percent): {abs(d) < bar}"
            )
        print("READING q1217 (no verdict)")


if __name__ == "__main__":
    main()
