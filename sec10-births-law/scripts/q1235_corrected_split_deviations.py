"""q1235 (8 Oct 2026, registered in amendment 2006): deviations A_meas - a_pred(order n) of the births law with the derived series for capacity K_b 61 to 80, with the corrected split amplitudes (tower_K<K_b+1>_As2.json) at odd K_b, and for K_b 81 against the registered tables.
a_pred: functions of q1209 (capacity half-line c50c D = 110, coefficients d22_out_from_log.json e_2 .. e_16), orders 5, 6, 7, 8, 10, 12, 14, 16. Births: A_cross of tower_K<K_b+1>_birth_out.json (unchanged).
Control (first): with the OLD split amplitudes the order-16 deviations of the 20 events equal those of q1209_out.txt to 0.001e-9, and the order-12 deviations at K_b 61, 70, 79, 80 equal the printed -0.094, -0.027, -0.014, -0.008 to 0.001e-9.
Items: M1 the largest |dev(16)| over the ten splits K_b 61 to 79 is below 0.03e-9; M2 the K_b 81 deviations with the corrected A_s are below 0.03e-9 in absolute value against the derived table of order 12 (q1209_predictions.json) and against the table of order 16 (q1209_predictions_o16.json); M3 the largest |dev| over the ten corrected splits increases from order 12 to 14 and from 14 to 16 by less than 0.005e-9.
Usage: py q1235_corrected_split_deviations.py  (from temporal_within/centre_scripts/)"""

import json, os, re, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import q1209_births_law_derived_series as G

A = "../scratch"
ORD = (5, 6, 7, 8, 10, 12, 14, 16)


def meas(K, new):
    if K % 2 == 0:
        return json.load(open(f"{A}/tower_K{K + 1}_birth_out.json"))["A_cross"]
    return json.load(open(f"{A}/tower_K{K + 1}_As{'2' if new else ''}.json"))["A_s"]


def devs(new):
    return {n: {K: (meas(K, new) - G.pred_series(G.Ncap, K, n)) * 1e9 for K in range(61, 81)} for n in ORD}


def main():
    old = devs(False)
    txt = open(f"{A}/q1209_out.txt", encoding="utf-8").read()
    m = re.search(r"per-event deviations at NMAX 16 \(1e-9\): capacity ([^;]*);", txt)
    ref16 = {int(k): float(v) for k, v in re.findall(r"(\d+) ([-+]\d+\.\d+)", m.group(1))}
    c1 = all(abs(old[16][K] - ref16[K]) < 0.001 + 5e-4 for K in range(61, 81))
    ref12 = {61: -0.094, 70: -0.027, 79: -0.014, 80: -0.008}
    c2 = all(abs(old[12][K] - v) < 0.001 + 5e-4 for K, v in ref12.items())
    print(
        f"control: old amplitudes reproduce q1209_out.txt: order 16 at 20 events {c1}, order 12 at four events {c2}"
    )
    if not (c1 and c2):
        print("READING q1235: VOID (control)")
        sys.exit(2)
    new = devs(True)
    for n in ORD:
        s = [K for K in range(61, 80, 2)]
        b = [K for K in range(62, 81, 2)]
        print(
            f"order {n:2d}: max |dev| splits (new) {max(abs(new[n][K]) for K in s):.3f}, splits (old) {max(abs(old[n][K]) for K in s):.3f}, births {max(abs(new[n][K]) for K in b):.3f}, all 20 (new) {max(abs(v) for v in new[n].values()):.3f} (1e-9)"
        )
    print("order 16 corrected deviations: " + ", ".join(f"{K}: {new[16][K]:+.3f}" for K in range(61, 81)))
    print("order 12 corrected deviations: " + ", ".join(f"{K}: {new[12][K]:+.3f}" for K in range(61, 81)))
    t12 = json.load(open("q1209_predictions.json"))["capacity"]["81"]
    t16 = json.load(open("q1209_predictions_o16.json"))["capacity"]["81"]
    a81n = meas(81, True)
    a81o = meas(81, False)
    t1 = json.load(open("q1201_table.json"))["a_pred"]
    t1 = t1["81"] if isinstance(t1, dict) else None
    d12, d16 = (a81n - t12) * 1e9, (a81n - t16) * 1e9
    print(
        f"K_b 81: A_s old {a81o:.12f}, new {a81n:.12f}; deviation against the order-12 table {d12:+.4f}e-9 (old {(a81o - t12) * 1e9:+.4f}), against the order-16 table {d16:+.4f}e-9 (old {(a81o - t16) * 1e9:+.4f})"
        + (
            f"; against the immune-curve table {(a81n - t1) * 1e9:+.4f}e-9 (old {(a81o - t1) * 1e9:+.4f})"
            if t1 is not None
            else ""
        )
    )
    sp = lambda n: max(abs(new[n][K]) for K in range(61, 80, 2))
    m1 = sp(16) < 0.03
    m2 = abs(d12) < 0.03 and abs(d16) < 0.03
    m3 = sp(14) - sp(12) < 0.005 and sp(16) - sp(14) < 0.005
    print(
        f"item M1 (splits at order 16 < 0.03e-9): max {sp(16):.4f} -> {m1}; item M2 (K_b 81 < 0.03e-9 against the tables of order 12 and 16): {m2}; item M3 (splits do not worsen from order 12 to 14 to 16): {sp(12):.4f}, {sp(14):.4f}, {sp(16):.4f} -> {m3}"
    )
    print("READING q1235: M1 " + str(m1) + ", M2 " + str(m2) + ", M3 " + str(m3) + " (no verdict)")


if __name__ == "__main__":
    main()
