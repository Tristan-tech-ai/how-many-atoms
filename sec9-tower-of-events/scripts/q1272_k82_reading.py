"""q1272 (8 Oct 2026, a reading, no item; amendment 2044 'reading without item'): deviation of the measured birth A_cross of K_b = 82 from the order-16 law with the cubic and the quintic spline of the capacity half-line N_half (q1239 variants A and B),.
Control (first): variant A at order 16 reproduces the table value of q1209_predictions_o16.json for K_b 82 within 1e-12, and variant B differs from A by more than 1e-12.
Usage: py q1272_k82_reading.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.optimize import brentq
import q1239_interpolation_sensitivity as Z

G = Z.G
A = json.load(open("../scratch/tower_K83_birth_out.json"))["A_cross"]


def pred(N, K, nmax):
    return brentq(
        lambda a: 2 * N(a) - K - 1 / 6 - G.delta((N(a + Z.H) - N(a - Z.H)) / (2 * Z.H), nmax),
        30.0,
        70.0,
        xtol=1e-14,
    )


NA, _ = Z.make("A")
NB, _ = Z.make("B")
t16 = json.load(open("q1209_predictions_o16.json"))["capacity"]["82"]
ok = abs(pred(NA, 82, 16) - t16) < 1e-12 and abs(pred(NB, 82, 16) - pred(NA, 82, 16)) > 1e-12
print(
    f"control: variant A order 16 {pred(NA, 82, 16):.12f} against the table {t16:.12f}; B differs from A by {abs(pred(NB, 82, 16) - pred(NA, 82, 16)):.2e}: {ok}"
)
if not ok:
    print("READING q1272: VOID (control)")
    sys.exit(2)
print(f"A_cross {A:.12f}")
for n in (12, 16):
    print(
        f"order {n}: cubic spline deviation {(A - pred(NA, 82, n)) * 1e9:+.4f}e-9, quintic spline deviation {(A - pred(NB, 82, n)) * 1e9:+.4f}e-9"
    )
print("READING q1272 (no verdict)")
