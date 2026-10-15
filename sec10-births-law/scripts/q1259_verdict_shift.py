"""q1259 (8 Oct 2026, a reading, no item): shift of the first-version residuals at the split events K_b 61 to 79 when the corrected split amplitudes (tower_K<K_b+1>_As2.json) replace the first ones (tower_K<K_b+1>_As.json), and what it does to the registered quantities of tests T98 (step at K_b 71, printed 2.68e-9, band 1.4e-9 to 2.6e-9), T116 (ratio at K_b 78, printed 1.1917, band 1.08 to 1.19) and T143 (ratio at K_b 79, printed 1.2015, band 1.185 to 1.215). Draft reading 4 finding 12.
Residual r_K = A_meas - a_pred of the first version, so the corrected residual of a split is r_K - delta_K with delta_K = A_s(first) - A_s(corrected); the births are unchanged. Residuals used for the ratios: the printed list of Section 6.4 (3 digits).
Control (first): delta at K_b 81 equals 0.109e-9 (Section 6.4) within 0.002e-9.
Usage: py q1259_verdict_shift.py  (from temporal_within/centre_scripts/)"""

import json, sys

A = "../scratch"
dl = (
    lambda K: (
        json.load(open(f"{A}/tower_K{K + 1}_As.json"))["A_s"]
        - json.load(open(f"{A}/tower_K{K + 1}_As2.json"))["A_s"]
    )
    * 1e9
)
ok = abs(dl(81) - 0.109) < 0.002
print(f"control: delta at K_b 81 {dl(81):.4f}e-9 (0.109): {ok}")
if not ok:
    print("READING q1259: VOID (control)")
    sys.exit(2)
d = {K: dl(K) for K in range(61, 82, 2)}
print("delta (first - corrected) at the splits (1e-9): " + ", ".join(f"{K}: {v:+.4f}" for K, v in d.items()))
r = dict(
    zip(
        range(61, 80),
        [
            3.10,
            5.05,
            6.83,
            8.50,
            10.3,
            11.9,
            13.7,
            15.6,
            17.7,
            19.9,
            22.6,
            25.7,
            29.4,
            34.0,
            39.6,
            46.5,
            55.1,
            65.7,
            78.9,
        ],
    )
)
print(
    f"T98 step at K_b 71: printed 2.68e-9, corrected {2.68 - d[71]:.3f}e-9 (band upper edge 2.6e-9: still above {2.68 - d[71] > 2.6})"
)
print(
    f"T116 ratio at K_b 78: printed 1.1917, corrected {1.1917 * r[77] / (r[77] - d[77]):.4f} (band upper edge 1.19: still above {1.1917 * r[77] / (r[77] - d[77]) > 1.19})"
)
print(
    f"T143 ratio at K_b 79: printed 1.2015, corrected {1.2015 - d[79] / r[78]:.4f} (band 1.185 to 1.215: still inside {1.185 < 1.2015 - d[79] / r[78] < 1.215})"
)
print("READING q1259 (no verdict)")
