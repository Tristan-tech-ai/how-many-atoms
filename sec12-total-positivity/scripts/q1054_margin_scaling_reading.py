# READING q1054: J (t* - 1), J the number of positive atoms, against J for the capacity families (q991 for K 61-66, T91 for K 67), the stored states (q988) and the immune/NPMLE rows (q1053).
# J = (K - 2) / 2 for even K (no centre atom), (K - 3) / 2 for odd K. No verdict.
import re

A = "scratch/"
Jof = lambda K: (K - 2) // 2 if K % 2 == 0 else (K - 3) // 2
txt = open(A + "q991_out.txt", encoding="utf-8").read()
print("READING q1054: J (t* - 1) of the capacity families (q991)")
for line in txt.splitlines():
    m = re.match(r"(first|middle|last) state of the family: (.*)", line)
    if m:
        vals = re.findall(r"K (\d+): t\* ([0-9.]+) \(K\(t\* - 1\) = ([0-9.]+)\)", m.group(2))
        print(
            m.group(1)
            + ": "
            + ", ".join(f"K {K} J {Jof(int(K))}: {Jof(int(K)) * (float(t) - 1):.3f}" for K, t, c in vals)
        )
t67 = {"first": 1.1253, "median": 1.1319, "last": 1.1473}
print("K 67 (T91): " + ", ".join(f"{k}: J {Jof(67)}: {Jof(67) * (t - 1):.3f}" for k, t in t67.items()))
print(
    "stored states (q988 values of 1792, state position in the family not recorded): K 26 (A 21.2821, t* 1.4771, optimal): J 12: %.3f; K 40 (A 29.85, 1.2524, optimal): J 19: %.3f; K 40 (A 30.1191, 1.2917, centre positive): J 19: %.3f; K 70 (A 46.5787, 1.1489, centre positive): J 34: %.3f; K 80 (A 51.513057, 1.1214, centre positive): J 39: %.3f"
    % (12 * 0.4771, 19 * 0.2524, 19 * 0.2917, 34 * 0.1489, 39 * 0.1214)
)
print(
    "immune rows (q1053): row 300 K 35 J 17 t* 1.2425: %.3f; row 580 K 39 J 19 t* 1.2159: %.3f"
    % (17 * 0.2425, 19 * 0.2159)
)
print(
    "NPMLE rows (q1053): row 300 K 29 J 14 t* 1.3692: %.3f; row 680 K 37 J 18 t* 1.2325: %.3f"
    % (14 * 0.3692, 18 * 0.2325)
)
print("END q1054 READING (no verdict)")
