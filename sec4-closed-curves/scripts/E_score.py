"""E_score (Amendment 1328): measured near-end shortfalls of an arc state against (a) a 1D optimum at the same half-length
(q567-format JSON) and (b) the record's capacity half-line depths (centre_scripts/c50_cap_D100_norefit.json, last stage), and the
registered test: PASS if |delta_1(pred) - delta_1(meas)| <= 0.15 delta_1(meas).
Usage: py E_score.py ARC.json PRED_DELTA1_PERCENT [ONED.json]"""

import sys, json, math

S = json.load(open(sys.argv[1]))
RC = float(S["RC"])
T = float(S["TH0"]) * math.pi / 180
X = int(S["X"])
P = sorted([float(a) for a in S["P"]], reverse=True)
a = RC * T
s = [0.0] + [RC * (T - p) for p in P] + ([a] if X else [])
ga = [s[i + 1] - s[i] for i in range(4)]
H = json.load(open("../../centre_scripts/c50_cap_D100_norefit.json"))["stages"][-1]["d"]
gh = [H[i + 1] - H[i] for i in range(4)]
print(
    f"arc {sys.argv[1]}: RC {RC:g}, TH0 {T*180/math.pi:.6f}, a {a:.6f}, K {2 + X + 2*len(P)}, C {float(S['C']):.13f}"
)
print("gap  arc        half-line  delta vs half-line (%)   R^2 delta")
for j in range(4):
    print(
        f" {j + 1}   {ga[j]:.7f}  {gh[j]:.7f}  {100*(gh[j] - ga[j])/gh[j]:+.5f}               {(gh[j] - ga[j])/gh[j]*RC**2:+.4f}"
    )
d1 = None
if len(sys.argv) > 3:
    Q = json.load(open(sys.argv[3]))
    k = min(Q.keys(), key=lambda x: abs(float(x) - a))
    q = Q[k]
    A = float(k)
    d = sorted(A - x for x in q["xs"] if x > -1e-9)
    g1 = [d[i + 1] - d[i] for i in range(4)]
    print(f"1D {sys.argv[3]}: A {A:.6f} (|A - a| {abs(A - a):.1e}), K {q['K']}")
    print("gap  1D         delta vs 1D (%)   R^2 delta    1D vs half-line (rel.)")
    for j in range(4):
        print(
            f" {j + 1}   {g1[j]:.7f}  {100*(g1[j] - ga[j])/g1[j]:+.5f}          {(g1[j] - ga[j])/g1[j]*RC**2:+.4f}      {(g1[j] - gh[j])/gh[j]:+.1e}"
        )
    d1 = 100 * (g1[0] - ga[0]) / g1[0]
else:
    d1 = 100 * (gh[0] - ga[0]) / gh[0]
pr = float(sys.argv[2])
err = abs(pr - d1) / d1
print(
    f"TEST gap 1: predicted {pr:.4f} %, measured {d1:.4f} %, |pred - meas|/meas = {100*err:.2f} % -> {'PASS' if err <= 0.15 else 'FAIL'} (criterion 15 %)"
)
