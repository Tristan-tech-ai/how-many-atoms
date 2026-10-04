"""Bindings for the low-precision numbers of the inner-ring supplement (4 Oct 2026): every r_p, mixing weight and band parameter
of Table 1 is read from the amendment that registered that test. Writes bindings_supplement.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P = "supplement_inner_rings"
B = []


def b(printed, context, expr, why, kind="definition"):
    B.append(dict(paper=P, printed=printed, context=context, expr=expr, why=why, kind=kind))


# caption: band forms
b("0.15", r"fitted on capacity ($0.15$", r"L(1438, r'eccentric point plus\s+(0\.15) to 0\.35 eps\^2')", "1438 quartet band")
b("0.35", r"to $0.35\,\epsilon_i^2$", r"L(1438, r'eccentric point plus\s+0\.15 to (0\.35) eps\^2')", "1438 quartet band")
b("0.25", r"LFP $0.25\,\epsilon_i^2 \pm", r"L(1456, r'eccentric\s+point\s+plus\s+(0\.25)\s+eps\^2\s+\(the\s+capacity\s+quartet\s+residual')", "1456 LFP quartet band centre")
b(r"5\times 10^{-4}", r"LFP $0.25\,\epsilon_i^2 \pm 5\times 10^{-4}$", r"L(1456, r'half-width (5e-4); the polar point')", "1456 half-width")
b(r"5\times 10^{-4}", r"eccentric point $-5\times 10^{-4}$", r"L(1459, r'\[eccentric - (5e-4), eccentric \+ 0\.25 eps\^2 \+ 5e-4\]')", "1459 band form")
b("0.25", r"eccentric point $-5\times 10^{-4}$", r"L(1459, r'\[eccentric - 5e-4, eccentric \+ (0\.25) eps\^2 \+ 5e-4\]')", "1459 band form")
b(r"2.0\times 10^{-4}", r"of $-2.0\times 10^{-4} \pm", r"L(1468, r'miss = -(2\.0e-4) \+- 5e-5')", "1468 (c) secondary band")
b(r"5\times 10^{-5}", r"of $-2.0\times 10^{-4} \pm 5\times 10^{-5}$", r"L(1468, r'miss = -2\.0e-4 \+- (5e-5)')", "1468 (c) secondary band")
# rows: the registered problem of each test
rows = [
    ("3.6", "capacity, ellipse, 3.6 & pair", r"L(1433, r'rp (3\.6) pair loss at rm 3\.396546')"),
    ("3.6", "capacity, ellipse, 3.6 & quartet", r"L(1434, r'PRE-REGISTERED rp (3\.6) quartet at 3\.430938')"),
    ("4.3", "capacity, ellipse, 4.3 & pair", r"L(1438, r'eccentric form at rp (4\.3) \(inner')"),
    ("4.3", "capacity, ellipse, 4.3 & quartet", r"L(1438, r'eccentric form at rp (4\.3) \(inner')"),
    ("0.2", "capacity, mixed $0.2$, 3.7 & pair", r"L(1453, r'mixed curve lam (0\.2), rp 3\.7 lost')"),
    ("3.7", "capacity, mixed $0.2$, 3.7 & pair", r"L(1453, r'mixed curve lam 0\.2, rp (3\.7) lost')"),
    ("0.2", "capacity, mixed $0.2$, 3.7 & quartet", r"L(1470, r'quartet on the mixed curve lam (0\.2), rp 3\.7')"),
    ("3.7", "capacity, mixed $0.2$, 3.7 & quartet", r"L(1470, r'quartet on the mixed curve lam 0\.2, rp (3\.7)')"),
    ("2.6", "LFP, ellipse, 2.6 & pair", r"L(1444, r'LFP pair on the ellipse rp (2\.6) is lost')"),
    ("2.6", "LFP, ellipse, 2.6 & quartet", r"L(1456, r'LFP quartet on the ellipse rp (2\.6) is lost')"),
    ("0.1", "LFP, mixed $0.1$, 2.6 & pair", r"L(1466, r'FUNC lfp LAM (0\.1) rp 2\.6')"),
    ("2.6", "LFP, mixed $0.1$, 2.6 & pair", r"L(1466, r'FUNC lfp LAM 0\.1 rp (2\.6)')"),
    ("0.1", "LFP, mixed $0.1$, 2.6 & quartet", r"L(1468, r'LFP quartet, lam (0\.1), rp 2\.6')"),
    ("2.6", "LFP, mixed $0.1$, 2.6 & quartet", r"L(1468, r'LFP quartet, lam 0\.1, rp (2\.6)')"),
    ("2.8", "LFP, ellipse, 2.8 & pair", r"L(1477, r'LFP rp (2\.8): the largest')"),
    ("2.9", "NPMLE, ellipse, 2.9 & pair", r"L(1448, r'inner pair on the ellipse rp (2\.9) lost')"),
    ("2.9", "NPMLE, ellipse, 2.9 & quartet", r"L(1459, r'quartet on the ellipse rp (2\.9) is lost')"),
    ("0.2", "NPMLE, mixed $0.2$, 2.9 & pair", r"L(1466, r'FUNC\s+rd\s+LAM\s+(0\.2)\s+rp\s+2\.9')"),
    ("2.9", "NPMLE, mixed $0.2$, 2.9 & pair", r"L(1466, r'FUNC\s+rd\s+LAM\s+0\.2\s+rp\s+(2\.9)')"),
    ("0.2", "NPMLE, mixed $0.2$, 2.9 & quartet", r"L(1468, r'rd quartet, lam (0\.2), rp 2\.9')"),
    ("2.9", "NPMLE, mixed $0.2$, 2.9 & quartet", r"L(1468, r'rd quartet, lam 0\.2, rp (2\.9)')"),
    ("0.1", "NPMLE, mixed $0.1$, 2.9 & pair", r"L(1468, r'rd pair, lam (0\.1), rp 2\.9')"),
    ("2.9", "NPMLE, mixed $0.1$, 2.9 & pair", r"L(1468, r'rd pair, lam 0\.1, rp (2\.9)')"),
    ("3.1", "NPMLE, ellipse, 3.1 & pair", r"L(1477, r'rd rp (3\.1): sign change')"),
    ("3.7", "capacity, ellipse, 3.7--4.0 & 6 events", r"L(1421, r'PRE-REGISTERED at rp (3\.7)')"),
    ("4.0", "capacity, ellipse, 3.7--4.0 & 6 events", r"L(1426, r'PRE-REGISTERED at a second rp \((4\.0)\)')"),
]
for printed, ctx, expr in rows:
    b(printed, ctx, expr, "r_p or mixing weight of the registered test (the six polar events: 1421-1431, rp 3.7, 4.0, 3.85)")
json.dump(B, open(os.path.join(HERE, "bindings_supplement.json"), "w", encoding="utf-8"), indent=1)
print(len(B), "bindings")
