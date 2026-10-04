"""Write bindings_paper1_fail.json: bindings for the paper 1 numbers that failed the second gate pass (4 Oct), each recomputed
from the registration log or a primary output (births_check.json and inner_rings_check.json are built from the log only).
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
G = "temporal_within/gate_truth/"
BC, IR = G + "births_check.json", G + "inner_rings_check.json"
BO = "temporal_within/agents/B_out/"
out = []


def B(printed, ctx, expr, why, kind=None):
    d = {"paper": "paper1", "printed": printed, "context": ctx, "expr": expr, "why": why}
    if kind:
        d["kind"] = kind
    out.append(d)


BIRTH = f"3e-7 if J('{BC}', 'max_abs_miss') <= 3e-7 else -1"
WB = "upper bound: the six registered births miss by at most 2.91e-7 in r_m (births_check.json, ring zeros from Amendments 1406, 1443, 1447, 1452, 1462, 1464)"
B("3\\times 10^{-7}", "point held to $3\\times 10^{-7}$ in all three", BIRTH, WB)
B("3\\times 10^{-7}", "All six registered births passed, within $3\\times 10^{-7}$", BIRTH, WB)
B("3\\times 10^{-7}", "within $3\\times 10^{-7}$ of the prediction", BIRTH, WB)

# copying error h_{K-2}: solved ring (b from Amendment 1283, c from the full-density file) against the linear response (cc_cap13K*.log)
SOL12 = f"(J('{BO}cc13K12_M6_e0.182742.json', 'c', 4) - L(1283, r'K 12 at eps 0\\.182742: b (?:[\\d.]+, ){{4}}([\\d.]+) \\(j 2-10\\)'))"
SOL10 = f"(J('{BO}cc13K10_M5_e0.20724.json', 'c', 3) - L(1283, r'K 10 at eps 0\\.207240: b (?:[\\d.]+, ){{3}}([\\d.]+) \\(j 2-8\\)'))"
LR12 = f"T('{BO}cc_cap13K12.log', r'LR b_j - c_j\\^\\(K/2\\):\\s+(?:[-\\d.e]+, ){{4}}(-[\\d.e-]+),')"
LR10 = f"T('{BO}cc_cap13K10.log', r'LR b_j - c_j\\^\\(K/2\\):\\s+(?:[-\\d.e]+, ){{3}}(-[\\d.e-]+),')"
REL = f"100 * max(abs(-{SOL12} - {LR12}) / abs({LR12}), abs(-{SOL10} - {LR10}) / abs({LR10}))"
B("2.2", "copying error $h_{K-2}$ to within $2.2\\%$", f"2.2 if {REL} <= 2.2 else -1",
  "upper bound: |h_{K-2}(solved) - h_{K-2}(LR)| / |h_{K-2}(LR)| is 2.17 % (K = 12) and 1.44 % (K = 10)")
# |h_j| for j <= 8 on rp 1.3, K = 12
HJ = ", ".join(f"abs(L(1283, r'K 12 at eps 0\\.182742: b (?:[\\d.]+, ){{{i}}}([\\d.]+),') - J('{BO}cc13K12_M6_e0.182742.json', 'c', {i}))" for i in range(4))
B("4\\times 10^{-6}", "are at most $4\\times 10^{-6}$ for", f"4e-6 if max({HJ}) <= 4e-6 else -1",
  "upper bound: |b_j - c_j| for j = 2, 4, 6, 8 is 0.9e-6, 2.3e-6, 3.7e-6, 1.6e-6 (b to six digits in Amendment 1283)")
B("20", "certified box rounded to 20 decimals", "20", "format of the table cells (each cell matches the polished 40-digit state rounded to 20 decimals)", "definition")
# first LFP event in d = 8 against the shift law (d - 1)(alpha + beta d) fixed by d = 2, 3 (Amendment 1573 values, d8_e1.log)
D1 = "L(1573, r'reproduces d 1 e1 ([\\d.]+)')"
D2 = "L(1573, r'd 2 e1\\s+([\\d.]+)')"
D3 = "L(1573, r'd 3 e1\\s+([\\d.]+)')"
D8 = "T('temporal_within/centre_scripts/q868_runs/d8_e1.log', r'd 8 e1 \\(birth of the centre\\): m ([\\d.]+)')"
SH8 = f"({D8} - {D1})"
PR8 = f"(7 * (3 * ({D3} - {D1}) - 5 * ({D2} - {D1})))"
B("2.138", "its shift is $2.138$ against", SH8, "d 8 e1 3.19499053 (d8_e1.log) minus d 1 e1 1.05674351 (Amendment 1573)")
B("1.137", "against $1.137$ from", PR8, "shift_8 = 7 (3 s_3 - 5 s_2) from the d = 2, 3 e1 shifts of Amendment 1573")
B("88", "misses by $88\\%$: its shift", f"100 * ({SH8} - {PR8}) / {PR8}", "relative miss of the d = 8 e1 shift")
# inner rings
B("16", "16 of 19 registered; three failures", f"J('{IR}', 'passed')", "verdicts recomputed from band and observed value for each registered inner-ring test (inner_rings_check.json; every number found in the log)")
B("19", "16 of 19 registered; three failures", f"J('{IR}', 'registered')", "registered inner-ring tests (inner_rings_check.json)")
# shape term resolution: half-width of the difference of two brackets over the shape signal (Amendments 1511-1514)
DLO = "L(1513, r'p 2 \\(disc control\\): final bracket \\[([\\d.]+),')"
DHI = "L(1513, r'p 2 \\(disc control\\): final bracket \\[[\\d.]+, ([\\d.]+)\\]')"
P3LO = "L(1514, r'settings identical to 1513: final bracket \\[([\\d.]+),')"
P3HI = "L(1514, r'settings identical to 1513: final bracket \\[[\\d.]+, ([\\d.]+)\\]')"
S3 = "(L(1512, r'predicted C ([\\d.]+); equal-area disc') - L(1512, r'equal-area disc \\(area-only alternative\\) ([\\d.]+)\\.'))"
B("2.3", "(resolution about $6\\%$, $2.3\\%$)", f"100 * (({P3HI} - {P3LO}) + ({DHI} - {DLO})) / 2 / {S3}",
  "p = 3: (bracket width at p 3 + disc bracket width)/2 over the shape signal 3.567e-3 = 2.31 % (Amendments 1512-1514)")
# numbers changed after the agent's pass (strict bounds); expressions taken from the agent's checked bindings by context
AG = json.load(open(os.path.join(HERE, "bindings_paper1_weak.json"), encoding="utf-8"))
ag = lambda ctx: next(x["expr"] for x in AG if x["context"] == ctx)
B("0.53", "first five ring losses on cooling lay within $0.53", f"0.53 if {ag('Its first five ring losses on cooling lay within')} <= 0.53 else -1",
  "upper bound over the five DA events counting the independent brackets; the largest bracket end is +0.5214 % (K 8, Amendment 1320)")
B("1.31", "$d = 4$ within $1.31", f"1.31 if {ag('$d = 4$ within $1.3')} <= 1.31 else -1",
  "upper bound over the NPMLE d = 4 events of q872_npmle_d4.log; largest |shift/quadratic - 1| is 1.3047 % (K_eff 2)")
B("0.0048", "missed by $0.0048$", ag("The midpoint rule held to $0.005"), "2D e1 1.53499461 against the 1D midpoint 1.53019517 (Amendments 1572, 1573)")
B("0.00012", "(after the fact), $0.00012", ag("(after the fact), $0.0001"), "2D e3 3.32182818 against the 1D midpoint 3.32195167 (Amendment 1573)")
R1462 = "\\| 1\\.4016 \\(K 18; K 20\\) \\| "
B("1.4\\times 10^{-10}", "(18 and 20 atoms to $1.4\\times 10^{-10}$ on the mixed curve)",
  f"1.4e-10 if abs(L(1462, r'{R1462}([-+][\\d.e-]+);') - L(1462, r'{R1462}[-+][\\d.e-]+; ([-+][\\d.e-]+)')) <= 1.4e-10 else -1",
  "upper bound: centre excess of the K 18 and K 20 rings at rm 1.4016 differs by 1.336e-10 (Amendment 1462)")
B("30", "about 30 wide in $K$ (16 ring sizes each)", ag("of about 30 ring sizes"), "mean span in K of the four complete phase runs of Amendment 1369 (182-212, 214-244, 246-276, 278-308)")
B("16", "about 30 wide in $K$ (16 ring sizes each)", "(L(1369, r'\\+1 for \\d+-(\\d+), -1 for 214') - L(1369, r'\\+1 for (\\d+)-\\d+, -1 for 214')) / 2 + 1",
  "even ring sizes in the run 182-212 of Amendment 1369")
json.dump(out, open(os.path.join(HERE, "bindings_paper1_fail.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(out), "bindings")
