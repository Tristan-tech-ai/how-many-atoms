"""make_q1889.py (15 Oct 2026, task RLFP): writes q1889 (LFP, amendment 1461) and mix_rm_walk_lfp.py as text copies of q1888 and
mix_rm_walk.py; every replaced string is asserted. Usage (from temporal_within): py small_gap_q206/make_q1889.py
"""

src = open("small_gap_q206/q1888_mixed_rd_birth_positive_rings_verdict.py", encoding="utf-8").read()
reps = [
    (
        '"""q1888 (14 Oct 2026, task MIX301): the registered test of amendment 1463 applied to NPMLE rings on the mixed curve whose masses are all\npositive. Written before any such ring at rm <= 1.7410 exists (the LAM continuation to 0.2 at rm 1.7416 was still running).',
        '"""q1889 (15 Oct 2026, task MIXL; text copy of q1888 for the LFP): the registered test of amendment 1461 applied to LFP rings on the\nmixed curve whose masses are all positive. Written before any such ring at rm < 1.4036 on LAM 0.1 exists (the LAM continuation was\nstill running).',
    ),
    (
        "Criterion of 1463, unchanged: rd boundary rings on the mixed curve LAM 0.2, rp 2.1, s 2 (D_solver.py FUNC rd SRC_S 2 LAM 0.2, 200\nGauss-Hermite nodes) have center excess D(0) - C changing sign at rm in [1.73615, 1.73655]; FAIL otherwise.",
        "Criterion of 1461, unchanged: LFP boundary rings on the mixed curve LAM 0.1, rp 1.7 (D_solver.py FUNC lfp LAM 0.1; here 120\nGauss-Hermite nodes, 128 bits, as the stored rings of 1462) have center excess r(0) - C changing sign at rm in [1.40137, 1.40177]; FAIL\notherwise.",
    ),
    (
        "Input: the output of mix_rm_walk.py (lines 'rm R: residual X, smallest mass M, scan max S, D(0) - C E').",
        "Input: the output of mix_rm_walk_lfp.py (lines 'rm R: residual X, smallest mass M, scan max S, r(0) - C E').",
    ),
    (
        "  INCOMPLETE  a ring at 1.73655 or at 1.73615 is missing, or any ring in [1.73615, 1.73655] fails the precondition;",
        "  INCOMPLETE  a ring at 1.40177 or at 1.40137 is missing, or any ring in [1.40137, 1.40177] fails the precondition;",
    ),
    (
        "  PASS        D(0) - C at 1.73655 and at 1.73615 have opposite signs (a sign change inside the band) and no ring with rm > 1.73655 read",
        "  PASS        r(0) - C at 1.40177 and at 1.40137 have opposite signs (a sign change inside the band) and no ring with rm > 1.40177 read",
    ),
    (
        "Last line: 'VERDICT q1888: PASS|FAIL|INCOMPLETE|VOID'.",
        "Last line: 'VERDICT q1889: PASS|FAIL|INCOMPLETE|VOID'.",
    ),
    (
        "Usage (from temporal_within): py small_gap_q206/q1888_mixed_rd_birth_positive_rings_verdict.py FILE",
        "Usage (from temporal_within): py small_gap_q206/q1889_mixed_lfp_birth_positive_rings_verdict.py FILE",
    ),
    (
        r'LINE = re.compile(r"^rm ([0-9.]+): residual ([-+0-9.e]+), smallest mass ([-+0-9.e]+), scan max .*, D\(0\) - C ([-+0-9.e]+)\s*$")',
        r'LINE = re.compile(r"^rm ([0-9.]+): residual ([-+0-9.e]+), smallest mass ([-+0-9.e]+), scan max .*, r\(0\) - C ([-+0-9.e]+)\s*$")',
    ),
    ("HI, LO = 1.73655, 1.73615", "HI, LO = 1.40177, 1.40137"),
    (
        '    return "rm %s: residual %.3e, smallest mass %+.6e, scan max [9.0, 1e-13], D(0) - C %+.10e" % (rm, res, mm, ex)',
        '    return "rm %s: residual %.3e, smallest mass %+.6e, scan max [9.0, 1e-13], r(0) - C %+.10e" % (rm, res, mm, ex)',
    ),
    (
        '    good = "\\n".join([fmt("1.7380", 5e-3), fmt("1.73655", 1e-5), fmt("1.73635", 2e-7), fmt("1.73615", -1e-5)])\n    above = "\\n".join([fmt("1.7380", -5e-3), fmt("1.73655", -1e-5), fmt("1.73615", -2e-5)])\n    nochange = "\\n".join([fmt("1.73655", 1e-5), fmt("1.73615", 2e-6)])\n    negmass = "\\n".join([fmt("1.73655", 1e-5), fmt("1.73635", 1e-6, mm=-14.1), fmt("1.73615", -1e-5)])\n    missing = "\\n".join([fmt("1.73655", 1e-5)])',
        '    good = "\\n".join([fmt("1.4030", 5e-3), fmt("1.40177", 1e-5), fmt("1.40157", 2e-7), fmt("1.40137", -1e-5)])\n    above = "\\n".join([fmt("1.4030", -5e-3), fmt("1.40177", -1e-5), fmt("1.40137", -2e-5)])\n    nochange = "\\n".join([fmt("1.40177", 1e-5), fmt("1.40137", 2e-6)])\n    negmass = "\\n".join([fmt("1.40177", 1e-5), fmt("1.40157", 1e-6, mm=-14.1), fmt("1.40137", -1e-5)])\n    missing = "\\n".join([fmt("1.40177", 1e-5)])',
    ),
    (
        'print("  rm %.5f  residual %.2e  smallest mass %+.4e  D(0) - C %+.6e" % (r, res, mm, ex))',
        'print("  rm %.5f  residual %.2e  smallest mass %+.4e  r(0) - C %+.6e" % (r, res, mm, ex))',
    ),
]
for a, b in reps:
    assert src.count(a) == 1, a[:80]
    src = src.replace(a, b)
src = src.replace("VERDICT q1888", "VERDICT q1889")
open(
    "small_gap_q206/q1889_mixed_lfp_birth_positive_rings_verdict.py", "w", encoding="utf-8", newline="\n"
).write(src)

w = open("mix_rm_walk.py", encoding="utf-8").read()
wr = [
    (
        '"""mix_rm_walk.py (14 Oct 2026, task MIX301, reading, no verdict): walk the positive NPMLE ring on the mixed curve (rd, s 2, rp 2.1, LAM 0.2,\nK 20, 200 Gauss-Hermite nodes, 160 bits) from rm 1.7416 down through the band [1.73615, 1.73655] registered in amendment 1463, with',
        '"""mix_rm_walk_lfp.py (15 Oct 2026, task MIXL, reading, no verdict; text copy of mix_rm_walk.py): walk the positive LFP ring on the mixed\ncurve (lfp, rp 1.7, LAM 0.1, X1 Y1 G3, 120 Gauss-Hermite nodes, 128 bits) from rm 1.4036 down through the band [1.40137, 1.40177] of 1461, with',
    ),
    (
        'START = os.path.join(W, "D_newton_harm_rm1.7416_rdmxcont2_lam0.2000.json")',
        'START = os.path.join(W, "D_newton_harm_rm1.4036_ngh120_p128_lfpmxY_lam0.10000.json")',
    ),
    (
        'RMS = ["1.7410", "1.7404", "1.7398", "1.7392", "1.7386", "1.7380", "1.7375", "1.7370", "1.7368", "1.73665", "1.73655", "1.73645",\n       "1.73635", "1.73625", "1.73615", "1.73605"]',
        'RMS = ["1.4032", "1.4028", "1.4024", "1.4020", "1.4018", "1.40177", "1.40167", "1.40157", "1.40147", "1.40137", "1.40127"]',
    ),
    ('TAG = "_rdmxpos_K20"', 'TAG = "_lfpmxpos"'),
    (
        'env = dict(os.environ, FUNC="rd", RP="2.1", SRC_S="2", LAM="0.2000", PREC="160", NGH="200", TAG=TAG)',
        'env = dict(os.environ, FUNC="lfp", RP="1.7", SRC_S="1", LAM="0.1000", PREC="128", NGH="120", TAG=TAG)',
    ),
    (
        '    out = os.path.join(W, "D_newton_harm_rm%s%s.json" % (rm, TAG))',
        '    out = os.path.join(W, "D_newton_harm_rm%s_ngh120_p128%s.json" % (rm, TAG))',
    ),
    (
        '    c = subprocess.run([sys.executable, os.path.join(HERE, "I_rd_centre.py"), out, "200", "160"], capture_output=True, text=True, cwd=HERE)\n    m = re.search(r"D\\(0\\) - C = ([-+0-9.e]+)", c.stdout)',
        '    c = subprocess.run([sys.executable, os.path.join(HERE, "I_lfp_centre.py"), out, "120", "128"], capture_output=True, text=True, cwd=HERE)\n    m = re.search(r"r\\(0\\) - C = ([-+0-9.e]+)", c.stdout)',
    ),
    (
        '    print("rm %s: residual %.3e, smallest mass %+.6e, scan max %s, D(0) - C %s" % (rm, res, mmin, d.get("scanmax"), ex), flush=True)',
        '    print("rm %s: residual %.3e, smallest mass %+.6e, scan max %s, r(0) - C %s" % (rm, res, mmin, d.get("scanmax"), ex), flush=True)',
    ),
    ('print("END mix_rm_walk", flush=True)', 'print("END mix_rm_walk_lfp", flush=True)'),
]
for a, b in wr:
    assert w.count(a) == 1, a[:80]
    w = w.replace(a, b)
open("mix_rm_walk_lfp.py", "w", encoding="utf-8", newline="\n").write(w)
print("ok")
