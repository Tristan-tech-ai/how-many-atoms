"""selftest_readouts.py (6 Oct 2026): does each pure-data readout of the structure tests discriminate? (6 Oct: success and failure of a claim must come from an unbiased pure-code gate, and the gate itself must be shown not to be buggy or vacuous.)
For every readout script q877, q888, q891, q892, q893, q896, q897, q898, q899, q900 the harness builds a sandbox with copies of the scripts and of the data they read, then runs them on
  clean       the stored data (positive control: the recorded verdict and the expected row counts must be reproduced, so that a silent zero-row exit 0 is caught),
  R (reverse) the physics reversed: chain rows in reverse order, LFP half-length m -> 100 - m, capacity positions moved inward with A,
  M (mass)    the masses shuffled within each row, positions intact (cannot test a quantity that depends on the row sum of the masses),
  S (scale)   every mass multiplied by (1 + 0.02 n), positions intact (changes the row sum),
  P (jitter)  the positions multiplied by (1 + 0.02 n), n standard normal, re-sorted, masses intact.
Each test states what must change. A test fails when a negative control does not move the key statistic in the stated direction (a vacuous gate) or when the clean run does not reproduce the recorded one (a changed or broken gate).
Usage: py gate_selftest/selftest_readouts.py   (from temporal_within/). Writes gate_selftest/selftest_out.txt. Exit status 1 if any test fails.
"""

import glob, json, os, random, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT_SCRIPTS = [
    "q888_O1_orientation.py",
    "q891_T27_relaxation.py",
    "q892_T28_lattice_slots.py",
    "q893_T31_mass_law.py",
    "q896_lfp_structure.py",
    "q897_lfp_cell_law.py",
    "q898_flat_density_level.py",
    "q899_capacity_structure.py",
    "q900_capacity_flows.py",
]
CS_SCRIPTS = [os.path.join("centre_scripts", "q877_A1A2_readout.py")]
CS_DATA = ["c33_immc_D146.json", "c33_npmc_D110.json", "c80_lfp_states.json"]


def corrupt_chain(d, kind, rng):
    rows = d["rows"]
    if kind == "R":
        d["rows"] = rows[::-1]
    elif kind == "M":
        for r in rows:
            v = [float(x) for x in r["v"]]
            rng.shuffle(v)
            r["v"] = v
    elif kind == "P":
        for r in rows:
            p = sorted(float(x) * (1 + 0.02 * rng.gauss(0, 1)) for x in r["p"])
            r["p"] = p
    elif kind == "S":  # scale every mass by (1 + 0.02 n): the row sum changes, a shuffle would not change it
        for r in rows:
            r["v"] = [float(x) * (1 + 0.02 * rng.gauss(0, 1)) for x in r["v"]]
    return d


def corrupt_lfp(S, kind, rng):
    for r in S:
        if kind == "R":
            r["m"] = str(100 - float(r["m"]))
        elif kind == "M":
            w = list(r["masses"])
            rng.shuffle(w)
            r["masses"] = w
        elif kind == "P":
            r["positions"] = [
                str(x) for x in sorted(float(x) * (1 + 0.02 * rng.gauss(0, 1)) for x in r["positions"])
            ]
    return S


def corrupt_cap(d, kind, rng):
    A = float(str(d["A"]).replace("np.float64(", "").rstrip(")"))
    if kind == "R":
        d["u"] = [str(float(x) * (1 - 0.05 * (A - 35.0))) for x in d["u"]]
    elif kind == "M":
        w = list(d["w"])
        rng.shuffle(w)
        d["w"] = w
    elif kind == "P":
        d["u"] = [str(x) for x in sorted(float(x) * (1 + 0.02 * rng.gauss(0, 1)) for x in d["u"])]
    return d


def build(kind):
    sb = tempfile.mkdtemp(prefix="sandbox_", dir=HERE)
    rng = random.Random(20261006)
    os.makedirs(os.path.join(sb, "centre_scripts"))
    for s in ROOT_SCRIPTS:
        shutil.copy(os.path.join(TW, s), sb)
    for s in CS_SCRIPTS:
        shutil.copy(os.path.join(TW, s), os.path.join(sb, s))
    for f in CS_DATA:
        shutil.copy(os.path.join(TW, "centre_scripts", f), os.path.join(sb, "centre_scripts", f))
    for f in ("results_q584.json", "results_q601.json"):
        d = json.load(open(os.path.join(TW, f)))
        if kind != "clean":
            d = corrupt_chain(d, kind, rng)
        json.dump(d, open(os.path.join(sb, f), "w"))
    S = json.load(open(os.path.join(TW, "centre_scripts", "c80_lfp_states.json")))
    if kind != "clean":
        S = corrupt_lfp(S, kind, rng)
    json.dump(S, open(os.path.join(sb, "centre_scripts", "c80_lfp_states.json"), "w"))
    for f in glob.glob(os.path.join(TW, "results_q2*_gauss_K*_A*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if kind != "clean" and isinstance(d, dict) and all(k in d for k in ("A", "u", "w", "we")):
            d = corrupt_cap(d, kind, rng)
        json.dump(d, open(os.path.join(sb, os.path.basename(f)), "w"))
    return sb


def run(sb, script):
    cwd = os.path.join(sb, "centre_scripts") if script.startswith("centre_scripts") else sb
    name = os.path.basename(script)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, name],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    return r.stdout + (("\nSTDERR: " + r.stderr[-400:]) if r.returncode else ""), r.returncode


def nums(pat, text):
    return [int(x) for x in re.findall(pat, text)]


outs = {}
for kind in ("clean", "R", "M", "P", "S"):
    sb = build(kind)
    for s in ROOT_SCRIPTS + CS_SCRIPTS:
        outs[(os.path.basename(s), kind)] = run(sb, s)
    shutil.rmtree(sb, ignore_errors=True)

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))


def O(s, k):
    return outs[(s, k)][0]


# exit codes and row counts of the clean runs (positive control, catches a silent zero-row exit 0)
for s in ROOT_SCRIPTS + [os.path.basename(c) for c in CS_SCRIPTS]:
    check(
        f"{s} clean run exits 0 and prints",
        outs[(s, "clean")][1] == 0 and len(O(s, "clean")) > 200,
        f"exit {outs[(s, 'clean')][1]}, {len(O(s, 'clean'))} characters",
    )
c = O("q877_A1A2_readout.py", "clean")
check("q877 clean: 612 and 544 rows read", "rows 612" in c and "rows 544" in c, "")
check(
    "q888 clean: 612 and 544 rows read",
    "[immune] 612 rows" in O("q888_O1_orientation.py", "clean")
    and "[NPMLE] 544 rows" in O("q888_O1_orientation.py", "clean"),
    "",
)
check("q896 clean: 62 states", "62 states" in O("q896_lfp_structure.py", "clean"), "")
check(
    "q899 clean: K 50 13, K 60 19, K 70 6 states",
    all(
        x in O("q899_capacity_structure.py", "clean")
        for x in ("K 50: 13 states", "K 60: 19 states", "K 70: 6 states")
    ),
    "",
)
check(
    "q900 clean: K 50 13, K 60 19, K 70 6 states",
    all(
        x in O("q900_capacity_flows.py", "clean")
        for x in ("K 50: 13 states", "K 60: 19 states", "K 70: 6 states")
    ),
    "",
)

# T1 q877 A2: clean zero violations, reversed rows many
a2c = nums(r"A2: pairs \d+, violations (\d+)", O("q877_A1A2_readout.py", "clean"))
a2r = nums(r"A2: pairs \d+, violations (\d+)", O("q877_A1A2_readout.py", "R"))
check("q877 A2 (outward flow): clean has 0 violations in both chains", a2c == [0, 0], str(a2c))
check(
    "q877 A2: reversed rows give many violations in both chains",
    len(a2r) == 2 and all(x > 100 for x in a2r),
    str(a2r),
)
# T2 q888 (b)
bc = nums(r"\(b\) c [0-9.e+-]+ -> [0-9.e+-]+, violations (\d+)", O("q888_O1_orientation.py", "clean"))
bm = nums(r"\(b\) c [0-9.e+-]+ -> [0-9.e+-]+, violations (\d+)", O("q888_O1_orientation.py", "S"))
check(
    "q888 (b) centre mass growth: clean has 0 violations in 10 families",
    len(bc) == 10 and sum(bc) == 0,
    f"{len(bc)} families, sum {sum(bc)}",
)
check(
    "q888 (b): masses scaled by noise give violations (a shuffle keeps the row sum and so cannot test c = 1 - sum v; the first version of this control used a shuffle and was invalid)",
    sum(bm) > 0,
    f"sum {sum(bm)}",
)
# T3 q891
v = lambda s, k: (
    re.findall(r"VERDICT: (\w+)", O(s, k))[-1] if re.findall(r"VERDICT: (\w+)", O(s, k)) else "NONE"
)
check(
    "q891 T27 relaxation: clean PASS, reversed FAIL, shuffled masses FAIL",
    (v("q891_T27_relaxation.py", "clean"), v("q891_T27_relaxation.py", "R"), v("q891_T27_relaxation.py", "M"))
    == ("PASS", "FAIL", "FAIL"),
    str(
        (
            v("q891_T27_relaxation.py", "clean"),
            v("q891_T27_relaxation.py", "R"),
            v("q891_T27_relaxation.py", "M"),
        )
    ),
)
# T4 q892: count FAIL family lines
nf = lambda k: len(re.findall(r"-> FAIL", O("q892_T28_lattice_slots.py", k)))
check(
    "q892 T28 lattice: clean has 3 failing families (the recorded ones), jittered and reversed have more",
    nf("clean") == 3 and nf("P") > 3 and nf("R") > 3,
    f"clean {nf('clean')}, P {nf('P')}, R {nf('R')}",
)
# T5 q893
npass = lambda k: len(re.findall(r"-> PASS", O("q893_T31_mass_law.py", k)))
check(
    "q893 T31 mass law: clean 19 passing families, shuffled masses fewer",
    npass("clean") == 19 and npass("M") < 19,
    f"clean {npass('clean')}, M {npass('M')}",
)
# T6 q896
l1c = nums(r"violations (\d+)", O("q896_lfp_structure.py", "clean"))[:1]
l1r = nums(r"violations (\d+)", O("q896_lfp_structure.py", "R"))[:1]
check(
    "q896 L1 outward flow: clean 0 violations, reversed many",
    l1c == [0] and l1r and l1r[0] > 100,
    f"clean {l1c}, R {l1r}",
)
l2 = lambda k: len(re.findall(r"L2 innermost mass [0-9.]+ -> [0-9.]+ PASS", O("q896_lfp_structure.py", k)))
check(
    "q896 L2 innermost mass: clean 13 families pass, shuffled masses fewer",
    l2("clean") == 13 and l2("M") < 13,
    f"clean {l2('clean')}, M {l2('M')}",
)
# T7 q897
g = lambda k: nums(r"families with median \|e\| below 0\.03: (\d+) of", O("q897_lfp_cell_law.py", k))
check(
    "q897 LFP cell law: clean 13 of 13, shuffled masses below 11",
    g("clean") == [13] and g("M") and g("M")[0] < 11,
    f"clean {g('clean')}, M {g('M')}",
)
# T8 q898
im = lambda k: re.findall(r"\[immune\] -> (\w+)", O("q898_flat_density_level.py", k))
check(
    "q898 flat-density level: clean immune PASS, shuffled masses FAIL",
    im("clean") == ["PASS"] and im("M") == ["FAIL"],
    f"clean {im('clean')}, M {im('M')}",
)
# T9 q899
check(
    "q899 capacity structure: clean PASS, shuffled masses FAIL, inward positions FAIL",
    (
        v("q899_capacity_structure.py", "clean"),
        v("q899_capacity_structure.py", "M"),
        v("q899_capacity_structure.py", "R"),
    )
    == ("PASS", "FAIL", "FAIL"),
    str(
        (
            v("q899_capacity_structure.py", "clean"),
            v("q899_capacity_structure.py", "M"),
            v("q899_capacity_structure.py", "R"),
        )
    ),
)
# T10 q900
bb = lambda k: sum(nums(r"\(b\) violations (\d+)", O("q900_capacity_flows.py", k))) + sum(
    nums(r"\(c\) violations (\d+)", O("q900_capacity_flows.py", k))
)
aa = lambda k: sum(nums(r"\(a\) violations (\d+)", O("q900_capacity_flows.py", k)))
check(
    "q900 capacity flows: clean (b),(c) 0 violations and (a) 4; shuffled masses give (b)/(c) violations; inward positions give (a) violations",
    bb("clean") == 0 and aa("clean") == 4 and bb("M") > 0 and aa("R") > 4,
    f"clean (b)+(c) {bb('clean')}, (a) {aa('clean')}; M (b)+(c) {bb('M')}; R (a) {aa('R')}",
)

width = max(len(n) for n, _, _ in results)
lines = [f"{'PASS' if ok else 'FAIL'}  {n.ljust(width)}  {d}" for n, ok, d in results]
bad = sum(1 for _, ok, _ in results if not ok)
lines.append(f"\n{len(results) - bad} of {len(results)} self-tests pass")
text = "\n".join(lines)
print(text)
open(os.path.join(HERE, "selftest_out.txt"), "w", encoding="utf-8").write(text + "\n")
sys.exit(1 if bad else 0)
