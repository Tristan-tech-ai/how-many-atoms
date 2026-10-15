"""q945 (7 Oct 2026): driver of the tower of the capacity support (T66): alternating births (even -> odd) and splits (odd -> even), each followed by the family trace and the registered readouts. The driver only runs scripts and writes their outputs to files;
every verdict is the last line of the script that produced it (q938 birth side, q911 odd family, q931 split, q933 even family, q926b structure of all states); the driver decides nothing.
Usage: py q945_tower.py EVEN_DIR K_EVEN N_EVENTS    (from temporal_within/; EVEN_DIR holds the finished even family K_EVEN, its last state not optimal with the verdict 'centre positive')
Environment: Q945_GRID (frozen grid of the new families, default 46.0), Q945_DA (default 0.005). Outputs: scratch/tower_K<K>_*.txt|json|log, families in tower/K<K>/, stage files in tower/K<K>_stage and tower/K<K>_split. Stops at the first failing stage (nonzero exit of a script that must succeed).
"""

import glob, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
A = "scratch"
GRID = os.environ.get("Q945_GRID", "46.0")
DA = os.environ.get("Q945_DA", "0.005")
PY = sys.executable
MARK = "RESULT q941"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fail(msg):
    log("STAGE FAILED:", msg)
    sys.exit(1)


def clean(v):
    return float(str(v).replace("np.float64(", "").rstrip(")"))


def states(d, K):
    out = []
    for f in sorted(glob.glob(f"{d}/results_q220_gauss_K{K}_A*.json")):
        j = json.load(open(f))
        out.append((clean(j["A"]), f, str(j["verdict"]), float(j["residual"])))
    out.sort()
    return out


def split_family(st):
    opt = []
    for s in st:
        if s[2] == "OPTIMAL" and s[3] < 1e-20:
            opt.append(s)
        elif opt:
            break
    later = [s for s in st if opt and s[0] > opt[-1][0] and s[2] != "OPTIMAL" and s[3] < 1e-20]
    return opt, later


def popen(cmd, logf, env=None):
    return subprocess.Popen(
        [PY] + cmd, stdout=open(logf, "w", encoding="utf-8"), stderr=subprocess.STDOUT, env=env
    )


def run(cmd, logf, env=None):
    p = popen(cmd, logf, env)
    p.wait()
    return p.returncode


def family_readouts(fam, K, even):
    opt, later = split_family(states(fam, K))
    lf = f"{A}/tower_family_list_K{K}.txt"
    open(lf, "w").write(
        "\n".join(
            f"{re.search(r'_A([0-9.]+)[.]json', s[1]).group(1)} {s[1].replace(chr(92), '/')}" for s in opt
        )
        + "\n"
    )
    rc1 = run(["q926b_family_readout_q924b.py", str(K), lf], f"{A}/tower_K{K}_q926b.txt")
    rc2 = run(
        ["q933_even_family.py" if even else "q911_odd_family.py", str(K), fam],
        f"{A}/tower_K{K}_{'q933' if even else 'q911'}.txt",
    )
    log(f"K {K} family readouts: q926b exit {rc1}, {'q933' if even else 'q911'} exit {rc2}")


def trace_family(base, fam, K, a_start):
    os.makedirs(fam, exist_ok=True)
    tl = f"{A}/tower_K{K}_trace.log"
    tr = popen(["q942_trace_cubic.py", base, fam, DA, str(a_start + 1.0), GRID], tl)
    wt = popen(["q946_watch_generic.py", fam, str(K), tl, MARK], f"{A}/tower_K{K}_watch.log")
    return tr, wt


def birth(even_dir, K):
    opt, later = split_family(states(even_dir, K))
    if not opt or not later or later[0][2] != "centre positive":
        fail(f"even family K {K} in {even_dir} does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    Ko = K + 1
    stage = f"tower/K{Ko}_stage"
    fam = f"tower/K{Ko}"
    os.makedirs(stage, exist_ok=True)
    log(f"birth K {K} -> {Ko}: first non-optimal state A {hi[0]}, last optimal A {lo[0]}")
    rc = run(["q922_born_centre_run.py", hi[1], stage], f"{A}/tower_K{Ko}_born.log")
    base = glob.glob(f"{stage}/results_q220_gauss_K{Ko}_A*.json")
    if rc != 0 or not base:
        fail(f"q922 for K {Ko} exit {rc}")
    base = sorted(base)[0]
    tr, wt = trace_family(base, fam, Ko, hi[0])
    bj = f"{A}/tower_K{Ko}_birth_out.json"
    rcb = run(["q944_birth_generic.py", base, hi[1], lo[1], bj, GRID], f"{A}/tower_K{Ko}_birth.log")
    if rcb != 0:
        fail(f"q944 for K {Ko} exit {rcb}")
    run(["q938_birth_verdict.py", bj], f"{A}/tower_K{Ko}_q938.txt")
    tr.wait()
    wt.wait()
    family_readouts(fam, Ko, False)
    return fam, Ko


def split(odd_dir, K):
    opt, later = split_family(states(odd_dir, K))
    if not opt or not later or later[0][2] != "centre positive":
        fail(f"odd family K {K} in {odd_dir} does not end with a centre-positive state")
    hi, lo = later[0], opt[-1]
    mid = min(opt, key=lambda s: abs(s[0] - (lo[0] - 0.08)))
    Ke = K + 1
    odir = f"tower/K{Ke}_split"
    fam = f"tower/K{Ke}"
    os.makedirs(odir, exist_ok=True)
    log(
        f"split K {K} -> {Ke}: first non-optimal A {hi[0]}, last optimal A {lo[0]}, negative control A {mid[0]}"
    )
    env = dict(os.environ, Q935_GRID=GRID, Q935_ODIR=odir)
    asj = f"{A}/tower_K{Ke}_As.json"
    if run(["q935_split_generic.py", "As", asj, lo[1], hi[1]], f"{A}/tower_K{Ke}_As.log", env) != 0:
        fail(f"A_s for K {Ke}")
    scans = {
        tag: (
            f"{A}/tower_K{Ke}_scan_{tag}.json",
            popen(
                ["q935_split_generic.py", "scan", s[1], f"{A}/tower_K{Ke}_scan_{tag}.json"],
                f"{A}/tower_K{Ke}_scan_{tag}.log",
                env,
            ),
        )
        for tag, s in (("hi", hi), ("lo", lo), ("mid", mid))
    }
    for tag, (jf, p) in scans.items():
        p.wait()
        if p.returncode != 0:
            fail(f"scan {tag} for K {Ke} exit {p.returncode}")
    run(
        ["q931_split_verdict.py", scans["hi"][0], scans["lo"][0], scans["mid"][0], asj],
        f"{A}/tower_K{Ke}_q931.txt",
    )
    root = json.load(open(scans["hi"][0]))["root"]
    if not root or "eps" not in root:
        fail(f"no root for K {Ke}")
    cands = [
        f
        for f in glob.glob(f"{odir}/results_q220_gauss_K{Ke}_A*_eps*.json")
        if abs(clean(json.load(open(f))["A"]) - hi[0]) < 1e-9
    ]
    if len(cands) != 1:
        fail(f"root state file for K {Ke}: {len(cands)} candidates")
    tr, wt = trace_family(cands[0], fam, Ke, hi[0])
    tr.wait()
    wt.wait()
    family_readouts(fam, Ke, True)
    return fam, Ke


def main():
    even_dir, K, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    kind = "birth"
    d = even_dir
    for i in range(n):
        d, K = (birth if kind == "birth" else split)(d, K)
        kind = "split" if kind == "birth" else "birth"
        log(f"event {i + 1} of {n} done: family K {K} in {d}")
    print(f"RESULT q945: {n} events done up to K {K}", flush=True)


if __name__ == "__main__":
    main()
