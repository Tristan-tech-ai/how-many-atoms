"""q911 (6 Oct 2026): blind forward test of the centre-family structure on the capacity of the Gaussian channel (T61; criteria registered in an amendment before any state of the family existed).
Data: the saved states of the capacity walk from the K 60 state at A = 41.21 with a centre atom added (q220_walk_A.py on results_q910_seed_K61_A41.2100.json), files results_q220_gauss_K<K>_A<A>.json
in a directory (argument 2, default capacity_odd) with the fields A, K, u (interior positive atoms, ascending), w (per-atom masses), we (mass of the atom on the wall at A), wc (mass of the centre atom), residual, verdict.
The family is the run of states with verdict OPTIMAL and residual < 1e-20, sorted by A. It is COMPLETED when a later saved state has a verdict other than OPTIMAL with residual < 1e-20 (the K + 1 optimum has a split pair).
Definitions as in q895 (amendment 1663) with the capacity masses: per-atom interior masses w_j; positions P = [0, u_1, ..., u_m, A]; cell_k = (P[k+1] - P[k-1])/2 for the interior atom k = 1..m; mu_k = w_k / cell_k; mu_bar = mean of mu_k over k = 3..8;
the centre atom's cell is u_1. m_ref = c0 of the least-squares fit w_k = c0 + c1 u_k^2 + c2 u_k^4 over the six innermost atoms; r = wc / m_ref. Lattice slot u = u_1/(u_2 - u_1).
H1 (A2): between consecutive states every interior atom moves outward, u_j(A') > u_j(A) for every j.
H2: the centre mass wc increases strictly from state to state.
H3: r increases strictly from state to state; at the last state of a completed family r lies in [0.999, 1.0).
H4: u at the first state of the family lies in [0.50, 0.60]; at the last state of a completed family u lies in [0.98, 1.02].
H5: at the last state of a completed family |wc / (mu_bar u_1) - 1| <= 1e-3.
Verdict (last line): PASS = H1-H5 all examined and passed; FAIL = any examined item fails; INCOMPLETE = no failure but the family is not completed (H3 last, H4 last and H5 not examined); VOID = fewer than 4 OPTIMAL states, an unreadable file or a control that does not turn the verdict into FAIL.
Controls inside the script (each applied to the data, each must give FAIL): R states in reverse order of A; M the per-atom masses shuffled within each state (seed 20261006); S every mass multiplied by 1 + 0.02 n; P the positions multiplied by 1 + 0.02 n and re-sorted.
A completed family is not needed for R, M, S, P to be detected (H1, H2, H3 and the first-state item act on the states that exist). Exit: 0 PASS, 1 FAIL, 2 VOID, 3 INCOMPLETE.
Usage: py q911_odd_family.py [K] [DIR]   (from temporal_within/; K default 61, DIR default capacity_odd)"""

import glob, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 61
DIR = sys.argv[2] if len(sys.argv) > 2 else "capacity_odd"


def void(msg):
    print("VOID q911: " + msg, flush=True)
    sys.exit(2)


def clean(v):
    return float(str(v).replace("np.float64(", "").rstrip(")"))


def load(dirn):
    files = sorted(glob.glob(os.path.join(dirn, f"results_q220_gauss_K{K}_A*.json")))
    if not files:
        void(f"no state files for K {K} in {dirn}")
    st = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            void(f"unreadable {f}: {e}")
        if d.get("wc") is None:
            void(f"state without centre atom in {f}")
        st.append(
            {
                "A": clean(d["A"]),
                "u": np.array([clean(x) for x in d["u"]]),
                "w": np.array([clean(x) for x in d["w"]]),
                "we": clean(d["we"]),
                "wc": clean(d["wc"]),
                "res": float(d["residual"]),
                "verdict": str(d["verdict"]),
                "f": f,
            }
        )
    st.sort(key=lambda s: s["A"])
    return st


def split_family(st):
    opt = []
    for s in st:
        if s["verdict"] == "OPTIMAL" and s["res"] < 1e-20:
            opt.append(s)
        elif opt:
            break
    last_a = opt[-1]["A"] if opt else None
    later = [
        s
        for s in st
        if last_a is not None and s["A"] > last_a and s["verdict"] != "OPTIMAL" and s["res"] < 1e-20
    ]
    return opt, bool(later)


def mref(s):
    u = s["u"][:6]
    w = s["w"][:6]
    X = np.vstack([np.ones(6), u**2, u**4]).T
    c, *_ = np.linalg.lstsq(X, w, rcond=None)
    return c[0]


def lattice(s):
    return s["u"][0] / (s["u"][1] - s["u"][0])


def law(s):
    u = s["u"]
    w = s["w"]
    m = len(u)
    P = np.concatenate(([0.0], u, [s["A"]]))
    mu = np.array([w[k - 1] / ((P[k + 1] - P[k - 1]) / 2) for k in range(1, m + 1)])
    mub = mu[2:8].mean()
    return s["wc"] / (mub * u[0])


def items(opt, completed):
    out = {}
    n = len(opt)
    viol = sum(
        1 for i in range(n - 1) for j in range(len(opt[i]["u"])) if not (opt[i + 1]["u"][j] > opt[i]["u"][j])
    )
    out["H1"] = (viol == 0, f"violations {viol} in {n - 1} steps")
    wc = [s["wc"] for s in opt]
    out["H2"] = (all(b > a for a, b in zip(wc, wc[1:])), f"wc {wc[0]:.6f} -> {wc[-1]:.6f}")
    r = [s["wc"] / mref(s) for s in opt]
    inc = all(b > a for a, b in zip(r, r[1:]))
    out["H3a"] = (inc, f"r {r[0]:.4f} -> {r[-1]:.6f}")
    uf = lattice(opt[0])
    out["H4first"] = (0.50 <= uf <= 0.60, f"u first {uf:.4f}")
    if completed:
        out["H3last"] = (0.999 <= r[-1] < 1.0, f"r last {r[-1]:.6f}")
        ul = lattice(opt[-1])
        out["H4last"] = (0.98 <= ul <= 1.02, f"u last {ul:.4f}")
        lw = law(opt[-1])
        out["H5"] = (abs(lw - 1) <= 1e-3, f"law ratio {lw:.5f}")
    return out


def verdict_of(opt, completed):
    it = items(opt, completed)
    fails = [k for k, (ok, _) in it.items() if not ok]
    if fails:
        return "FAIL", it, fails
    return ("PASS" if completed else "INCOMPLETE"), it, fails


def corrupt(opt, kind):
    rng = random.Random(20261006)
    nrng = np.random.default_rng(20261006)
    out = []
    for s in opt:
        t = dict(s)
        t["u"] = s["u"].copy()
        t["w"] = s["w"].copy()
        if kind == "M":
            rng.shuffle(t["w"])
        elif kind == "S":
            t["w"] = t["w"] * (1 + 0.02 * nrng.standard_normal(len(t["w"])))
            t["wc"] = t["wc"] * (1 + 0.02 * nrng.standard_normal())
            t["we"] = t["we"] * (1 + 0.02 * nrng.standard_normal())
        elif kind == "P":
            t["u"] = np.sort(t["u"] * (1 + 0.02 * nrng.standard_normal(len(t["u"]))))
        out.append(t)
    if kind == "R":
        out = out[::-1]
    return out


def synthetic():
    """a family built to satisfy H1-H5 (positive control): u_j = d (j - 1 + h), w_j = d, wc = r d, h from 0.55 to 1, r from 0.90 to 0.9995, d growing with A."""
    opt = []
    for i in range(12):
        d = 1 + 0.02 * i
        h = 0.55 + 0.45 * i / 11
        r = 0.90 + (0.9995 - 0.90) * i / 11
        u = d * (np.arange(12) + h)
        opt.append({"A": float(u[-1] + d), "u": u, "w": np.full(12, d), "we": 0.1, "wc": r * d})
    return opt


def main():
    st = load(DIR)
    opt, completed = split_family(st)
    print(
        f"[K {K}] files {len(st)}, OPTIMAL family states {len(opt)}, A {opt[0]['A'] if opt else None} to {opt[-1]['A'] if opt else None}, completed {completed}"
    )
    if len(opt) < 4:
        void(f"fewer than 4 OPTIMAL states ({len(opt)})")
    v, it, fails = verdict_of(opt, completed)
    for k, (ok, msg) in it.items():
        print(f"   {k}: {'PASS' if ok else 'FAIL'}  {msg}")
    ctrl = {}
    for kind in ("R", "M", "S", "P"):
        vc, itc, fc = verdict_of(corrupt(opt, kind), completed)
        ctrl[kind] = vc == "FAIL"
        print(f"   control {kind}: verdict {vc}, failed items {fc}")
    ctrl["PC"] = verdict_of(synthetic(), True)[0] == "PASS"
    print(f"   control PC (synthetic family that must PASS): {ctrl['PC']}")
    if not all(ctrl.values()):
        print("VOID q911: controls that failed: " + str([k for k, t in ctrl.items() if not t]), flush=True)
        sys.exit(2)
    print(
        f"VERDICT q911 K {K}: {v}; states {len(opt)}; completed {completed}; failed items {fails}; controls R M S P turn FAIL and PC passes"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
