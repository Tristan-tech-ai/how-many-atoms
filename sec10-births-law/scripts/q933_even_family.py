"""q933 (7 Oct 2026): forward test of the family without a centre atom that follows a split on the capacity of the Gaussian channel (T65; criteria registered in an amendment before any state beyond the root state was read).
Data: files results_q220_gauss_K<K>_A<A>.json in a directory (arguments: K default 62, DIR default capacity_even62_family) with the fields A, K, u (interior positive atoms, ascending), w (per-atom masses), we, residual, verdict (no centre atom).
The family is the run of states with verdict OPTIMAL and residual < 1e-20, sorted by A. It is COMPLETED when a later saved state has a verdict other than OPTIMAL with residual < 1e-20.
Definitions (as q895 and q911 for a family without a centre atom): positions P = [-u_1, u_1, ..., u_m, A]; cell_k = (P[k+1] - P[k-1])/2 for the interior atom k = 1..m; mu_k = w_k / cell_k; mu_bar = mean of mu_k over k = 3..8; lattice slot w = 2 u_1/(u_2 - u_1).
E1 (A2): between consecutive states every interior atom moves outward.
E2: the innermost pair mass w_1 increases strictly from state to state.
E3: the slot w at the first state is below 0.5.
E4: the slot w at the last state of a completed family lies in [0.98, 1.02].
E5: at the last state of a completed family |mu_1/mu_bar - 1| <= 1e-3.
E6: the first state after a completed family has the verdict "centre positive" (the next optimum has a centre atom: a birth).
Verdict (last line): PASS = E1-E6 examined and passed; FAIL = any examined item fails; INCOMPLETE = no failure and the family is not completed (E4, E5, E6 not examined); VOID = fewer than 4 OPTIMAL states, an unreadable file or a control that does not behave.
Controls (each applied to the data, each must give FAIL): R states in reverse order; M masses shuffled within each state (seed 20261007); S every mass times 1 + 0.02 n; P positions times 1 + 0.02 n re-sorted; and PC, a synthetic family built to satisfy E1-E6, which must give PASS.
Exit: 0 PASS, 1 FAIL, 2 VOID, 3 INCOMPLETE. Usage: py q933_even_family.py [K] [DIR]"""

import glob, json, os, random, sys
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 62
DIR = sys.argv[2] if len(sys.argv) > 2 else "capacity_even62_family"


def void(msg):
    print("VOID q933: " + msg, flush=True)
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
        if d.get("wc") is not None:
            void(f"state with a centre atom in {f}")
        st.append(
            {
                "A": clean(d["A"]),
                "u": np.array([clean(x) for x in d["u"]]),
                "w": np.array([clean(x) for x in d["w"]]),
                "we": clean(d["we"]),
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
    return opt, later


def slot(s):
    return 2 * s["u"][0] / (s["u"][1] - s["u"][0])


def law(s):
    u = s["u"]
    w = s["w"]
    m = len(u)
    P = np.concatenate(([-u[0]], u, [s["A"]]))
    mu = np.array([w[k - 1] / ((P[k + 1] - P[k - 1]) / 2) for k in range(1, m + 1)])
    return mu[0] / mu[2:8].mean()


def items(opt, later):
    out = {}
    n = len(opt)
    viol = sum(
        1 for i in range(n - 1) for j in range(len(opt[i]["u"])) if not (opt[i + 1]["u"][j] > opt[i]["u"][j])
    )
    out["E1"] = (viol == 0, f"violations {viol} in {n - 1} steps")
    w1 = [s["w"][0] for s in opt]
    bad = sum(1 for a, b in zip(w1, w1[1:]) if not b > a)
    out["E2"] = (bad == 0, f"w_1 {w1[0]:.6f} -> {w1[-1]:.6f}, non-increasing steps {bad}")
    sf = slot(opt[0])
    out["E3"] = (sf < 0.5, f"slot first {sf:.4f}")
    if later:
        sl = slot(opt[-1])
        out["E4"] = (0.98 <= sl <= 1.02, f"slot last {sl:.4f}")
        lw = law(opt[-1])
        out["E5"] = (abs(lw - 1) <= 1e-3, f"law ratio {lw:.5f}")
        out["E6"] = (
            later[0]["verdict"] == "centre positive",
            f"next state A {later[0]['A']} verdict {later[0]['verdict']}",
        )
    return out


def verdict_of(opt, later):
    it = items(opt, later)
    fails = [k for k, (ok, _) in it.items() if not ok]
    if fails:
        return "FAIL", it, fails
    return ("PASS" if later else "INCOMPLETE"), it, fails


def corrupt(opt, kind):
    rng = random.Random(20261007)
    nrng = np.random.default_rng(20261007)
    out = []
    for s in opt:
        t = dict(s)
        t["u"] = s["u"].copy()
        t["w"] = s["w"].copy()
        if kind == "M":
            rng.shuffle(t["w"])
        elif kind == "S":
            t["w"] = t["w"] * (1 + 0.02 * nrng.standard_normal(len(t["w"])))
        elif kind == "P":
            t["u"] = np.sort(t["u"] * (1 + 0.02 * nrng.standard_normal(len(t["u"]))))
        out.append(t)
    if kind == "R":
        out = out[::-1]
    return out


def synthetic():
    """a family built to satisfy E1-E6: u_j = d (j - 1 + s), w_j = the cell of atom j (density 1), s from 0.2 to 0.5, d growing; the next state has the verdict centre positive"""
    opt = []
    for i in range(12):
        d = 1 + 0.02 * i
        s = 0.2 + 0.3 * i / 11
        u = d * (np.arange(12) + s)
        A = float(u[-1] + d)
        P = np.concatenate(([-u[0]], u, [A]))
        w = np.array([(P[k + 1] - P[k - 1]) / 2 for k in range(1, 13)])
        opt.append({"A": A, "u": u, "w": w, "we": 0.1, "res": 1e-40, "verdict": "OPTIMAL"})
    later = [
        {
            "A": opt[-1]["A"] + 0.005,
            "u": opt[-1]["u"],
            "w": opt[-1]["w"],
            "we": 0.1,
            "res": 1e-40,
            "verdict": "centre positive",
        }
    ]
    return opt, later


def main():
    st = load(DIR)
    opt, later = split_family(st)
    print(
        f"[K {K}] files {len(st)}, OPTIMAL family states {len(opt)}, A {opt[0]['A'] if opt else None} to {opt[-1]['A'] if opt else None}, completed {bool(later)}"
    )
    if len(opt) < 4:
        void(f"fewer than 4 OPTIMAL states ({len(opt)})")
    v, it, fails = verdict_of(opt, later)
    for k, (ok, msg) in it.items():
        print(f"   {k}: {'PASS' if ok else 'FAIL'}  {msg}")
    ctrl = {}
    for kind in ("R", "M", "S", "P"):
        vc, itc, fc = verdict_of(corrupt(opt, kind), later)
        ctrl[kind] = vc == "FAIL"
        print(f"   control {kind}: verdict {vc}, failed items {fc}")
    so, sl = synthetic()
    ctrl["PC"] = verdict_of(so, sl)[0] == "PASS"
    print(f"   control PC (synthetic family that must PASS): {ctrl['PC']}")
    if not all(ctrl.values()):
        void("controls that failed: " + str([k for k, t in ctrl.items() if not t]))
    print(
        f"VERDICT q933 K {K}: {v}; states {len(opt)}; completed {bool(later)}; failed items {fails}; controls R M S P turn FAIL and PC passes"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
