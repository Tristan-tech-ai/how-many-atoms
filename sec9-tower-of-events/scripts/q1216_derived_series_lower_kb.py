"""q1216 (8 Oct 2026, prepared under amendment 1973): how low in K_b does the births law with the derived centre series (orders up to 16, no immune input) reach?
Capacity half-line c50c D = 110 and NPMLE half-line c33_npmc_D110, exactly as q1209. Coefficients: d22_out_from_log.json (e2 .. e16).
The transitions compared are the STORED outputs of the c60 family (capacity K_b 3 to 59) and of the q607 family (NPMLE K_b 2 to 52); they exist since 25 Sep to 1 Oct and are not blind.
For each K_b the table holds a_n, the root of 2 N(a) - K_b - 1/6 - sum_{m=2}^{n} e_m / v^(2m) = 0, for n = 5 to 16, bracketed by +-2.5 around the zeroth-order root of 2 N(a) = K_b + 1/6.
Rules fixed before any deviation is computed:
  bar_K  = 0.3e-9 + 3 |a_16 - a_14|                                   (order-16 item L1)
  n_opt  = the n in 6..16 with the smallest |a_n - a_(n-1)|; a_opt = a_(n_opt); u = |a_(n_opt+1) - a_(n_opt)| (for n_opt = 16: |a_16 - a_15|); bar_opt = 0.3e-9 + 3 u   (optimal-truncation item L1')
  L2     = |dev(16)| <= |dev(12)| + 0.1e-9 (higher orders do not worsen the agreement, for events where L1 holds).
  K*     = the smallest K_b such that L1 holds for every stored event from K_b up to 59 (capacity) or 52 (NPMLE); K*' the same for L1'.
--registration-run : controls and the table, no stored transition amplitude read; writes q1216_predictions.json with a sha256 of the table.
--read             : the items for each stored event, K*, K*'.
Usage: py q1216_derived_series_lower_kb.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import hashlib, json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.optimize import brentq
import q1209_births_law_derived_series as G

CAP_K = list(range(10, 61))
NPM_K = list(range(6, 54))
ORD = list(range(5, 17))
CAP_FILES = {
    3: "c60_gate_K3",
    4: "c60_gate_K4",
    5: "c60_gate_K5",
    6: "c60_gate_K6",
    13: "c60_gate_K13",
    20: "c60_K20",
    25: "c60_K25",
    47: "c60_d38_K47",
    48: "c60_d38_K48",
    49: "c60_d38_K49",
    50: "c60_d38_K50",
    51: "c60_u50b_K51",
    52: "c60_u50_K52",
    53: "c60_u61_K53",
    55: "c60_u61_K55",
    57: "c60_u61_K57",
    59: "c60_u61_K59",
}
for k in range(7, 13):
    CAP_FILES[k] = f"c60_small_K{k}"
for k in list(range(14, 20)) + [21, 22, 23, 24]:
    CAP_FILES[k] = f"c60_K{k}"
for k in range(26, 39):
    CAP_FILES[k] = f"c60_small_K{k}"
for k in range(39, 47):
    CAP_FILES[k] = f"c60_f3_K{k}"
NPM_FILES = {k: f"q607_sm_K{k}" for k in range(2, 10)}
NPM_FILES.update({k: f"q607_lo_K{k}" for k in range(10, 25)})
NPM_FILES.update({k: f"q607_gp_K{k}" for k in (25, 26, 27, 28, 31)})
NPM_FILES.update({k: f"q607_hl_K{k}" for k in range(34, 43)})
NPM_FILES.update(
    {
        43: "q607_d38_K43",
        44: "q607_d38_K44",
        46: "q607_d38_K46",
        47: "q607_d38_K47",
        48: "q607_d38_K48",
        50: "q607_u50_K50",
        52: "q607_u50_K52",
    }
)


def root(N, K, nmax):
    z = lambda a: 2 * N(a) - K - 1 / 6
    a0 = brentq(z, 1.0, 140.0, xtol=1e-12)
    f = lambda a: z(a) - G.delta((N(a + G.H) - N(a - G.H)) / (2 * G.H), nmax)
    lo, hi = a0 - 2.5, a0 + 2.5
    if f(lo) * f(hi) > 0:
        return float("nan")
    return brentq(f, lo, hi, xtol=1e-14)


def table():
    out = {"capacity": {}, "npmle": {}}
    for name, N, ks in (("capacity", G.Ncap, CAP_K), ("npmle", G.Nnp, NPM_K)):
        for K in ks:
            a = {n: root(N, K, n) for n in ORD}
            rho = (N(a[16] + G.H) - N(a[16] - G.H)) / (2 * G.H) if a[16] == a[16] else float("nan")
            out[name][str(K)] = {"a": {str(n): a[n] for n in ORD}, "v": float(4 * np.pi * rho)}
    return out


def rules(r):
    a = {int(n): x for n, x in r["a"].items()}
    if any(x != x for x in a.values()):
        return None
    bar = 0.3e-9 + 3 * abs(a[16] - a[14])
    nopt = min(range(6, 17), key=lambda n: abs(a[n] - a[n - 1]))
    u = abs((a[nopt + 1] if nopt < 16 else a[15]) - a[nopt])
    u = abs(a[16] - a[15]) if nopt == 16 else u
    return a, bar, nopt, a[nopt], 0.3e-9 + 3 * u


def regrun():
    ok = True
    a80 = G.pred_series(G.Ncap, 80, 16)
    g = abs(a80 - 51.484240309367) < 1e-11
    ok &= g
    print(
        f"control: capacity K_b 80 at order 16 (q1209 function) {a80:.12f} against q1211 51.484240309367 -> {g}"
    )
    b80 = root(G.Ncap, 80, 16)
    g = abs(b80 - a80) < 1e-11
    ok &= g
    print(f"control: the bracketed root function gives {b80:.12f} for K_b 80 -> {g}")
    d = json.load(open("q607_u1205_K60.json"))["a_t"] - root(G.Nnp, 60, 16)
    g = abs(d) < 0.3e-9
    ok &= g
    print(f"control: NPMLE K_b 60 deviation at order 16 {d * 1e9:+.4f}e-9 (bar 0.3e-9) -> {g}")
    for name, F in (("capacity", CAP_FILES), ("npmle", NPM_FILES)):
        for K, f in F.items():
            j = json.load(open(f + ".json"))
            g = isinstance(j, dict) and j.get("K") == K and (j.get("a_t") or j.get("A_t")) is not None
            ok &= g
            if not g:
                print(f"control: file {f} does not carry K = {K} and a transition amplitude -> False")
    print(
        "control: every named stored file carries its K and an amplitude (the amplitude is not read here):",
        ok,
    )
    if not ok:
        print("REGISTRATION-RUN q1216: VOID (control)")
        sys.exit(2)
    t = table()
    sh = hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
    t["sha256"] = sh
    json.dump(t, open("q1216_predictions.json", "w"), indent=1)
    for name in ("capacity", "npmle"):
        for K, r in t[name].items():
            q = rules(r)
            if q is None:
                print(f"{name} K_b {K}: no root in the bracket for some order")
                continue
            a, bar, nopt, aopt, bopt = q
            print(
                f"{name} K_b {K}: v {r['v']:.3f}, a_16 {a[16]:.10f}, bar {bar * 1e9:.3f}e-9, n_opt {nopt}, a_opt {aopt:.10f}, bar_opt {bopt * 1e9:.3f}e-9"
            )
    print("sha256", sh)
    print("REGISTRATION-RUN q1216: controls True, table written")


def read():
    js = json.load(open("q1216_predictions.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    ok = h == sh
    print(f"control: hash of the table {h[:16]}... equals the stored one: {ok}")
    if not ok:
        print("READING q1216: VOID (control)")
        sys.exit(2)
    for name, F, kmax in (("capacity", CAP_FILES, 59), ("npmle", NPM_FILES, 52)):
        rows = {}
        for K in sorted(F):
            if str(K) not in js[name]:
                continue
            q = rules(js[name][str(K)])
            j = json.load(open(F[K] + ".json"))
            m = j.get("a_t") or j.get("A_t")
            if q is None:
                print(f"{name} K_b {K}: no root")
                rows[K] = (False, False, False)
                continue
            a, bar, nopt, aopt, bopt = q
            d16 = m - a[16]
            d12 = m - a[12]
            dop = m - aopt
            L1 = abs(d16) < bar
            L1o = abs(dop) < bopt
            L2 = abs(d16) <= abs(d12) + 0.1e-9
            rows[K] = (L1, L1o, L2)
            print(
                f"{name} K_b {K}: A_stored {m:.10f}; dev(12) {d12 * 1e9:+.4f}e-9, dev(16) {d16 * 1e9:+.4f}e-9 (bar {bar * 1e9:.3f}e-9) L1 {L1}; n_opt {nopt} dev(opt) {dop * 1e9:+.4f}e-9 (bar {bopt * 1e9:.3f}e-9) L1' {L1o}; L2 {L2}"
            )

        def kstar(i):
            best = None
            for K in sorted(rows, reverse=True):
                if rows[K][i]:
                    best = K
                else:
                    break
            return best

        print(
            f"{name}: events {len(rows)}, L1 true {sum(v[0] for v in rows.values())}, L1' true {sum(v[1] for v in rows.values())}, L2 true {sum(v[2] for v in rows.values())}; K* = {kstar(0)}, K*' = {kstar(1)}"
        )
    print("READING q1216 (no verdict)")


if __name__ == "__main__":
    regrun() if mode == "--registration-run" else read()
