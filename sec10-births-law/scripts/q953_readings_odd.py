"""q953 (7 Oct 2026): generic text copy of q928 for any odd family of the tower (arguments K DIR LIST; no verdict rests on this script; the verdicts are q911 and q926b).
Reads the family states of the list file with the helper functions of q911 and the 106 outputs of q924b, counts what it processes and stops with VOID on any mismatch (missing file, missing line, a count other than 106).
Usage: py q953_readings_odd.py K DIR LIST > scratch/tower_K<K>_readings.txt   (from temporal_within/)"""

import json, os, re, sys
import numpy as np

K_ARG = sys.argv[1]
DIR_ARG = sys.argv[2]
LIST = sys.argv[3]
sys.argv = [sys.argv[0], K_ARG, DIR_ARG]
import q911_odd_family as Q

ARCH = "scratch"


def void(msg):
    print("VOID q953: " + msg, flush=True)
    sys.exit(2)


rows = [l.split() for l in open(LIST).read().splitlines() if l.strip()]
if len(rows) < 4:
    void(f"family list has {len(rows)} rows")
A_l = []
sig = []
wc = []
w1 = []
mr = []
slot = []
law = []
K4 = []
K3 = []
K1 = []
K2 = []
dg = []
nmin = []
verd = []
fl = {k: 0 for k in ("K1", "K2", "K3", "K3c", "K4", "K5", "K6", "K7")}
pat = {
    "K1": re.compile(
        r"K1: columns (\d+), max error ([0-9.e+-]+) -> (\w+); control .* fails as it must (\w+)"
    ),
    "K2": re.compile(
        r"K2: max \|b_an - b_fd\| ([0-9.e+-]+) against ([0-9.e+-]+) -> (\w+); control .* fails as it must (\w+)"
    ),
    "K3": re.compile(
        r"K3: linear residual ([0-9.e+-]+) \((\w+)\); ratios ([0-9.e+-]+) ([0-9.e+-]+) ([0-9.e+-]+) conditions \((\w+), (\w+), (\w+), (\w+)\) -> (\w+)"
    ),
    "K3c": re.compile(
        r"K3 controls: D1 conditions .* fails as it must (\w+); D2 conditions .* fails as it must (\w+)"
    ),
    "K4": re.compile(
        r"K4: du_j/dA > 0 for all \d+ positions (\w+) \(min ([0-9.e+-]+), max ([0-9.e+-]+)\); dw_1/dA = ([0-9.e+-]+) \((\w+)\); dwe/dA = ([0-9.e+-]+); dwc/dA = ([0-9.e+-]+) \((\w+)\)"
    ),
    "H": re.compile(r"reduced Hessian \d+ x \d+; diagonal from ([0-9.e+-]+) to ([0-9.e+-]+)"),
    "TP": re.compile(
        r"\[clean\] TP orders 1-3: minors examined \{1: (\d+), 2: (\d+), 3: (\d+)\}, failures (\d+)"
    ),
    "TPh": re.compile(r"\[clean\] TPh orders 4-5: minors examined \{4: (\d+), 5: (\d+)\}, failures (\d+)"),
    "K5": re.compile(r"K5: C-sign (\w+), C-inv (\w+), TP (\w+), TPh (\w+) -> (\w+)"),
    "K67": re.compile(
        r"K6 positive control passes all four gates: (\w+); K7 negative controls fail as they must: (\w+)"
    ),
    "V": re.compile(r"VERDICT q924b \S+: (\w+);"),
}


def num(x):
    return float(x)


minors = []
tpfail = 0
n_pass = 0
for a4, f in rows:
    s = json.load(open(f))
    A_l.append(Q.clean(s["A"]))
    sig.append(float(s["centre_signal"]))
    wc.append(Q.clean(s["wc"]))
    st = {
        "u": np.array([Q.clean(x) for x in s["u"]]),
        "w": np.array([Q.clean(x) for x in s["w"]]),
        "A": Q.clean(s["A"]),
        "wc": Q.clean(s["wc"]),
    }
    w1.append(st["w"][0])
    mr.append(Q.mref(st))
    slot.append(Q.lattice(st))
    law.append(Q.law(st))
    verd.append(str(s["verdict"]))
    t = open(os.path.join(ARCH, f"q924b_K{K_ARG}_A{a4}_out.txt"), encoding="utf-8").read()
    got = {}
    for k, p in pat.items():
        m = p.search(t)
        if not m:
            void(f"line {k} missing in q924b output for A {a4}")
        got[k] = m.groups()
    g = got["K1"]
    K1.append(num(g[1]))
    fl["K1"] += g[2] == "True" and g[3] == "True"
    g = got["K2"]
    K2.append(num(g[0]))
    fl["K2"] += g[2] == "True" and g[3] == "True"
    g = got["K3"]
    K3.append([num(g[2]), num(g[3]), num(g[4])])
    fl["K3"] += g[1] == "True" and all(x == "True" for x in g[5:9]) and g[9] == "True"
    g = got["K3c"]
    fl["K3c"] += g[0] == "True" and g[1] == "True"
    g = got["K4"]
    K4.append(
        {
            "A": float(a4),
            "umin": num(g[1]),
            "umax": num(g[2]),
            "dw1": num(g[3]),
            "dwe": num(g[5]),
            "dwc": num(g[6]),
            "okpos": g[0],
            "okdwc": g[7],
        }
    )
    fl["K4"] += g[0] == "True" and g[7] == "True"
    g = got["H"]
    dg.append((num(g[0]), num(g[1])))
    g = got["TP"]
    minors.append(tuple(int(x) for x in g[:3]))
    tpfail += int(g[3])
    g = got["TPh"]
    minors[-1] = minors[-1] + (int(g[0]), int(g[1]))
    tpfail += int(g[2])
    g = got["K5"]
    fl["K5"] += all(x == "True" for x in g)
    g = got["K67"]
    fl["K6"] += g[0] == "True"
    fl["K7"] += g[1] == "True"
    n_pass += got["V"][0] == "PASS"
n = len(rows)
print(f"READINGS q953 K {K_ARG}: family states {n}, q924b outputs read {n}, q924b PASS lines {n_pass}")
print(f"counts of True over {n} states: " + ", ".join(f"{k} {v}" for k, v in fl.items()))
print(
    f"states in the file list: first A {A_l[0]}, last A {A_l[-1]}, span {A_l[-1] - A_l[0]:.4f}, all verdicts OPTIMAL {all(v == 'OPTIMAL' for v in verd)}"
)
print(f"minors per state (orders 1-3 and 4-5) distinct sets {sorted(set(minors))}; total failures {tpfail}")
print(
    f"K1 max error over states: min {min(K1):.3e}, max {max(K1):.3e}; K2 max error: min {min(K2):.3e}, max {max(K2):.3e}"
)
for i, d in enumerate(("1e-12", "1e-14", "1e-16")):
    print(f"K3 ratio at delta {d}: min {min(r[i] for r in K3):.3e}, max {max(r[i] for r in K3):.3e}")
print(
    f"K4 du_j/dA: smallest over all positions and states {min(k['umin'] for k in K4):.4f}, largest {max(k['umax'] for k in K4):.4f}"
)
print(
    f"K4 dwc/dA: min {min(k['dwc'] for k in K4):.6f}, max {max(k['dwc'] for k in K4):.6f}, positive in {sum(k['dwc'] > 0 for k in K4)} states; first {K4[0]['dwc']:.5f}, last {K4[-1]['dwc']:.6f}"
)
neg = [k["A"] for k in K4 if k["dw1"] < 0]
pos = [k["A"] for k in K4 if k["dw1"] > 0]
print(
    f"K4 dw_1/dA: min {min(k['dw1'] for k in K4):.5f} (A {min(K4, key=lambda k: k['dw1'])['A']}), max {max(k['dw1'] for k in K4):.6f} (A {max(K4, key=lambda k: k['dw1'])['A']}); negative in {len(neg)} states (A {neg[0]} to {neg[-1]}), positive in {len(pos)} states (from A {pos[0]})"
)
print(f"K4 dwe/dA: min {min(k['dwe'] for k in K4):.7f}, max {max(k['dwe'] for k in K4):.7f}")
print(
    f"reduced Hessian diagonal: most negative entry over states {min(d[0] for d in dg):.4f} to {max(d[0] for d in dg):.4f}; least negative entry {min(d[1] for d in dg):.3e} to {max(d[1] for d in dg):.3e}"
)
i = int(np.argmin(sig))
j = int(np.argmin(w1))
print(
    f"centre signal: first {sig[0]:.4e}, minimum {sig[i]:.4e} at A {A_l[i]}, last {sig[-1]:.4e}, all non-positive {all(x <= 0 for x in sig)}"
)
st_all = Q.load(DIR_ARG)
opt_, later_ = Q.split_family(st_all)
if not later_:
    print("no state after the family (not completed)")
else:
    nxt = [x for x in st_all if x["A"] > opt_[-1]["A"] and x["verdict"] != "OPTIMAL"][0]
    sp = json.load(open(nxt["f"]))
    print(
        f"state after the family A {sp['A']}: verdict {sp['verdict']}, centre signal {float(sp['centre_signal']):.4e}, wc {float(sp['wc']):.8f}, residual {float(sp['residual']):.2e}"
    )
print(
    f"centre mass wc: first {wc[0]:.7f}, last {wc[-1]:.9f}; innermost pair w_1: first {w1[0]:.7f}, minimum {w1[j]:.7f} at A {A_l[j]}, last {w1[-1]:.9f}"
)
print(
    f"m_ref: first {mr[0]:.7f}, last {mr[-1]:.9f}; r = wc/m_ref: first {wc[0] / mr[0]:.5f}, last {wc[-1] / mr[-1]:.6f}"
)
print(
    f"lattice slot u_1/(u_2-u_1): first {slot[0]:.4f}, last {slot[-1]:.4f}; law wc/(mu_bar u_1): first {law[0]:.4f}, last {law[-1]:.5f}"
)
print("END q953 READINGS (no verdict)")
