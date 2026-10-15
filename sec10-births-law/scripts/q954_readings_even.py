"""q954 (7 Oct 2026): generic text copy of q936 for any even family of the tower (arguments K DIR LIST; READINGS of a family without a centre atom). No verdict rests on this script; the verdicts are q933 and q926b.
Reads the family states listed in scratch/family_list_K62.txt (lines 'A4 file') with the helper functions of q933 and the matching outputs of q924b, counts what it processes and stops with VOID on any missing file or line.
Usage: py q954_readings_even.py K DIR LIST > scratch/tower_K<K>_readings.txt   (from temporal_within/)"""

import json, os, re, sys
import numpy as np

K_ARG = sys.argv[1]
DIR_ARG = sys.argv[2]
LIST = sys.argv[3]
sys.argv = [sys.argv[0], K_ARG, DIR_ARG]
import q933_even_family as Q

ARCH = "scratch"


def void(msg):
    print("VOID q954: " + msg, flush=True)
    sys.exit(2)


rows = [l.split() for l in open(LIST).read().splitlines() if l.strip()]
if len(rows) < 4:
    void(f"family list has {len(rows)} rows")
A_l = []
sig = []
off = []
w1 = []
slot = []
law = []
K4 = []
K3 = []
K1 = []
K2 = []
dg = []
verd = []
minors = []
tpfail = 0
n_pass = 0
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
        r"K4: du_j/dA > 0 for all (\d+) positions (\w+) \(min ([0-9.e+-]+), max ([0-9.e+-]+)\); dw_1/dA = ([0-9.e+-]+) \((\w+)\); dwe/dA = ([0-9.e+-]+)"
    ),
    "H": re.compile(r"reduced Hessian (\d+) x (\d+); diagonal from ([0-9.e+-]+) to ([0-9.e+-]+)"),
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


for a4, f in rows:
    s = json.load(open(f))
    A_l.append(Q.clean(s["A"]))
    verd.append(str(s["verdict"]))
    sig.append(float(s["centre_signal"]))
    off.append(float(s["off_atom_max"]))
    st = {
        "u": np.array([Q.clean(x) for x in s["u"]]),
        "w": np.array([Q.clean(x) for x in s["w"]]),
        "A": Q.clean(s["A"]),
    }
    w1.append(st["w"][0])
    slot.append(Q.slot(st))
    law.append(Q.law(st))
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
            "n": int(g[0]),
            "umin": num(g[2]),
            "umax": num(g[3]),
            "dw1": num(g[4]),
            "dwe": num(g[6]),
        }
    )
    fl["K4"] += g[1] == "True" and g[5] == "True"
    g = got["H"]
    dg.append((num(g[2]), num(g[3])))
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
j = int(np.argmin(sig))
print(f"READINGS q954 K {K_ARG}: family states {n}, q924b outputs read {n}, q924b PASS lines {n_pass}")
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
    f"K4 positions counted per state: {sorted(set(k['n'] for k in K4))}; du_j/dA: smallest over all positions and states {min(k['umin'] for k in K4):.4f}, largest {max(k['umax'] for k in K4):.4f} (A {max(K4, key=lambda k: k['umax'])['A']})"
)
print(
    f"K4 dw_1/dA: min {min(k['dw1'] for k in K4):.6f} (A {min(K4, key=lambda k: k['dw1'])['A']}), max {max(k['dw1'] for k in K4):.6f} (A {max(K4, key=lambda k: k['dw1'])['A']}), positive in {sum(k['dw1'] > 0 for k in K4)} states; first {K4[0]['dw1']:.6f}, last {K4[-1]['dw1']:.6f}"
)
print(f"K4 dwe/dA: min {min(k['dwe'] for k in K4):.7f}, max {max(k['dwe'] for k in K4):.7f}")
print(
    f"reduced Hessian diagonal: first entry {min(d[0] for d in dg):.4f} to {max(d[0] for d in dg):.4f}; last entry {min(d[1] for d in dg):.3e} to {max(d[1] for d in dg):.3e}"
)
print(
    f"centre signal D(0) - C: first {sig[0]:.4e}, minimum {sig[j]:.4e} at A {A_l[j]}, last {sig[-1]:.4e}, all non-positive {all(x <= 0 for x in sig)}; off-atom maximum first {off[0]:.4e}, last {off[-1]:.4e}, all non-positive {all(x <= 0 for x in off)}"
)
print(
    f"innermost pair mass w_1: first {w1[0]:.7f}, last {w1[-1]:.9f}, strictly increasing {all(b > a for a, b in zip(w1, w1[1:]))}"
)
print(
    f"lattice slot 2 u_1/(u_2 - u_1): first {slot[0]:.4f}, last {slot[-1]:.4f}; law mu_1/mu_bar: first {law[0]:.4f}, last {law[-1]:.5f}"
)
print("END q954 READINGS (no verdict)")
