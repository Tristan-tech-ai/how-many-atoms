"""q1699 (10 Oct 2026, READING, no verdict word, plan_c28_ladder.md): control of the copies q1696_ball_rows_ng.py and q1697_krawczyk_ng.py (generator scratch/make_q1696.py) that carry the bulk law through c20, c24 or c28.
Predictions written before the run (from the registration text of 11:27, with P3 corrected before any output existed, because f26 does not enter the law at NG = 10, so a changed digit in f26 cannot make a NG = 10 control fail):
 P1 with NG = 10 the arb formal depths of the new module for k = 1..121 (pair nu0 = -0.093773586, d0 = +0.001159841, q.formal_tail) equal those of the frozen q1598 module to 1e-50;
 P2 with NG = 12 the arb formal depths differ from the float depths dep[12] of q1613 by less than 1e-13 for k = 1..121;
 P3 selftest: S1 a copy with one digit of the f20 numerator changed makes the P1 check fail (largest difference above 1e-50); S2 a copy with one digit of the f26 string changed makes the P4 check fail;
 P4 with NG = 14 the arb coefficients f22, f24, f26, f28 of the module equal the decimals printed in q1557_stdout_ORD28.txt (16 digits) to a relative 1e-14, and len(Fa) = 15, len(ga) = 15;
 P5 with NG = 14 the arb formal depths differ from those of NG = 12 by between 1e-13 and 1e-10 (largest over k = 30..60), the size of the rest after the law through c24 (about 1e-11).
Usage: py q1699_ng_module_control.py (about 3 to 6 minutes, one process, no Newton solve)."""

import os, sys, io, re, contextlib, importlib.util, time

sys.dont_write_bytecode = True
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
GEN = os.path.join(here, "..", "scratch")
sys.path.insert(0, GEN)
import make_q1696 as G

t0 = time.time()
src = open(os.path.join(here, "q1613_rest_vs_next_orders.py"), encoding="utf-8").read()
src = src[: src.index("LIM = ")]
ns = {"__file__": os.path.join(here, "q1613_rest_vs_next_orders.py"), "__name__": "q1613ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, ns)
NU, D0, dep = ns["NU"], ns["D0"], ns["dep"]
QF = ns["q"]


def load_ng(path, ng, name):
    os.environ["Q1696_NG"] = str(ng)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["x"]
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def depths(mod):
    mod.NU, mod.D0 = NU, D0
    dt, wt = mod.formal_tail(0, 121)
    return dt


def maxdiff(a, b):
    return max(float((x - y).abs_upper()) for x, y in zip(a, b))


QF.NU, QF.D0 = NU, D0
dfrozen = QF.formal_tail(0, 121)[0]
print("frozen depths ready [%.0fs]" % (time.time() - t0), flush=True)
Q97 = os.path.join(here, "q1697_krawczyk_ng.py")
m10 = load_ng(Q97, 10, "q1697_ng10")
d10 = depths(m10)
p1v = maxdiff(d10, dfrozen)
print(
    "P1 largest difference NG = 10 against the frozen module: %.3e [%.0fs]" % (p1v, time.time() - t0),
    flush=True,
)
m12 = load_ng(Q97, 12, "q1697_ng12")
d12 = depths(m12)
fl = np.array([float(t) for t in d12])
p2v = float(np.abs(fl - dep[12]).max())
print(
    "P2 largest difference NG = 12 arb against the float dep[12] of q1613: %.3e [%.0fs]"
    % (p2v, time.time() - t0),
    flush=True,
)
m14 = load_ng(Q97, 14, "q1697_ng14")
d14 = depths(m14)
print("NG = 14 depths ready [%.0fs]" % (time.time() - t0), flush=True)
from flint import arb

dec = {n: v[1] for n, v in G.forms().items()}


def p4_check(mod):
    mm = mod.m
    idx = {22: 11, 24: 12, 26: 13, 28: 14}
    rel = {n: float(abs(mm.Fa[idx[n]] / arb(dec[n].lower()) - 1).abs_upper()) for n in idx}
    return max(rel.values()), rel, (len(mm.Fa), len(mm.ga))


p4v, rel, lens = p4_check(m14)
print(
    "P4 relative differences of f22, f24, f26, f28 against the printed decimals: %s; len(Fa), len(ga) = %s"
    % (", ".join("%.1e" % rel[n] for n in sorted(rel)), lens),
    flush=True,
)
print(
    "f26 = %s, f28 = %s (reading 139: -3.62175e-11, -1.57030e-11)"
    % (m14.m.Fa[13].str(10, radius=False), m14.m.Fa[14].str(10, radius=False))
)
p5v = max(float((a - b).abs_upper()) for a, b in zip(d14[29:60], d12[29:60]))
print("P5 largest |d(NG=14) - d(NG=12)| over k = 30..60: %.3e [%.0fs]" % (p5v, time.time() - t0), flush=True)
tmp = os.path.join(GEN, "q1699_selftest_copies")
G.build(tmp, "x20", "20")
G.build(tmp, "x26", "26")
s20 = load_ng(os.path.join(tmp, "q1697x20_krawczyk_ng.py"), 10, "q1697x20")
s1v = maxdiff(depths(s20), dfrozen)
s26 = load_ng(os.path.join(tmp, "q1697x26_krawczyk_ng.py"), 14, "q1697x26")
s2v = p4_check(s26)[0]
print(
    "S1 corrupted f20 digit: P1 difference %.3e (must exceed 1e-50); S2 corrupted f26 digit: P4 relative difference %.3e (must exceed 1e-14) [%.0fs]"
    % (s1v, s2v, time.time() - t0),
    flush=True,
)
lab = lambda p: "holds" if p else "does not hold"
print(
    "P1 %s; P2 %s; P3 %s (S1 %s, S2 %s); P4 %s; P5 %s"
    % (
        lab(p1v < 1e-50),
        lab(p2v < 1e-13),
        lab(s1v > 1e-50 and s2v > 1e-14),
        lab(s1v > 1e-50),
        lab(s2v > 1e-14),
        lab(p4v < 1e-14 and lens == (15, 15)),
        lab(1e-13 <= p5v <= 1e-10),
    )
)
print("READING q1699 done (labels by the script; no verdict word)")
