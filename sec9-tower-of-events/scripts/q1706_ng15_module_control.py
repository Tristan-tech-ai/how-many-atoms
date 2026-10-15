"""q1706 (10 Oct 2026, READING, no verdict word): control of the copies q1703_ball_rows_ng15.py and q1704_krawczyk_ng15.py (generator scratch/make_q1703.py) that carry the bulk law through c30 (NG = 15).
Predictions written before the run:
 P1 with NG = 14 the arb formal depths of the new module for k = 1..121 equal those of the q1697 module (NG = 14) to 1e-50;
 P2 the arb coefficients f22, f24, f26, f28, f30 of the new module equal the decimals printed in q1557_stdout_ORD30.txt (16 digits) to a relative 1e-14, and len(Fa) = 16, len(ga) = 16 at NG = 15;
 P3 with NG = 15 the arb formal depths differ from those of NG = 14 by between 1e-12 and 5e-11 (largest over k = 30..60);
 P4 selftest: a copy with one digit of the f26 string changed makes the P2 check fail (relative difference above 1e-14).
Usage: py q1706_ng15_module_control.py (about 1 to 3 minutes, no Newton solve)."""

import os, sys, io, contextlib, importlib.util, time

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
GEN = os.path.join(here, "..", "scratch")
sys.path.insert(0, GEN)
import make_q1703 as G
from flint import arb

t0 = time.time()
NU, D0 = -0.093773586, 0.001159841


def load(path, ng, name):
    os.environ["Q1696_NG"] = str(ng)
    os.environ["Q1703_NG"] = str(ng)
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
    return mod.formal_tail(0, 121)[0]


def maxdiff(a, b, lo=0, hi=121):
    return max(float((x - y).abs_upper()) for x, y in zip(a[lo:hi], b[lo:hi]))


dec = {n: v[1] for n, v in G.forms().items()}


def p2_check(mod):
    mm = mod.m
    idx = {22: 11, 24: 12, 26: 13, 28: 14, 30: 15}
    rel = {n: float(abs(mm.Fa[idx[n]] / arb(dec[n].lower()) - 1).abs_upper()) for n in idx}
    return max(rel.values()), rel, (len(mm.Fa), len(mm.ga))


Q97 = os.path.join(here, "q1697_krawczyk_ng.py")
Q04 = os.path.join(here, "q1704_krawczyk_ng15.py")
a14 = depths(load(Q97, 14, "q1697_ng14"))
n14 = load(Q04, 14, "q1704_ng14")
d14 = depths(n14)
p1v = maxdiff(d14, a14)
print(
    "P1 largest difference NG = 14, new module against q1697: %.3e [%.0fs]" % (p1v, time.time() - t0),
    flush=True,
)
n15 = load(Q04, 15, "q1704_ng15")
d15 = depths(n15)
p2v, rel, lens = p2_check(n15)
print(
    "P2 relative differences of f22, f24, f26, f28, f30 against the printed decimals: %s; len(Fa), len(ga) = %s"
    % (", ".join("%.1e" % rel[n] for n in sorted(rel)), lens),
    flush=True,
)
print("f30 = %s (reading 157: -7.25424082052e-12)" % n15.m.Fa[15].str(12, radius=False))
p3v = maxdiff(d15, d14, 29, 60)
print("P3 largest |d(NG=15) - d(NG=14)| over k = 30..60: %.3e [%.0fs]" % (p3v, time.time() - t0), flush=True)
tmp = os.path.join(GEN, "q1706_selftest_copies")
G.build(tmp, "x26", "26")
s26 = load(os.path.join(tmp, "q1704x26_krawczyk_ng15.py"), 15, "q1704x26")
s4 = p2_check(s26)[0]
print(
    "P4 corrupted f26 digit: P2 relative difference %.3e (must exceed 1e-14) [%.0fs]"
    % (s4, time.time() - t0),
    flush=True,
)
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 %s; P2 %s; P3 %s; P4 %s"
    % (lab(p1v < 1e-50), lab(p2v < 1e-14 and lens == (16, 16)), lab(1e-12 <= p3v <= 5e-11), lab(s4 > 1e-14))
)
print("READING q1706 done (labels by the script; no verdict word)")
