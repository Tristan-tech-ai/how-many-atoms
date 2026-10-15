"""q1715 (10 Oct 2026, READING, no verdict word): is the exponential fall of the inner deviation delta_k of q1708_z_NG14_N90.json (ratio 0.70 per atom for k = 12..30, reading 170) the inward mode of the linearised bulk comb? Per-depth rate_k = ln(delta_k / delta_(k+1)) / (d_(k+1) - d_k) against the slow root kappa_in(rho) of the quadratic-background dispersion relation of centre_scripts/d41_dispersion.py (functions copied here), with rho = 1 / (d_(k+1) - d_k).
NOT BLIND: the ratios 0.699..0.709 per atom were read (reading 170), and the predictions below come from a hand estimate with the table of d41_dispersion_out.txt (kappa_in = 0.467 at rho = 0.90, about 0.52 per atom at spacing 1/rho).
Predictions:
 P1 rate_k / kappa_in(rho_k) lies between 0.5 and 0.9 for every k = 16..26;
 P2 that ratio varies by less than 10 percent over k = 16..26 (largest over smallest below 1.10);
 P3 rate_k is not equal to Re kappa_out (the complex pair) within 20 percent: rate_k / Re kappa_out lies outside [0.8, 1.2] for at least eight of the eleven values of k = 16..26.
Control: kappa_in at rho = 0.90 equals the table value 0.4672624926 of d41_dispersion_out.txt to 1e-8.
Usage: py q1715_inner_rate_vs_dispersion.py (seconds; needs q1708_z_NG14_N90.json)."""

import os, sys, json
import numpy as np
import mpmath as mp

mp.mp.dps = 50
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
pi = mp.pi
s0 = 1 / (16 * pi**2)
k2 = 1 / (8 * pi**2)
k4 = 1 / (64 * pi**4) - 1 / (96 * pi**2)
c6 = 1 / (512 * pi**6) - 13 / (768 * pi**4)
c8 = 1 / (4096 * pi**8) - 17 / (1536 * pi**6) - 1 / (3072 * pi**4)
c10 = 1 / (32768 * pi**10) - 29 / (6144 * pi**8) - 109 / (368640 * pi**6)


def f(r):
    return 1 + k2 / r**2 + k4 / r**4 + c6 / r**6 + c8 / r**8 + c10 / r**10


def rhop(r):
    return s0 * f(r) / r**2


def disp_quad(kap, rho, sgn):
    v = 4 * pi * rho
    be = pi * rhop(rho)
    a = 1 / (1 - 4j * be)
    k = sgn * 1j * kap
    return mp.exp(-a * (v * k + k * k)) + mp.exp(mp.conj(a) * (v * k - k * k)) - 2


def kin(rho):
    """slow real root kappa_in > 0 of the quadratic-background relation: the first sign change of the real function g(kappa) = Re disp_quad on a grid, refined by bisection (the relation has a trivial double root at 0)"""
    g = lambda x: mp.re(disp_quad(x, rho, +1))
    xs = [mp.mpf(i) / 200 for i in range(40, 400)]
    prev = g(xs[0])
    for a, b in zip(xs[:-1], xs[1:]):
        gb = g(b)
        if prev * gb < 0:
            return mp.findroot(g, (a, b), solver="anderson", tol=1e-30, verify=False)
        prev = gb
    return mp.mpf("nan")


def kout(rho):
    v = 4 * pi * rho
    try:
        z = mp.findroot(
            lambda w: disp_quad(w, rho, -1), 2 * pi / (v - 1j * mp.sqrt(2)), tol=1e-30, verify=False
        )
    except Exception:
        return mp.mpf("nan")
    return mp.re(z) if abs(disp_quad(z, rho, -1)) < 1e-20 else mp.mpf("nan")


ctl = abs(kin(mp.mpf("0.90")) - mp.mpf("0.4672624926")) < 1e-8
D = json.load(open(os.path.join(here, "q1708_z_NG14_N90.json")))
N = D["N"]
os.environ["Q1696_NG"] = str(D["NG"])
import io, contextlib, importlib.util

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("q1697m", os.path.join(here, "q1697_krawczyk_ng.py"))
q = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(q)
from flint import arb

q.NU, q.D0 = D["nu_used"], D["d0_used"]
dt, wt = q.formal_tail(0, N + 1)
z = [arb(s) for s in D["z"]]
d = np.array([float(z[k].mid()) for k in range(N)])
delta = np.array([float((z[k] - dt[k]).mid()) for k in range(N)])
rows = []
for k in range(14, 29):
    dd = d[k] - d[k - 1]
    rho = 1 / dd
    rate = float(mp.log(delta[k - 1] / delta[k])) / dd
    rows.append((k, rho, rate, float(kin(mp.mpf(rho))), float(kout(mp.mpf(rho)))))
print(
    "k   rho    rate per depth   kappa_in   Re kappa_out   rate/kappa_in   rate/Re kappa_out   (rate uses atoms k and k+1)"
)
for k, rho, rate, ki, ko in rows:
    print(
        "%2d  %.3f   %.4f          %.4f     %.4f         %.3f           %.3f"
        % (k, rho, rate, ki, ko, rate / ki, rate / ko)
    )
sel = [r for r in rows if 16 <= r[0] <= 26]
ri = [r[2] / r[3] for r in sel]
ro = [r[2] / r[4] for r in sel]
lab = lambda q_: "holds" if q_ else "does not hold"
p1 = all(0.5 <= x <= 0.9 for x in ri)
p2 = max(ri) / min(ri) < 1.10
p3 = sum(1 for x in ro if not (0.8 <= x <= 1.2)) >= 8
print(
    "range of rate/kappa_in over k = 16..26: %.3f to %.3f (largest over smallest %.3f); rate/Re kappa_out: %.3f to %.3f"
    % (min(ri), max(ri), max(ri) / min(ri), min(ro), max(ro))
)
print("P1 %s; P2 %s; P3 %s; control: %s" % (lab(p1), lab(p2), lab(p3), lab(ctl)))
print("READING q1715 done (labels by the script; no verdict word)")
