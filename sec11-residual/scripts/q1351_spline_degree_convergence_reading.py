"""q1351 (8 Oct 2026, EXPLORATORY reading, amendment 2121; no item, no verdict): convergence of the prediction a_med of the capacity with the spline degree of the counting function (run through tight_run.py). q1350 found that the period-4 oscillation of the relative misfit of the closure, with amplitude 0.039 (degree 5), 0.045 (7), 0.017 (9), 0.004 (11), 0.001 (13) at K_b 26-33, vanishes with the degree. This reading prints a_med(degree) - a_med(degree 13) in units of 1e-12 for the degrees 5, 7, 9, 11, 15, 17, 19 at K_b = 20, 26, 30, 34, 38, 42, 46, 50, 54, 59, and the same for the hard-wall stage k = 0.25 (K_b 20-45).
Control (first): degree 5 reproduces the registered quintic a_med of q1250 (QQ.root) to 1e-12 at K_b 36; the root in a bracket 15 to 80 equals the root in the bracket 1 to 140 for degree 5.
Usage: py tight_run.py q1351_spline_degree_convergence_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import q1342_hardwall_refit_tight_reading as P
B = P.B
QQ, Z = B.QQ, B.Z
C = Z.C
nhw = B.nhw
from scipy.optimize import brentq as _bq


def root2(Nf, K):
    z = lambda a: 2 * Nf(a) - K - 1 / 6
    a0 = _bq(z, 5.0, 55.0, xtol=1e-12)
    f = lambda a: z(a) - float(QQ.T3(1.0 / (4 * np.pi * (Nf(a + QQ.H) - Nf(a - QQ.H)) / (2 * QQ.H)) ** 2))
    return _bq(f, a0 - 2.5, a0 + 2.5, xtol=1e-14)


N5 = QQ.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, 5)
ctl = abs(root2(N5, 36) - QQ.root(N5, 36)) < 1e-12
print(f"control (degree 5, bracket 15-80 equals QQ.root at K_b 36): {ctl}")
if not ctl:
    print("READING q1351: VOID (control)")
    sys.exit(2)
DEG = (5, 7, 9, 11, 13, 15, 17, 19)


def table(name, mk, Ks):
    ro = {d: {K: root2(mk(d), K) for K in Ks} for d in DEG}
    print(f"{name}: a_med(degree) - a_med(13) in 1e-12, columns K_b {list(Ks)}")
    for d in DEG:
        print(f"  degree {d:2d}: " + " ".join(f"{(ro[d][K] - ro[13][K]) / 1e-12:+9.2f}" for K in Ks))


table(
    "capacity",
    lambda d: QQ.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, d),
    (20, 26, 30, 34, 38, 42, 46, 50, 54, 59),
)
st = json.load(open("c52_hw025c.json"))["stages"]
import q1304_hardwall_family_blind as H4
from scipy.optimize import least_squares

NOF = H4.NOF
stg = [s for s in st if s["D"] == 92.0 and s["conv"]][-1]
dd = np.array(stg["d"][: 1 + stg["nfree"]])
jj = np.arange(1, len(dd) + 1) - 0.5
mm = (dd >= 4) & (dd <= 30)
law = tuple(
    least_squares(lambda q: np.array([NOF(t, q[0], q[1]) for t in dd[mm]]) - jj[mm], [0.0558, 0.5482]).x
)
table(
    "hard wall k = 0.25",
    lambda d: QQ.make(lambda a: NOF(a, *law), dd[dd <= 72.0], d),
    (20, 26, 30, 34, 38, 42, 45),
)
print("READING q1351 (exploratory, no verdict)")
