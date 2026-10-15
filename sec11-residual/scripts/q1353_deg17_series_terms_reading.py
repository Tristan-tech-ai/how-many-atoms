"""q1353 (8 Oct 2026, EXPLORATORY reading, amendment 2121; no item, no verdict): series terms of the closure R / Da = c0 + c1 x + c2 x^2 + c3 x^3 (x = 1 / v^2) on the degree-17 events of q1352 (q1352_events.json, repaired instrument): (a) the capacity K_b 20-59 with 2, 3, 4 terms; (b) the hard-wall exponents with at least 8 events with 2 and 3 terms; (c) joint fits with c0 per exponent and ONE common c1, or common c1 and c2 (capacity included as k = 0). Relative-misfit least squares as in q1319/q1320.
Control (first): synthetic recovery of (c0, c1, c2) = (1.9, -144, 900) by the 3-term fit at the real capacity v^2 to 1e-8; the 2-term capacity fit K_b 26-59 reproduces q1352 (c0 1.8455, c1 -144.77) to 2e-3.
Usage: py tight_run.py q1353_deg17_series_terms_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import q1321_hardwall_common_c1_blind as B
Da = B.Da
ev = json.load(open("q1352_events.json"))
hw = {float(k): v for k, v in ev["hard_wall_deg17"].items()}
cap = ev["capacity_deg17"]


def fit(R, v, nt, fixed=None):
    R = np.array(R)
    v = np.array(v)
    d = np.array([Da(x) for x in v])
    x = 1.0 / v
    A = np.array([[(x[i] ** j) * d[i] / R[i] for j in range(nt)] for i in range(len(R))])
    c, *_ = np.linalg.lstsq(A, np.ones(len(R)), rcond=None)
    mod = sum(c[j] * x**j for j in range(nt)) * d
    return c, float(np.sqrt(np.mean(np.log(mod / R) ** 2)))


def joint(sets, common):
    """c0 per set; common coefficients c1..c_common (len common)"""
    ns = len(sets)
    rows = []
    rhs = []
    for i, (R, v) in enumerate(sets):
        R = np.array(R)
        v = np.array(v)
        d = np.array([Da(x) for x in v])
        x = 1.0 / v
        for j in range(len(R)):
            a = np.zeros(ns + common)
            a[i] = d[j] / R[j]
            for p in range(common):
                a[ns + p] = x[j] ** (p + 1) * d[j] / R[j]
            rows.append(a)
            rhs.append(1.0)
    c, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    rms = []
    for i, (R, v) in enumerate(sets):
        R = np.array(R)
        v = np.array(v)
        d = np.array([Da(x) for x in v])
        mod = (c[i] + sum(c[ns + p] / v ** (p + 1) for p in range(common))) * d
        rms.append(float(np.sqrt(np.mean(np.log(mod / R) ** 2))))
    return c, rms


Kc = np.array([r[0] for r in cap])
Rc = np.array([r[1] for r in cap])
Vc = np.array([r[2] for r in cap])
vs = Vc[:20]
ds = np.array([Da(x) for x in vs])
cs, _ = fit((1.9 - 144 / vs + 900 / vs**2) * ds, vs, 3)
c26, _ = fit(Rc[Kc >= 26], Vc[Kc >= 26], 2)
ctl = bool(
    np.all(np.abs(cs / np.array([1.9, -144, 900]) - 1) < 1e-8)
    and abs(c26[0] - 1.8455) < 2e-3
    and abs(c26[1] + 144.77) < 2e-2
)
print(
    f"control (synthetic 3-term recovery; capacity 2-term K_b 26-59 c0 {c26[0]:.4f}, c1 {c26[1]:.2f}): {ctl}"
)
if not ctl:
    print("READING q1353: VOID (control)")
    sys.exit(2)
print("(a) capacity, degree 17, K_b 20-59 (n %d): terms -> coefficients, rms of ln R" % len(Kc))
for nt in (2, 3, 4):
    c, r = fit(Rc, Vc, nt)
    print(f"   {nt} terms: " + ", ".join(f"{x:.4g}" for x in c) + f"; rms {r:.5f}")
print("(b) hard wall exponents with >= 8 events: 2 terms and 3 terms (c0, c1, c2), rms")
for k in sorted(hw):
    if len(hw[k]) < 8:
        continue
    R = [e[1] for e in hw[k]]
    v = [e[2] for e in hw[k]]
    c2, r2 = fit(R, v, 2)
    c3, r3 = fit(R, v, 3)
    print(
        f"   k {k:5.3f} (n {len(R)}): 2 terms {c2[0]:.4f} {c2[1]:8.2f} rms {r2:.5f} | 3 terms {c3[0]:.4f} {c3[1]:8.2f} {c3[2]:9.1f} rms {r3:.5f}; c1/c0 {c3[1] / c3[0]:7.2f}, c2/c0 {c3[2] / c3[0]:8.1f}"
    )
ks = [k for k in sorted(hw) if len(hw[k]) >= 8]
sets = [([e[1] for e in hw[k]], [e[2] for e in hw[k]]) for k in ks] + [(list(Rc), list(Vc))]
names = ks + ["cap"]
print("(c) joint fits with c0 per set (hard wall k >= 8 events, plus the capacity as the last set):")
for common in (1, 2, 3):
    c, rms = joint(sets, common)
    ns = len(sets)
    print(
        f"   common c1..c{common}: "
        + ", ".join(f"{x:.4g}" for x in c[ns:])
        + "; c0: "
        + ", ".join(f"{n}: {c[i]:.4f}" for i, n in enumerate(names))
        + "; rms: "
        + ", ".join(f"{r:.4f}" for r in rms)
    )
print("READING q1353 (exploratory, no verdict)")
