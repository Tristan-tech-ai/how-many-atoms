"""q1349 (8 Oct 2026, EXPLORATORY reading, amendment 2121; no item, no verdict): is there an oscillation of period 4 in K_b in the relative misfit of the closure R / Da = c0 - C1 / v^2 (C1 = 146.413) once the instrument is repaired? Amendment 2035 (d) recorded 'a pattern of period four (- + + - - + + - - +), magnitudes 0.054 falling to 0.012' for the quintic alternation of the capacity, not explained; with the repaired a_med the misfit of the capacity at K_b 26 to 37 reads +0.054, -0.024, -0.049, +0.001, +0.024, -0.014, -0.026, -0.001, +0.011, -0.009, -0.015, -0.002 (q1344).
This reading (run through tight_run.py) prints, for the capacity with spline degrees 5, 7 and 9 and for the hard-wall exponents with at least 15 events, the relative misfit m(K_b) = (R / Da) / (c0 - C1 / v^2) - 1 with c0 fitted on K_b >= 36 (capacity) or K_b >= 30 (hard wall) at fixed C1, for K_b 14 to 45; then for each series the amplitude and phase of a fit m = a cos(pi K_b / 2) + b sin(pi K_b / 2) over windows of eight consecutive K_b, and the correlation of the sign pattern across series.
Interpolant rule (spline degrees 5, 7, 9): the capacity misfit must keep its sign pattern for all three degrees.
Control (first): a synthetic series m = 0.05 exp(-(K_b - 26) / 5) cos(pi (K_b - 26) / 2) is recovered by the window fit with amplitude 0.05 +- 5e-3 at the first window and a phase error below 0.1 rad.
Usage: py tight_run.py q1349_period4_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import q1342_hardwall_refit_tight_reading as P
B = P.B
QQ, Z, Da = B.QQ, B.Z, B.Da
C = Z.C
C1 = 146.413


def winfit(Ks, m, w=8):
    out = []
    Ks = np.array(Ks, float)
    m = np.array(m, float)
    for i in range(0, len(Ks) - w + 1):
        s = slice(i, i + w)
        if Ks[s][-1] - Ks[s][0] != w - 1:
            continue
        A = np.column_stack([np.cos(np.pi * Ks[s] / 2), np.sin(np.pi * Ks[s] / 2), np.ones(w)])
        c, *_ = np.linalg.lstsq(A, m[s], rcond=None)
        out.append(
            (
                int(Ks[s][0]),
                int(Ks[s][-1]),
                float(np.hypot(c[0], c[1])),
                float(np.arctan2(c[1], c[0])),
                float(c[2]),
            )
        )
    return out


Kt = np.arange(26, 46)
syn = 0.05 * np.exp(-(Kt - 26) / 5) * np.cos(np.pi * (Kt - 26) / 2)
w0 = winfit(Kt, syn)[0]
ph_true = np.arctan2(
    0.05 * np.exp(0) * np.sin(-np.pi * 26 / 2), 0.05 * np.cos(-np.pi * 26 / 2)
)  # phase of cos(pi (K-26)/2) written as a cos(pi K/2) + b sin(pi K/2)
ctl = (
    abs(w0[2] - 0.05 * np.mean(np.exp(-(np.arange(26, 34) - 26) / 5))) < 0.012
    and abs(np.angle(np.exp(1j * (w0[3] - ph_true)))) < 0.5
)
print(
    f"control (synthetic decaying period-4 series recovered: first window amplitude {w0[2]:.4f}, phase {w0[3]:.3f} against {ph_true:.3f}): {ctl}"
)
if not ctl:
    print("READING q1349: VOID (control)")
    sys.exit(2)
series = {}
q16 = __import__("q1216_derived_series_lower_kb")
for deg in (5, 7, 9):
    capN = QQ.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, deg)
    rows = []
    for K, f in sorted(q16.CAP_FILES.items()):
        if 20 <= K <= 59:
            j = json.load(open(f + ".json"))
            a = j.get("A_t") or j.get("a_t")
            rows.append((K, (a - QQ.root(capN, K)) * 1e9, Z.v2f(capN, a)))
    K_ = np.array([r[0] for r in rows])
    R_ = np.array([r[1] for r in rows])
    V_ = np.array([r[2] for r in rows])
    y = R_ / np.array([Da(v) for v in V_])
    sel = K_ >= 36
    c0 = np.mean(y[sel] + C1 / V_[sel])
    m = y / (c0 - C1 / V_) - 1
    series[f"capacity deg {deg}"] = (K_, m, c0)
for k in sorted(P.data):
    ev = P.data[k]
    if len(ev) < 15:
        continue
    K_ = np.array([e[0] for e in ev])
    R_ = np.array([e[1] for e in ev])
    V_ = np.array([e[2] for e in ev])
    y = R_ / np.array([Da(v) for v in V_])
    sel = K_ >= 30
    c0 = np.mean(y[sel] + C1 / V_[sel])
    series[f"hard wall k {k}"] = (K_, y / (c0 - C1 / V_) - 1, c0)
print("relative misfit m(K_b), K_b 14 to 45 (blank = no event):")
for name, (K_, m, c0) in series.items():
    d = {int(K): v for K, v in zip(K_, m)}
    print(
        f"  {name:18s} c0 {c0:.4f}: "
        + " ".join(f"{d[K]:+.3f}" if K in d else "  .   " for K in range(14, 46))
    )
print("window fits (K_b first-last, amplitude, phase in rad, offset) over eight consecutive K_b:")
for name, (K_, m, c0) in series.items():
    wf = [w for w in winfit(K_, m) if w[0] >= 14]
    print(
        f"  {name:18s}: "
        + "; ".join(f"{a}-{b}: {amp:.3f}, {ph:+.2f}, {off:+.3f}" for a, b, amp, ph, off in wf[::2])
    )
print("READING q1349 (exploratory, no verdict)")
