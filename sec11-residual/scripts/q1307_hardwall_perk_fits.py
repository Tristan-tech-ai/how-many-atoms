"""q1307 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): per-exponent fits ln(R / Da) = ln sigma + ln(1 + g / v^2) of the hard-wall family (k = 0.02, 0.05, 0.1, 0.25, 0.5, 0.75) on all located events: K_b 25-37 of q1304 (k = 0.02 to 0.25), K_b 29-36 of q1305 (k = 0.5, 0.75), K_b 14-24, 39, 40 of q1306 (k = 0.1 to 0.75); R = (A_t - a_med) 1e9 with the quintic a_med of the hard-wall stage c52 (D = 92), v^2 from the same N_half at A_t, inclusion rule |a_quintic - a_septic| <= 0.02 R, K_b 38 excluded. Compared with the free-edge values of q1295 / q1302 (k = 0.75: 2.375, -12.46; 0.25: 1.154, -25.10; 0.1: 0.839, -47.02) and the capacity (1.769, -77.73, relative-misfit fit, rms 0.0971).
Control (first): synthetic recovery of (sigma, g) = (1.5, -20) and (0.9, -40) at the real v^2 of k = 0.75 to 1e-7; the same fit function applied to the free-edge events of k = 0.75 parsed from q1301b_out.txt reproduces the q1295 values (2.3752, -12.465) within 5e-3 (the printed v^2 of q1301b_out.txt has two decimals and R five; the first version of this bar, 1e-3, failed with g -12.452 against -12.465); event counts per exponent printed.
Usage: py q1307_hardwall_perk_fits.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, re, sys

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1304_hardwall_family_blind as H
import numpy as np
from scipy.optimize import least_squares

QQ, Z, Da, nhw, locr = H.QQ, H.Z, H.Da, H.nhw, H.locr
TAGS = {0.02: "hw002", 0.05: "hw005", 0.1: "hw01", 0.25: "hw025", 0.5: "hw05", 0.75: "hw075"}
SRC = {
    "u1304": (list(range(25, 38)), (0.02, 0.05, 0.1, 0.25)),
    "u1305": (list(range(29, 37)), (0.5, 0.75)),
    "u1306": (list(range(14, 25)) + [39, 40], (0.1, 0.25, 0.5, 0.75)),
}
data = {}
count = {}
for k, tag in TAGS.items():
    sj = f"c52_{tag}c.json"
    N5, _ = nhw(sj, 92.0, 5)
    N7, _ = nhw(sj, 92.0, 7)
    rows = {}
    n = 0
    for src, (Ks, ks) in SRC.items():
        if k not in ks:
            continue
        for K in Ks:
            A = locr(f"c61_{src}_{tag}_K{K}")
            if A is None or K in rows:
                continue
            n += 1
            am5, am7 = QQ.root(N5, K), QQ.root(N7, K)
            R = (A - am5) * 1e9
            if R > 0 and abs(am5 - am7) * 1e9 <= 0.02 * R:
                rows[K] = (R, Z.v2f(N5, A))
    data[k] = sorted((K, r, v) for K, (r, v) in rows.items())
    count[k] = n


def fit(R, v):
    d = np.array([Da(x) for x in v])
    y = np.log(np.array(R) / d)
    best = None
    for g0 in (-5.0, -15.0, -30.0, -60.0):
        try:
            r = least_squares(
                lambda q: q[1] + np.log(1 + q[0] / v) - y,
                [g0, 0.5],
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=4000,
            )
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(np.exp(best.x[1])), float(np.sqrt(np.mean(best.fun**2)))


v75 = np.array([x[2] for x in data[0.75]])
ok = True
for s0, g0 in ((1.5, -20.0), (0.9, -40.0)):
    yy = np.log(s0) + np.log(1 + g0 / v75)
    r = least_squares(
        lambda q: q[1] + np.log(1 + q[0] / v75) - yy, [-10.0, 0.0], xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    ok = ok and abs(r.x[0] - g0) < 1e-7 * abs(g0) and abs(np.exp(r.x[1]) - s0) < 1e-7 * s0
free75 = {}
cur = None
for l in open("q1301b_out.txt", encoding="utf-8"):
    m = re.match(r"k = ([\d.]+): \d+ events", l)
    if m:
        cur = float(m.group(1))
        continue
    if l.startswith("capacity"):
        cur = None
    m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", l)
    if m and cur == 0.75 and int(m.group(1)) != 38:
        free75[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
gf, sf, _ = fit([x[1] for x in free75.values()], np.array([x[0] for x in free75.values()]))
okf = abs(sf - 2.3752) < 5e-3 * 2.3752 and abs(gf + 12.465) < 5e-3 * 12.465
print(
    f"control: synthetic recovery {ok}; free-edge k = 0.75 from q1301b_out.txt gives sigma {sf:.4f}, g {gf:.3f} (q1295: 2.3752, -12.465) {okf}; events located per exponent {count}; used {{k: len}} {{ {', '.join(f'{k}: {len(v)}' for k, v in data.items())} }}"
)
if not (ok and okf):
    print("READING q1307: VOID (control)")
    sys.exit(2)
print("k      events  K_b range   v^2 range        sigma        g      rms(ln R)")
res = {}
for k in TAGS:
    d = data[k]
    R = [x[1] for x in d]
    v = np.array([x[2] for x in d])
    g, s, rms = fit(R, v)
    res[k] = (s, g, rms)
    print(
        f"{k:5.2f}  {len(d):5d}   {d[0][0]:2d}-{d[-1][0]:2d}    {v.min():6.1f}-{v.max():6.1f}   {s:8.4f} {g:9.3f}   {rms:.4f}"
    )
print(
    "free edge (q1295 / q1302): k = 0.75: 2.375 / -12.46 (0.0021); k = 0.25: 1.154 / -25.10 (0.0142); k = 0.1: 0.839 / -47.02 (0.0349); capacity 1.769 / -77.73 (0.0971)"
)
print("READING q1307 (exploratory, no verdict)")
