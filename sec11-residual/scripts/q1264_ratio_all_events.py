"""q1264 (8 Oct 2026, a reading after the verdict of q1263, no item): the ratio dev_imm / dev_NPMLE of the quintic median-sum residuals at all events where both problems are known (immune: q624 table K_b 24 to 48, c19_u1231 K_b 29 to 37, c19_u1263 K_b 14 to 23; NPMLE: q607 files), at equal K_b and at equal v^2 (the NPMLE ln dev interpolated linearly in v^2 to the v^2 of the immune event).
Control (first): the immune dev at K_b 30 is +27.1162e-9 (q1250_out.txt) and at K_b 20 +1060.9230e-9 (q1263_out.txt), each within 0.001e-9 (K_b 30 has no stored NPMLE event).
Usage: py q1264_ratio_all_events.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np
import q1250_quintic_residual as Q
import q1252_order16_quintic as R

C, N, m = Q.C, Q.N, R.m
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
pi_ = np.array(sn["p"][: sn["nfree"]])
pi_ = pi_[pi_ <= 72]
Ni = Q.make(lambda a: Q.N4(a, (0.22573, 0.06806)), pi_, 5)
Nn = Q.make(lambda a: Q.N4(a, N.LAW), N.pn, 5)
ev = {int(e[1]): float(e[0]) for e in json.load(open("q624_immune_table.json")) if int(e[1]) >= 24}
for K in range(29, 38):
    ev[K] = json.load(open(f"c19_u1231_K{K}.json"))["a_t"]
for K in range(14, 24):
    ev[K] = json.load(open(f"c19_u1263_K{K}.json"))["a_t"]
npm = {K: gv(f) for K, f in m.NPM_FILES.items() if 6 <= K <= 52}
H = 1e-3
v2 = lambda Nf, a: (4 * np.pi * (Nf(a + H) - Nf(a - H)) / (2 * H)) ** 2
di = {K: (a - Q.root(Ni, K)) * 1e9 for K, a in ev.items() if K <= 48}
dn = {K: (a - Q.root(Nn, K)) * 1e9 for K, a in npm.items()}
vi = {K: v2(Ni, ev[K]) for K in di}
vn = {K: v2(Nn, npm[K]) for K in dn}
both = sorted(K for K in di if K in dn and dn[K] > 0 and di[K] > 0)
ok = abs(di[30] - 27.1162) < 0.001 and abs(di[20] - 1060.9230) < 0.001
print(f"control: immune dev K_b 30 {di[30]:+.4f}, K_b 20 {di[20]:+.4f} (q1263: +1060.9230): {ok}")
if not ok:
    print("READING q1264: VOID (control)")
    sys.exit(2)
rk = {K: di[K] / dn[K] for K in both}
ks = sorted(dn)
lx = np.array([vn[K] for K in ks if dn[K] > 0])
ly = np.array([np.log(dn[K]) for K in ks if dn[K] > 0])
o = np.argsort(lx)
rv = {}
for K in both:
    if lx[o].min() <= vi[K] <= lx[o].max():
        rv[K] = di[K] / np.exp(np.interp(vi[K], lx[o], ly[o]))
print(
    f"events with both problems known: {len(both)}; ratio at equal K_b: min {min(rk.values()):.3f} (K_b {min(rk, key=rk.get)}), max {max(rk.values()):.3f} (K_b {max(rk, key=rk.get)})"
)
print("ratio at equal K_b: " + ", ".join(f"{K}: {r:.3f}" for K, r in rk.items()))
print(
    f"ratio at equal v^2 ({len(rv)} events): min {min(rv.values()):.3f}, max {max(rv.values()):.3f}; "
    + ", ".join(f"{K}: {r:.3f}" for K, r in rv.items())
)
print(
    "v^2 imm / v^2 NPMLE at equal K_b: "
    + ", ".join(f"{K}: {vi[K] / vn[K]:.4f}" for K in both if K in (14, 20, 24, 30, 36, 42, 46))
)
print("READING q1264 (no verdict)")
