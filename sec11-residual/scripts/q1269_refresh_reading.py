"""q1269 (8 Oct 2026, a reading after a data refresh, no item): (a) differences between the refreshed 160-bit transitions (immune c19_u1267 K_b 36 to 48, NPMLE q607_u1268 K_b 34 to 48) and the stored ones (immune q624 table, NPMLE q607 families); (b) the ratios R_imm / R_NPMLE and R_cap / R_NPMLE at equal v^2 and the defect d = (R_imm - 2 R_NPMLE + R_cap) / (R_imm - R_NPMLE) with the refreshed values replacing the stored ones (quintic median-sum residuals as q1250).
Located (refresh): a_t not null, last residual below 1e-20, |a_t - last a| < 1e-9.
Control (first): the refreshed immune K_b 36 and 37 equal the c19_u1231 values (same code and settings, expected bit-identical, tolerance 1e-12) and the refreshed NPMLE K_b 34 equals the stored one within 1e-10. (A first version of this control used the q624 table value 28.0114454915, an old entry that is 1.8e-7 off; it voided the first run before any ratio was printed.)
Usage: py q1269_refresh_reading.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np
import q1250_quintic_residual as Q
import q1252_order16_quintic as R

C, N, m = Q.C, Q.N, R.m
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))


def loc(f):
    j = json.load(open(f + ".json"))
    hl = [x for x in j["hist"] if "res" in x and "a" in x]
    a = j.get("a_t")
    return a if (a is not None and hl and hl[-1]["res"] < 1e-20 and abs(a - hl[-1]["a"]) < 1e-9) else None


sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
pi_ = np.array(sn["p"][: sn["nfree"]])
pi_ = pi_[pi_ <= 72]
NL = {
    "capacity": (lambda a: C.Q.N_of(a, *C.CAPL), C.X),
    "npmle": (lambda a: Q.N4(a, N.LAW), N.pn),
    "immune": (lambda a: Q.N4(a, (0.22573, 0.06806)), pi_),
}
Nf = {k: Q.make(v[0], v[1], 5) for k, v in NL.items()}
old = {
    "capacity": {K: gv(f) for K, f in m.CAP_FILES.items() if 10 <= K <= 59},
    "npmle": {K: gv(f) for K, f in m.NPM_FILES.items() if 6 <= K <= 52},
}
im = {int(e[1]): float(e[0]) for e in json.load(open("q624_immune_table.json")) if int(e[1]) >= 24}
for K in range(29, 38):
    im[K] = json.load(open(f"c19_u1231_K{K}.json"))["a_t"]
for K in range(14, 24):
    im[K] = json.load(open(f"c19_u1263_K{K}.json"))["a_t"]
old["immune"] = {K: a for K, a in im.items() if K <= 48}
new = {n: dict(old[n]) for n in old}
nloc = {"immune": 0, "npmle": 0}
for K in range(36, 49):
    a = loc(f"c19_u1267_K{K}")
    if a is not None:
        new["immune"][K] = a
        nloc["immune"] += 1
for K in range(34, 49):
    a = loc(f"q607_u1268_K{K}")
    if a is not None:
        new["npmle"][K] = a
        nloc["npmle"] += 1
ok = (
    abs(new["immune"][36] - json.load(open("c19_u1231_K36.json"))["a_t"]) < 1e-12
    and abs(new["immune"][37] - json.load(open("c19_u1231_K37.json"))["a_t"]) < 1e-12
    and abs(new["npmle"][34] - old["npmle"][34]) < 1e-10
)
print(
    f"control: refreshed immune K_b 36 {new['immune'][36]:.12f}, NPMLE K_b 34 {new['npmle'][34]:.12f}: {ok}; located immune {nloc['immune']} of 13, NPMLE {nloc['npmle']} of 15"
)
if not ok:
    print("READING q1269: VOID (control)")
    sys.exit(2)
for n, ks in (("immune", range(36, 49)), ("npmle", range(34, 49))):
    print(
        f"{n} new - old (1e-9): "
        + ", ".join(
            f"{K}: {(new[n][K] - old[n][K]) * 1e9:+.3f}" for K in ks if K in old[n] and new[n][K] != old[n][K]
        )
    )
H = 1e-3
v2 = lambda n, a: (4 * np.pi * (Nf[n](a + H) - Nf[n](a - H)) / (2 * H)) ** 2


def tables(src):
    dev = {n: {K: (a - Q.root(Nf[n], K)) * 1e9 for K, a in src[n].items()} for n in NL}
    vv = {n: {K: v2(n, src[n][K]) for K in src[n]} for n in NL}
    return dev, vv


def interp(dev, vv, name, x, floor):
    ks = sorted(K for K in dev[name] if dev[name][K] > floor)
    xs = np.array([vv[name][K] for K in ks])
    ys = np.array([np.log(dev[name][K]) for K in ks])
    o = np.argsort(xs)
    return float(np.exp(np.interp(x, xs[o], ys[o]))) if xs.min() <= x <= xs.max() else None


for label, src in (("OLD", old), ("NEW", new)):
    dev, vv = tables(src)
    print(f"--- {label}: K_b, v^2, R_imm, R_NPMLE(v2), R_cap(v2), imm/NP, cap/NP, d")
    for K in range(30, 49):
        if K not in dev["immune"]:
            continue
        x = vv["immune"][K]
        rn = interp(dev, vv, "npmle", x, 0.0)
        rc = interp(dev, vv, "capacity", x, 0.09)
        if rn is None or rc is None:
            continue
        ri = dev["immune"][K]
        print(
            f"{K:3d} {x:6.1f} {ri:9.4f} {rn:9.4f} {rc:9.4f}  {ri / rn:.3f}  {rc / rn:.3f}  {(ri - 2 * rn + rc) / (ri - rn):+.3f}"
        )
print("READING q1269 (no verdict)")
