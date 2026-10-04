"""Registered centre births of paper 1 (Table of registered births): observed zero of the ring's centre excess against the
registered value, read from the registration log only.

The observed zero is the linear interpolation between the two ring values that bracket it, from the ring tables of Amendments
1443, 1447 (200 nodes), 1452, 1462, 1464; Amendment 1406 states its interpolated zero directly. Writes births_check.json.
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(os.path.dirname(HERE), "PREREG_Q32.md"), encoding="utf-8", errors="replace").read()
NUM = r"([-+]?\d\.\d+e[-+]?\d+)"


def blk(n):
    return re.search(r"^## Amendment %d\b.*?(?=^## Amendment )" % n, LOG, re.M | re.S).group(0)


def g(n, rx):
    return float(re.search(rx, blk(n), re.S).group(1))


def zero(x0, v0, x1, v1):
    return x0 + (x1 - x0) * (-v0) / (v1 - v0)


cases = {
    "capacity ellipse 2.8": (g(1406, r"Zero by linear interpolation (\d\.\d+)"), g(1406, r"registered (\d\.\d+), band")),
    "LFP ellipse 1.7": (zero(1.4038, g(1443, r"\| ring r\(0\) - C \| [-+\d.e]+ \| " + NUM),
                             1.4040, g(1443, r"\| ring r\(0\) - C \| [-+\d.e]+ \| [-+\d.e]+ \| " + NUM)),
                        g(1443, r"registered (\d\.\d+), band")),
    "NPMLE ellipse 2.1": (zero(1.7416, g(1447, r"200 nodes: " + NUM), 1.7417, g(1447, r"200 nodes: [-+\d.e]+\s+\(1\.7416\), " + NUM)),
                          g(1447, r"Registered (\d\.\d+), band")),
    "capacity mixed 0.2, 2.8": (zero(2.1756, g(1452, r"\| 2\.1756 \(K 28\) \| " + NUM), 2.1758, g(1452, r"\| 2\.1758 \(K 32\) \| " + NUM)),
                                g(1452, r"registered (\d\.\d+), band")),
    "LFP mixed 0.1, 1.7": (zero(1.4014, g(1462, r"\| 1\.4014 \(K 18\) \| " + NUM), 1.4016, g(1462, r"\| 1\.4016 \(K 18; K 20\) \| " + NUM)),
                           g(1462, r"registered (\d\.\d+), band")),
    "NPMLE mixed 0.2, 2.1": (zero(1.7362, g(1464, r"\| 1\.7362 \| " + NUM), 1.7364, g(1464, r"\| 1\.7364 \| " + NUM)),
                             g(1464, r"registered (\d\.\d+), band")),
}
RP = {"capacity ellipse 2.8": g(1406, r"optimum at rp (\d\.\d+) lies at rm"), "LFP ellipse 1.7": g(1443, r"filled ellipse rp (\d\.\d+)"),
      "NPMLE ellipse 2.1": g(1447, r"filled ellipse rp (\d\.\d+)"), "capacity mixed 0.2, 2.8": g(1452, r"lam 0\.2, rp (\d\.\d+)"),
      "LFP mixed 0.1, 1.7": g(1462, r"lam 0\.1, rp (\d\.\d+)"), "NPMLE mixed 0.2, 2.1": g(1464, r"lam 0\.2, rp (\d\.\d+)")}
out = {k: {"rp": RP[k], "observed": o, "registered": r, "miss": o - r} for k, (o, r) in cases.items()}
out["max_abs_miss"] = max(abs(v["miss"]) for v in out.values())
for k, v in out.items():
    print(k, v)
json.dump(out, open(os.path.join(HERE, "births_check.json"), "w", encoding="utf-8"), indent=1)
