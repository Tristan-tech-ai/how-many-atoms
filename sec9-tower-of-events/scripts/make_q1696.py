"""make_q1696 (10 Oct 2026, plan_c28_ladder.md): generator of the copies q1696_ball_rows_ng.py (copy of q1583_ball_rows.py) and q1697_krawczyk_ng.py (copy of q1598_krawczyk_free90_120.py).
The frozen files q1583 and q1598 are only read. The copy of q1583 gets the closed forms of f22, f24, f26 and f28 (strings copied from q1557_stdout_ORD28.txt, lines 14 to 17) and takes the order NG from the environment variable Q1696_NG (default 10, the old value).
The copy of q1598 reads the copy of q1583 and writes its z files under new names. Every replacement must match exactly once, otherwise the generator stops.
Usage: py make_q1696.py [OUTDIR [TAG [--corrupt20|--corrupt26]]]; the corrupt options change one digit of the f20 coefficient (used by NG = 10) or of the f26 string (used from NG = 13), for the selftest of q1699.
"""

import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SG = os.path.join(HERE, "..", "small_gap_q206")
ORD = os.path.join(SG, "q1557_stdout_ORD28.txt")


def forms(ns=(22, 24, 26, 28)):
    txt = open(ORD, encoding="utf-8").read()
    out = {}
    for n in ns:
        m = re.search(r"^\s*f%d = (.*?)\s{2,}= (\S+)\s*$" % n, txt, re.M)
        assert m, "f%d not found" % n
        out[n] = (m.group(1).strip(), m.group(2))
    return out


def once(s, a, b):
    assert s.count(a) == 1, "pattern found %d times: %s" % (s.count(a), a[:60])
    return s.replace(a, b)


def build(outdir=SG, tag="", corrupt=None):
    f = forms()
    if corrupt == "26":
        e = f[26][0]
        assert "35634047" in e
        f[26] = (e.replace("35634047", "35634048"), f[26][1])
    t1 = open(os.path.join(SG, "q1583_ball_rows.py"), encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in t1 else "\n"
    flo = nl.join("f%d = %s" % (n, re.sub(r"\bpi\b", "PI", f[n][0])) for n in (22, 24, 26, 28)) + nl
    arb = nl.join("f%da = %s" % (n, re.sub(r"\bpi\b", "PIA", f[n][0])) for n in (22, 24, 26, 28)) + nl
    t1 = once(
        t1,
        "F = [1.0, k1, k4, c6, c8, c10, f12, f14, f16, f18, f20]; NG = 10",
        flo
        + "F = [1.0, k1, k4, c6, c8, c10, f12, f14, f16, f18, f20, f22, f24, f26, f28]; NG = int(os.environ.get('Q1696_NG', '10'))",
    )
    t1 = once(
        t1,
        "Fa = [arb(1), k1a, k4a, c6a, c8a, c10a, f12a, f14a, f16a, f18a, f20a]",
        arb + "Fa = [arb(1), k1a, k4a, c6a, c8a, c10a, f12a, f14a, f16a, f18a, f20a, f22a, f24a, f26a, f28a]",
    )
    if corrupt == "20":
        t1 = once(
            t1.replace("64764727 / (190253629440 * PI ** 12)", "64764728 / (190253629440 * PI ** 12)", 1),
            "64764727 / (190253629440 * PIA ** 12)",
            "64764728 / (190253629440 * PIA ** 12)",
        )
    t1 = (
        "# q1696 (10 Oct 2026): copy of q1583_ball_rows.py with f22, f24, f26, f28 and the order NG from the environment Q1696_NG (default 10 = the old law). Made by scratch/make_q1696.py."
        + nl
        + t1
    )
    n1 = "q1696%s_ball_rows_ng.py" % tag
    n2 = "q1697%s_krawczyk_ng.py" % tag
    t2 = open(os.path.join(SG, "q1598_krawczyk_free90_120.py"), encoding="utf-8", newline="").read()
    t2 = t2.replace('os.path.join(here, "q1583_ball_rows.py")', 'os.path.join(here, "%s")' % n1)
    assert t2.count(n1) == 2, t2.count(n1)
    t2 = once(t2, 'types.ModuleType("q1583m")', 'types.ModuleType("q1696m")')
    t2 = once(t2, 'os.path.join(here, "q1598_z_N%d.json" % N)', 'os.path.join(here, "q1697_z_N%d.json" % N)')
    t2 = once(
        t2, 'os.path.join(here, "q1598_hist_N%d.json" % N)', 'os.path.join(here, "q1697_hist_N%d.json" % N)'
    )
    t2 = once(
        t2, 'os.path.join(here, "q1598_res_N%d.json" % N)', 'os.path.join(here, "q1697_res_N%d.json" % N)'
    )
    t2 = (
        "# q1697 (10 Oct 2026): copy of q1598_krawczyk_free90_120.py that reads %s and writes q1697_* files. Made by scratch/make_q1696.py.\n"
        % n1
        + t2
    )
    os.makedirs(outdir, exist_ok=True)
    for name, txt in ((n1, t1), (n2, t2)):
        open(os.path.join(outdir, name), "w", encoding="utf-8", newline="").write(txt)
    return os.path.join(outdir, n1), os.path.join(outdir, n2)


if __name__ == "__main__":
    a = sys.argv[1:]
    outdir = a[0] if a else SG
    tag = a[1] if len(a) > 1 else ""
    c = "20" if "--corrupt20" in a else ("26" if "--corrupt26" in a else None)
    print("written:", *build(outdir, tag, c))
