"""q1743 (10 Oct 2026, READING, no verdict word): the labels I1 to I5 of gpu_run8_prereg_2026-10-10.txt (sha256 7e6f0f22138a9ffb...), applied by code to line positions, signal values and counts of the GPU walk run 8 (started 13:11 on 10 Oct; the log walk.log had 4319 lines before it). Nothing is printed from the log except K_b numbers, event kinds, line numbers, signal values D''(0) of odd starts and counts; amplitudes are never printed (blind rule).
Predictions copied from the registration before the run ends:
 I1 after the K_b 144 event line of this run the log has a 'birth: odd family start' line with D''(0) negative and then a 'family K = 145 (odd)' line;
 I2 the first such D''(0) lies between -1.1e-23 and -0.55e-23;
 I3 the 'done:' line reports fails 0 and max arb residual at most 4.5e-55;
 I4 the GPU stays below 85 C (health log) and the throttle flag stays 0;
 I5 the folder holds the event file K_b 145 at the end, or the run ends with the phrase count 8 at the next birth.
Selftest (--selftest): a synthetic log with a positive D''(0) must make I1 fail; one with D''(0) = -3e-23 must make I1 hold and I2 fail; a log without the 'family K = 145' line must make I1 fail.
Usage: py q1743_gpu_run8_eval.py [--selftest] (from small_gap_q206/; seconds). Without the 'done:' line in fw_gpu_cap2_stdout.txt the script prints INCOMPLETE and exits with code 3.
"""

import os, re, sys, csv, glob, datetime

H = os.path.dirname(os.path.abspath(__file__))
START_LINE = 4319
RUN_START = "13:10:50"


def parse(lines, start=0):
    ev = []
    od = []
    fam = []
    ph = []
    for i, ln in enumerate(lines):
        if i < start:
            continue
        m = re.match(r"EVENT located: K_b = (\d+) \((birth|split)\)", ln)
        if m:
            ev.append((i + 1, int(m.group(1)), m.group(2)))
        m = re.search(r"birth: odd family start at A = [0-9.]+, wc [0-9.e+-]+, D''\(0\) ([-+0-9.e]+)", ln)
        if m:
            od.append((i + 1, float(m.group(1))))
        if re.match(r"family K = (\d+) \(odd\)", ln):
            fam.append((i + 1, int(re.match(r"family K = (\d+)", ln).group(1))))
        if "could not be started" in ln:
            ph.append(i + 1)
    return ev, od, fam, ph


def i12(ev, od, fam, kb=144):
    e = [x for x in ev if x[1] == kb]
    if not e:
        return False, False, None
    first = e[0][0]
    neg = [(l, v) for (l, v) in od if l > first and v < 0]
    if not neg:
        return False, False, None
    l, v = neg[0]
    ok = any(f[0] > l and f[1] == kb + 1 for f in fam)
    return ok, (-1.1e-23 <= v <= -0.55e-23), v


def e3(done_line):
    m = re.search(r"fails (\d+), max arb residual ([0-9.e+-]+)", done_line)
    return bool(m) and int(m.group(1)) == 0 and float(m.group(2)) <= 4.5e-55


def lab(x):
    return "holds" if x else "does not hold"


def selftest():
    mk = lambda v, fam: [
        "EVENT located: K_b = 144 (birth) A_event = 1",
        "   birth: odd family start at A = 1.5, wc 3.2e-06, D''(0) %s" % v,
    ] + (["family K = 145 (odd) from A = 1"] if fam else [])
    r1 = i12(*parse(mk("2.4e-18", True))[:3])
    r2 = i12(*parse(mk("-3e-23", True))[:3])
    r3 = i12(*parse(mk("-8e-24", False))[:3])
    r4 = i12(*parse(mk("-8e-24", True))[:3])
    ok = (
        (r1[0] is False)
        and (r2[0] is True and r2[1] is False)
        and (r3[0] is False)
        and (r4[0] is True and r4[1] is True)
    )
    print(
        "SELFTEST q1743: positive signal -> I1 fails %s; -3e-23 -> I1 holds, I2 fails %s; no family line -> I1 fails %s; -8e-24 -> I1 and I2 hold %s"
        % (r1[0] is False, r2[0] is True and r2[1] is False, r3[0] is False, r4[0] is True and r4[1] is True)
    )
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if "--selftest" in sys.argv:
    selftest()
W = os.path.join(H, "fw_gpu_cap2")
log = open(os.path.join(W, "walk.log"), encoding="utf-8", errors="replace").read().splitlines()
std = open(os.path.join(H, "fw_gpu_cap2_stdout.txt"), encoding="utf-8", errors="replace").read().splitlines()
done = [l for l in std if l.startswith("done:")]
if not done:
    print("INCOMPLETE: no done line yet")
    sys.exit(3)
ev, od, fam, ph = parse(log, START_LINE)
print("event lines of run 8 (line, K_b, kind): " + ", ".join("%d K_b %d %s" % e for e in ev))
print(
    "odd-start lines of run 8 (line, D''(0)): "
    + ", ".join("%d %.3e" % x for x in od)
    + "; family lines: "
    + ", ".join("%d K %d" % f for f in fam)
)
okf, ok2, v = i12(ev, od, fam)
print("I1 negative D''(0) odd start after the K_b 144 event, then a family K = 145 line: %s" % lab(okf))
print(
    "I2 first negative D''(0) %s between -1.1e-23 and -0.55e-23: %s"
    % (("%.3e" % v) if v is not None else "none", lab(ok2))
)
print("I3 %s: %s" % (done[-1].split(" (counts only")[0], lab(e3(done[-1]))))
allph = sum(1 for ln in log if "could not be started" in ln)
rows = [
    r
    for r in csv.DictReader(open(os.path.join(H, "..", "machine_health_log.csv")))
    if r["ts"][:10] == datetime.date.today().isoformat() and r["ts"][11:19] >= RUN_START
]
t = [float(r["gpu_temp"]) for r in rows if r["gpu_temp"]]
th = set(r["throttle"] for r in rows)
print(
    "I4 %d health samples since %s, maximum GPU temperature %s C, throttle values %s: %s"
    % (len(rows), RUN_START, max(t) if t else None, sorted(th), lab(bool(t) and max(t) < 85 and th == {"0"}))
)
ks = sorted(
    int(re.search(r"Kb(\d+)", os.path.basename(f)).group(1))
    for f in glob.glob(os.path.join(W, "event_Kb*.json"))
)
print("I5 event files K_b %s; phrase count %d: %s" % (ks, allph, lab(145 in ks or allph == 8)))
print("READING q1743 done (labels by the script; no verdict word)")
