"""q1742 (10 Oct 2026, READING, no verdict word; REPAIR of q1741, whose first output q1741_out_first_run_window_defect.txt is kept): the window of E1 in q1741 ended at the next event line of the SAME K_b, and the log holds three locate passes for K_b 142 (lines 3084, 3107, 3139), so the odd-start line 3141 after the last pass fell outside it. Window now: from the first K_b 142 event line to the first event line of a larger K_b (the K_b 143 event). E1(b) is printed for that window (phrase count up to the K_b 143 event line) and for the end of the log. Registered text below is unchanged.
Original docstring of q1741: : the labels E1 to E4 of gpu_run7_prereg_2026-10-10.txt, applied by code to counts and line positions of the GPU walk run 7 (registered 12:36:40). Nothing is printed from the log except K_b numbers, event kinds, line numbers and counts; amplitudes are never printed (blind rule).
The predictions are those of the registration, copied here before the run of this script:
 E1 the run passes the birth K_b 142: the log has a 'birth: odd family start' line after the line of the K_b 142 event, and the count of 'could not be started' stays 6;
 E2 the folder holds at least 6 event files at the end (K_b 143 exists), or the run ends with the phrase count 7 at the next birth;
 E3 the 'done:' line reports fails 0 and max arb residual at most 4.5e-55;
 E4 the GPU stays below 85 C (health log) and the throttle flag stays 0.
E1 is evaluated in two parts, as registered: part (a) the odd-start line after the K_b 142 event; part (b) the count of the phrase, taken at the end of the log, so that a later birth that adds a phrase makes (b) fail by the letter; the count up to the next event line is printed as a fact.
Selftest (--selftest): a synthetic log without the odd-start line after the K_b 142 event must make E1(a) fail; a synthetic log with a seventh phrase before the next event must make E1(b) fail; a done line with fails 1 must make E3 fail; a temperature of 86 must make E4 fail.
Usage: py q1741_gpu_run7_eval.py [--selftest] (from small_gap_q206/; seconds)."""

import os, re, sys, csv, glob, datetime

H = os.path.dirname(os.path.abspath(__file__))
RUN_START = "12:36:40"


def parse_log(lines):
    ev = []
    ph = []
    od = []
    for i, ln in enumerate(lines):
        m = re.match(r"EVENT located: K_b = (\d+) \((birth|split)\)", ln)
        if m:
            ev.append((i + 1, int(m.group(1)), m.group(2)))
        if "could not be started" in ln:
            ph.append(i + 1)
        if "birth: odd family start" in ln:
            od.append(i + 1)
    return ev, ph, od


def e1(ev, ph, od, kb=142, expected_phrases=6):
    mine = [e for e in ev if e[1] == kb]
    if not mine:
        return False, False, None, None, False
    first = mine[0][0]
    nxt = [e[0] for e in ev if e[1] > kb]
    nxt = nxt[0] if nxt else 10**9
    a = any(first < o < nxt for o in od)
    upto = sum(1 for p in ph if p < nxt)
    b_window = upto == expected_phrases
    b_end = len(ph) == expected_phrases
    return a, b_window, upto, len(ph), b_end


def e3(done_line):
    m = re.search(r"fails (\d+), max arb residual ([0-9.e+-]+)", done_line)
    return bool(m) and int(m.group(1)) == 0 and float(m.group(2)) <= 4.5e-55


def e4(rows):
    t = [float(r["gpu_temp"]) for r in rows if r["gpu_temp"]]
    th = set(r["throttle"] for r in rows)
    return bool(t) and max(t) < 85 and th == {"0"}, (max(t) if t else None), len(rows)


def lab(x):
    return "holds" if x else "does not hold"


def selftest():
    good = [
        "EVENT located: K_b = 142 (birth) A_event = 1",
        "   birth: odd family start at A = 1",
        "EVENT located: K_b = 143 (split) A_event = 2",
    ]
    ev, ph, od = parse_log(good)
    ok1 = e1(ev, [1] * 6, od)[0]
    multi = [
        "EVENT located: K_b = 142 (birth) A_event = 1",
        "EVENT located: K_b = 142 (birth) A_event = 1",
        "EVENT located: K_b = 142 (birth) A_event = 1",
        "   birth: odd family start at A = 1",
        "EVENT located: K_b = 143 (split) A_event = 2",
    ]
    okm = e1(*[parse_log(multi)[0], [1] * 6, parse_log(multi)[2]])[0]
    late = [good[0], good[2], good[1]]
    a2 = e1(parse_log(late)[0], [1] * 6, parse_log(late)[2])[0]
    sev = good + ["new family could not be started; stop"]
    ev, ph, od = parse_log(sev)
    b2 = e1(ev, ph + [1] * 6, od)[4]
    bw = e1(ev, [1] * 6 + ph[:0], od)[1]
    c3 = (
        e3("done: wall 1 s, events written 7, newton calls 8, fails 1, max arb residual 4.5e-55") is False
        and e3("done: wall 1 s, events written 7, newton calls 8, fails 0, max arb residual 4.5e-55") is True
    )
    c4 = (
        e4([dict(gpu_temp="86", throttle="0")])[0] is False
        and e4([dict(gpu_temp="50", throttle="0")])[0] is True
    )
    print(
        "SELFTEST q1742: E1(a) holds on the good log %s and on a log with three locate passes %s; fails when the odd-start line follows the K_b 143 event %s; E1(b) at the end fails with 7 phrases %s and holds in the window with 6 %s; E3 cases %s; E4 cases %s"
        % (ok1, okm, not a2, not b2, bw, c3, c4)
    )
    allok = ok1 and okm and (not a2) and (not b2) and bw and c3 and c4
    print("SELFTEST " + ("PASS" if allok else "FAIL"))
    sys.exit(0 if allok else 1)


if "--selftest" in sys.argv:
    selftest()
W = os.path.join(H, "fw_gpu_cap2")
log = open(os.path.join(W, "walk.log"), encoding="utf-8", errors="replace").read().splitlines()
ev, ph, od = parse_log(log)
if not ev:
    print("VOID: no event line found")
    sys.exit(2)
std = open(os.path.join(H, "fw_gpu_cap2_stdout.txt"), encoding="utf-8", errors="replace").read().splitlines()
done = [l for l in std if l.startswith("done:")]
if not done:
    print("VOID: no done line in the stdout file")
    sys.exit(2)
print("event lines (line, K_b, kind) of the whole log: " + ", ".join("%d K_b %d %s" % e for e in ev))
print("lines of 'could not be started': %s; lines of 'birth: odd family start': %s" % (ph, od))
a, bw, upto, n, be = e1(ev, ph, od)
print(
    "E1(a) odd-start line after the K_b 142 event and before the K_b 143 event: %s; E1(b) phrase count up to the K_b 143 event line is 6 (value %d): %s; E1(b) phrase count at the end of the log is 6 (value %d): %s"
    % (lab(a), upto, lab(bw), n, lab(be))
)
files = sorted(glob.glob(os.path.join(W, "event_Kb*.json")))
ks = sorted(int(re.search(r"Kb(\d+)", os.path.basename(f)).group(1)) for f in files)
p2 = (len(ks) >= 6 and 143 in ks) or (n == 7)
print("E2 event files K_b %s (%d files); phrase count %d: %s" % (ks, len(ks), n, lab(p2)))
print("E3 %s: %s" % (done[-1].split(" (counts only")[0], lab(e3(done[-1]))))
rows = (
    [
        r
        for r in csv.DictReader(open(os.path.join(H, "..", "machine_health_log.csv")))
        if r["ts"][:10] == datetime.date.today().isoformat() and r["ts"][11:19] >= RUN_START
    ]
    if True
    else []
)
ok4, tmax, ns = e4(rows)
print(
    "E4 %d health samples since %s, maximum GPU temperature %s C, throttle values all 0: %s"
    % (ns, RUN_START, tmax, lab(ok4))
)
print("READING q1742 done (labels by the script; no verdict word)")
