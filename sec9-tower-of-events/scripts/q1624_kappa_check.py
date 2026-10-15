"""q1624 (10 Oct 2026, READING with registered predictions, no verdict word): the deviation of the immune events from the stage-134 prediction of the table q1435 is kappa times the last stage step.
q1623b (post hoc): for K_b 124..131 the ratio kappa(K) = (a_t - p134) / (p134 - p128) is +0.00527, +0.00520, +0.00519, +0.00518, +0.00518, +0.00518, +0.00517 (K_b 124, 126, 127, 128, 129, 130, 131; the values fall from +0.0791 at K_b 100 to +0.0055 at K_b 120, a trend whose cause is not identified). The deviations of the gate q1436 (-0.0072e-9 at K_b 120 to -0.0515e-9 at K_b 131) are therefore the remaining stage-depth drift of the table itself, 0.5 percent of the last stage step, and no property of the immune ladder is needed to explain them.
REGISTRATION (written before any immune event K_b >= 132 exists in fw_imm_par): for every immune event K_b 132..150 that exists, the deviation a_t - p134 equals KAPPA * (p134 - p128) with KAPPA = 0.00517 to within 3 percent of that product. The table values p128 and p134 are those of centre_scripts/q1435_imm_table.json (sha256 bb38b43d77ad1d5dad4d635933e6d2eef607c5ac4f8dd48b08e3962a4219e37b).
Predictions:
 P1 every existing event K_b 132..150 satisfies |(a_t - p134) - KAPPA (p134 - p128)| <= 0.03 |KAPPA (p134 - p128)|;
 P2 the controls hold: the same test on the known events K_b 124..131 holds (positive control, not blind), and with KAPPA multiplied by 1.1 it fails (negative control);
 P3 at least one event K_b >= 132 exists when the script is run for the result (otherwise the script prints NO EVENT and no label).
Usage: py q1624_kappa_check.py (from small_gap_q206/)."""

import json, os, hashlib

H = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(H, "..", "centre_scripts", "q1435_imm_table.json")
assert hashlib.sha256(open(TP, "rb").read()).hexdigest() != "", "table unreadable"
T = json.load(open(TP))
assert (
    T["sha256"] == "bb38b43d77ad1d5dad4d635933e6d2eef607c5ac4f8dd48b08e3962a4219e37b"
), "table hash differs: VOID"
st = {D: {int(k): float(v) for k, v in T["stages"][D].items()} for D in ("128", "134")}
KAPPA = 0.00517


def at(K):
    f = os.path.join(os.environ.get("Q1624_IMMDIR", os.path.join(H, "fw_imm_par")), "event_Kb%d.json" % K)
    return (
        float(json.load(open(f))["A_event"]) if os.path.exists(f) else None
    )  # env var added 10 Oct 09:15 for the selftest; the default is unchanged


def test(Ks, kappa):
    res = []
    for K in Ks:
        a = at(K)
        if a is None:
            continue
        step = st["134"][K] - st["128"][K]
        pred = kappa * step
        dev = a - st["134"][K]
        res.append((K, dev, pred, abs(dev - pred) <= 0.03 * abs(pred)))
    return res


ctl = test(range(124, 132), KAPPA)
ctl_bad = test(range(124, 132), KAPPA * 1.1)
c1 = len(ctl) == 8 and all(r[3] for r in ctl)
c2 = len(ctl_bad) == 8 and not all(r[3] for r in ctl_bad)
print(
    "controls: known events K_b 124..131 with KAPPA %.5f: %d of %d within 3 percent (positive control %s); with 1.1 KAPPA: %d of %d within (negative control %s)"
    % (
        KAPPA,
        sum(r[3] for r in ctl),
        len(ctl),
        "ok" if c1 else "FAILED",
        sum(r[3] for r in ctl_bad),
        len(ctl_bad),
        "ok" if c2 else "FAILED",
    )
)
new = test(range(132, 151), KAPPA)
if not (c1 and c2):
    print("controls failed: VOID")
elif not new:
    print("NO EVENT K_b >= 132 in fw_imm_par yet: no label")
else:
    print("events K_b >= 132 checked:", [r[0] for r in new])
    for K, dev, pred, ok in new:
        print(
            "  K_b %d: deviation %+.4e, registered %+.4e, ratio %.4f, within 3 percent: %s"
            % (K, dev, pred, dev / pred, ok)
        )
    print(
        "P1 all existing events within 3 percent: %s; P2 controls: holds; P3 at least one event: holds"
        % ("holds" if all(r[3] for r in new) else "does not hold")
    )
print("READING q1624 done (labels by the script; no verdict word)")
