"""q1668 (10 Oct 2026, READING, no verdict word): labels for the missing births K_b 136 and 138 against the bars registered BEFORE the events exist, with a selftest. Differences are a_pred - A_event; the amplitudes are loaded inside the script and only differences are printed (BLIND RULE).
Bars, copied from q1662_prereg_136_138.txt (ladder pair, bulk law through c24, stored e_2..e_18) and q1663_prereg_136_138.txt (frozen refitted pair, same law and series; a_pred 77.388989108974 for K_b 136 and 78.257764318990 for K_b 138):
 L  ladder pair: |diff| < 3e-11 for each event; LQ1 K_b 136 diff in [-2.3e-11, -1.7e-11]; LQ2 K_b 138 diff in [-2.4e-11, -1.8e-11];
 F  frozen pair: |diff| < 5e-12 for each event; FQ1 K_b 136 diff in [-2.8e-12, -1.6e-12]; FQ2 K_b 138 diff in [-3.0e-12, -1.8e-12].
Selftest predictions (registration text of Q50, written before the first run):
 P1 on synthetic events at the frozen-pair prediction plus 2.2e-12 (K_b 136) and 2.4e-12 (K_b 138), that is diff = -2.2e-12 and -2.4e-12, the frozen labels F, FQ1 and FQ2 hold;
 P2 on synthetic events at the frozen-pair prediction minus 1e-11 (diff = +1e-11) the frozen label F does not hold;
 P3 with no event files the script prints NO EVENT and exits 0;
 P4 an event file without the key A_event makes it print VOID and exit 2 (tested in a subprocess). The first selftest run (kept as q1668_selftest_out_first_run_fail.txt) failed P4 because os._exit lost the unflushed VOID line; the repair is flush=True.
 P5 control: the frozen predictions printed equal 77.388989108974 and 78.257764318990.
Usage: py q1668_births_136_138_labels.py (real events, directory from BL_FWPROD or fw_prod) | py q1668_births_136_138_labels.py --selftest (from small_gap_q206/).
"""

import json, os, subprocess, sys, tempfile
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "..", "centre_scripts", "d22_out_18.json")))
E = {n: float(S["e%d" % n]) for n in range(2, 19)}
B.F = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13]
B.NG = 12
B.g = [1.0]
for n in range(1, 13):
    B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))
LADDER = (-0.0937736504642, 0.0011599156647)
FROZEN = (-0.0937736504642 - 2.8982e-10, 0.0011599156647 + 2.7012e-10)
ex = [E[n] for n in range(6, 19)]
BARS = {
    136: dict(L=3e-11, LQ=(-2.3e-11, -1.7e-11), F=5e-12, FQ=(-2.8e-12, -1.6e-12)),
    138: dict(L=3e-11, LQ=(-2.4e-11, -1.8e-11), F=5e-12, FQ=(-3.0e-12, -1.8e-12)),
}
lab = lambda q: "holds" if q else "does not hold"


def evaluate(verbose=True):
    out = {}
    for K in (136, 138):
        f = os.path.join(os.environ.get("BL_FWPROD", os.path.join(H, "fw_prod")), "event_Kb%d.json" % K)
        if not os.path.exists(f):
            continue
        try:
            A = float(json.load(open(f))["A_event"])
        except Exception as e:
            print("VOID: event file of K_b %d unreadable (%s)" % (K, type(e).__name__), flush=True)
            os._exit(2)
        dl = B.a_pred(K, *LADDER, ex) - A
        df = B.a_pred(K, *FROZEN, ex) - A
        b = BARS[K]
        out[K] = dict(
            dl=dl,
            df=df,
            L=abs(dl) < b["L"],
            LQ=b["LQ"][0] <= dl <= b["LQ"][1],
            F=abs(df) < b["F"],
            FQ=b["FQ"][0] <= df <= b["FQ"][1],
        )
        if verbose:
            print(
                "K_b %d: ladder diff %+.3e (bar %s, trend %s); frozen diff %+.3e (bar %s, trend %s)"
                % (K, dl, lab(out[K]["L"]), lab(out[K]["LQ"]), df, lab(out[K]["F"]), lab(out[K]["FQ"]))
            )
    return out


if "--selftest" in sys.argv:
    preds = {K: B.a_pred(K, *FROZEN, ex) for K in (136, 138)}
    print("frozen predictions: K_b 136 %.12f, K_b 138 %.12f" % (preds[136], preds[138]))

    def make(d, shift):
        os.makedirs(d, exist_ok=True)
        for K, s in zip((136, 138), shift):
            json.dump({"A_event": preds[K] + s}, open(os.path.join(d, "event_Kb%d.json" % K), "w"))

    res = {}
    with tempfile.TemporaryDirectory() as t:
        d1 = os.path.join(t, "a")
        make(d1, (2.2e-12, 2.4e-12))
        os.environ["BL_FWPROD"] = d1
        res[1] = evaluate()
        d2 = os.path.join(t, "b")
        make(d2, (-1e-11, -1e-11))
        os.environ["BL_FWPROD"] = d2
        res[2] = evaluate()
        d3 = os.path.join(t, "c")
        os.makedirs(d3)
        d4 = os.path.join(t, "d")
        os.makedirs(d4)
        json.dump({"x": 1}, open(os.path.join(d4, "event_Kb136.json"), "w"))
        r3 = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, BL_FWPROD=d3),
            capture_output=True,
            text=True,
        )
        r4 = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=dict(os.environ, BL_FWPROD=d4),
            capture_output=True,
            text=True,
        )
    p1 = all(res[1][K][x] for K in (136, 138) for x in ("F",)) and res[1][136]["FQ"] and res[1][138]["FQ"]
    p2 = all(not res[2][K]["F"] for K in (136, 138))
    p3 = r3.returncode == 0 and "NO EVENT" in r3.stdout
    p4 = r4.returncode == 2 and "VOID" in r4.stdout
    p5 = abs(preds[136] - 77.388989108974) < 5e-13 and abs(preds[138] - 78.257764318990) < 5e-13
    print(
        "P1 frozen labels hold on diff -2.2e-12 / -2.4e-12: %s; P2 bar does not hold on diff +1e-11: %s; P3 no files gives NO EVENT and exit 0: %s; P4 file without A_event gives VOID and exit 2: %s; P5 control predictions: %s"
        % (lab(p1), lab(p2), lab(p3), lab(p4), lab(p5))
    )
    print("SELFTEST " + ("PASS" if p1 and p2 and p3 and p4 and p5 else "FAIL"))
    sys.exit(0)
res = evaluate()
if not res:
    print("NO EVENT K_b 136 or 138 in the event directory: no label")
    sys.exit(0)
print("READING q1668 done (labels by the script; no verdict word)")
