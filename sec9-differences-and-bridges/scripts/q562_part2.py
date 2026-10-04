"""Appends PART 2 (the prediction numbers) to q562_prereg.txt; refuses if any case is missing. Also prints the mapping checks against the
non-outcome numbers the specs file already states (CBA rates 0.736 at D = 4 and 0.424 at D = 8 nats; Mao-Gray-Linder D = 0.0173 at
R = 1 bit, uniform [0, 1))."""

import json, os, time, math

cases = ["A9", "A10", "A11", "A12", "A8", "A13", "A3", "A4", "A1", "A2", "A5", "A5alt"]
missing = [c for c in cases if not os.path.exists(f"q562_pred_{c}.json")]
if missing:
    raise SystemExit(f"missing: {missing}")
P = {c: json.load(open(f"q562_pred_{c}.json")) for c in cases}
L = [
    f"PART 2 (prediction numbers), appended {time.strftime('%H:%M:%S')} (date stamp) before bridge_outcomes_SEALED.md is opened."
]
for c in cases:
    p = P[c]
    near = p["nearest_uniform_transition_distance"]
    flag = (
        ""
        if near is None
        else (" NEAR TRANSITION (%.3f)" % near if near < 0.1 else " (nearest transition %.2f)" % near)
    )
    L.append(
        f"  {c}: a = {p['a']:.4f} ({p['mode']}), predicted K = {p['K']}, certified {p['certified']}, I = {p['I_nats']:.4f} nats, D_noise = {p['D_noise']:.5f}{flag}"
    )
L.append("  Mapping checks against numbers already in the specs (not outcomes):")
for c, R in (("A11", 0.7360), ("A8", 0.4243)):
    L.append(f"    {c}: implied rate {P[c]['I_nats']:.4f} nats against the CBA rate {R} (K = 160)")
a13 = P["A13"]
D13 = a13["D_noise"] * 0.25 / a13["a"] ** 2
L.append(f"    A13: implied D = {D13:.5f} against Mao-Gray-Linder's 0.0173 at R = 1 bit")
L.append(
    "  Registered: A5 uses beta = 1 (K = %d); A5alt (beta = 2, K = %d) is secondary. A6 excluded."
    % (P["A5"]["K"], P["A5alt"]["K"])
)
open("q562_prereg.txt", "a").write("\n".join(L) + "\n")
print("\n".join(L))
