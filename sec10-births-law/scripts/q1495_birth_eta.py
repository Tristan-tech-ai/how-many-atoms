"""q1495 (9 Oct 2026, HELPER, no verdict word): rough time to the next event of each running capacity chain. Reads from the chain logs only the family position A and the elapsed seconds of the lines '   A = <A>: residual ... [<t>s]' (a position along the family, not an event amplitude; signals and residuals are not read) and takes the distance to the REGISTERED prediction of the next event from centre_scripts/q1451_cap_b20_table.json (stage 134). The speed is the progress over the last 8 family lines; the step size is adaptive, so the ETA is rough (a factor of 3 either way) and says nothing about the event itself.
Usage: py q1495_birth_eta.py (from small_gap_q206/)."""

import json, re

tab = json.load(open("../centre_scripts/q1451_cap_b20_table.json"))["stages"]["134"]
PAT = re.compile(
    r"\s+A = ([0-9.]+): residual [0-9.e+-]+ in \d+ it, signal [^ ]+ ?,? ?h [0-9.e+-]+\s+\[(\d+)s\]"
)


def fam_progress(fn):
    return [
        (float(m.group(1)), int(m.group(2)))
        for m in map(PAT.match, open(fn, encoding="utf8", errors="replace"))
        if m
    ]


for name, K in (
    ("capc_fw_cap_par_124", 124),
    ("capc_fw_cap_par_126", 126),
    ("capc_alt_128d", 128),
    ("capc_alt_128e", 128),
    ("capc_fw_cap_par_130", 130),
):
    pts = fam_progress(f"cap_logs/{name}.log")
    if len(pts) < 9:
        print(f"{name}: too few family lines ({len(pts)})")
        continue
    (a1, t1), (a0, t0) = pts[-1], pts[-9]
    speed = (a1 - a0) / max(t1 - t0, 1)
    dist = float(tab[str(K)]) - a1
    print(
        f"{name}: K_b {K}: {len(pts)} family lines, distance to the registered prediction {dist:.4f}, speed {speed * 60:.5f} A/min, rough ETA "
        + (f"{dist / speed / 60:.0f} min" if speed > 0 and dist > 0 else "n/a")
    )
