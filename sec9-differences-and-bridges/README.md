# Section IX: what does not carry over, and the bridges

## What does not carry over

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| ring counts of the capacity and NPMLE discs | `q668_npmle2d_cont.py` | `q668_*.log` |
| annulus births and the center | `q686f/g/h_annulus_births.py`, `q693_centre_dfrac.py` | `q686*.log`, `q693*.log` |
| capacity wall supports on ellipses, superellipses and other shapes | `q699_wall_kkt.py`, `q700_ba2d_ellipse.py`, `q701_ring_wall.py`, `q701b_ring_wall.py`, `q702_ba2d_rose.py` | `q699*.log` to `q702*.log` |

## The blind test against published solutions

A decision made under squared-error loss and a cost on mutual information (rational inattention) is a rate-distortion problem,
and rate-distortion with squared error is the NPMLE problem of the paper. We tested this bridge blind against twelve published
one-dimensional solutions.

1. The specifications of the twelve cases were extracted into `bridge_specs.md`, and the published outcomes were sealed in
   `bridge_outcomes_SEALED.md`.
2. Reading only the specifications, the mapping to our problem was fixed in `q562_prereg.txt` (part 1).
3. Every prediction was computed with our NPMLE solver (`q562_predict.py`, `q562_rdlib.py`) and appended (`q562_part2.py`,
   part 2 of `q562_prereg.txt`, with flags for cases near a count change). Outputs: `q562_pred_A*.json`, `q562_pred_A*.log`.
4. Only then was the sealed file opened.

Result: the predicted number of reproduction points or actions matched the published number in **9 of 12** cases (8 of 9 with a
uniform source, 1 of 3 with a truncated normal one), with no fitted parameter. Where the counts agree, the positions agree to the
published precision. The three misses are cases A1, A2 and A3 (compare `q562_pred_A*.log` with `bridge_outcomes_SEALED.md`);
in A3 the operating point sits 0.006 from a predicted count change.

Two further checks:
- `q505_rd_bridge.py` (with `q505_handcheck.py`): an independent rate-distortion solver finds the first six NPMLE change points of
  a uniform source to about 1e-5; these are the critical temperatures of deterministic annealing. Log: `q505_rd_bridge.log`.
- `q506_ri_replication.py`: our functional reproduces the four-action solution published by Jung, Kim, Matějka and Sims
  (*Discrete actions in information-constrained decision problems*) to their stated three decimals. Log:
  `q506_ri_replication.log`.

See [`FILES.md`](FILES.md) for every script.
