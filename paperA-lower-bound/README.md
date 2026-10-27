# A Lower Bound of Order A^(4/3) on the Number of Mass Points of the Capacity-Achieving Input of the Amplitude-Constrained Gaussian Channel

Code and data for the paper of this title by I Made Tristan Hope Firdaus (2026).

The paper proves that the capacity-achieving input of the scalar Gaussian channel with amplitude constraint A has at least
0.1660 A^(4/3) mass points for all large A (Theorem 1). The proof is analytic except for one number: the sign of an explicit
function J at 1.1648, which is evaluated in ball arithmetic (Appendix C). Section VI and Figs. 1, 3, 4 and 5 compare the theorem
with computed optima. These comparisons are checked numerically and are not part of the proof.

## Where to find what

| paper | file |
|---|---|
| Appendix C: the sign of J(1.1648) and the constants of Theorem 1 | [`../sec3-proved-in-one-dimension/scripts/q1796_lower_bound_constants_arb.py`](../sec3-proved-in-one-dimension/scripts/q1796_lower_bound_constants_arb.py), output [`../sec3-proved-in-one-dimension/logs/q1796_out.txt`](../sec3-proved-in-one-dimension/logs/q1796_out.txt) |
| Appendix C: the printed parts of J(1.1648) | `scripts/numbers_A2.py`, output `scripts/numbers_A2.json` |
| Fig. 1 | `scripts/make_fig_face.py`, outputs `scripts/fig_face.pdf` and `scripts/fig_face_numbers.json` |
| Fig. 2 (a schematic drawn with the true formulas) | `scripts/make_fig_map.py` |
| Fig. 3 | `scripts/make_fig_flat.py`, outputs `scripts/fig_flat.pdf` and `scripts/fig_flat_numbers.json` |
| Fig. 4 | `scripts/make_fig_step2.py`, outputs `scripts/fig_step2.pdf` and `scripts/fig_step2_numbers.json` |
| Fig. 5 and Observation 1 | `scripts/make_fig_count.py` and `scripts/summary_states.py`, output `scripts/numbers_states.json` |
| the computed optima of Section VI | [`data/stored-optima/`](data/stored-optima/) |
| the amplitudes at which the count changes (Fig. 5) | [`../checks/outputs/count_law_data.json`](../checks/outputs/count_law_data.json) |

## The computed optima

[`data/stored-optima/`](data/stored-optima/) holds the 2668 computed optima of Section VI, one JSON file each, at amplitudes from 17
to 52.48 with 20 to 83 mass points. Each file gives the amplitude `A`, the number of mass points `K` and the capacity `C` in nats.
The positive interior positions are `u`, their masses `w`, the mass of each end point `we`, and the mass of the center point `wc`
(or `None`). The file also gives the residual of the Kuhn-Tucker equations at the mass points and the verdict of the check of the
Kuhn-Tucker function on a grid. The
positions and masses are given to about 40 digits. They were computed by Newton's method in high-precision arithmetic, and every
residual is below 1e-20. They are not interval computations, so they do not prove the counts they show. The file names and
subfolders are those listed in `../checks/outputs/count_law_data.json`, which also lists the 138 amplitudes at which the count changes.

## Running the scripts

Install the packages of [`../requirements.txt`](../requirements.txt), then run any script from any folder, for example

```
python scripts/make_fig_face.py
```

Each script finds its inputs through `scripts/layout.py` and writes its outputs next to itself. All of them run in seconds, apart
from `scripts/summary_states.py`, which reads all 2668 optima. The ball-arithmetic script of Appendix C runs in a few seconds and ends
with `CHECK q1796: ALL THREE STATEMENTS HOLD`.
