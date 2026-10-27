# The Capacity-Achieving Input of a Two-by-Two Peak-Constrained Gaussian Channel Need Not Use the Ends of Its Ellipse

Code and data for the paper of this title by I Made Tristan Hope Firdaus (2026).

The paper shows that, for the filled ellipse with semi-axes 1 and 0.9597, the capacity-achieving input is a symmetric ring of 32
mass points on the boundary, and neither end of the major axis is one of them (Theorem 1, Proposition 1). The proof is
computer-assisted and uses ball arithmetic. The certificate is the package [`../sec5-certificate/`](../sec5-certificate/), which runs
on its own and is unchanged since commit `de63783`. This folder holds the separate evaluator of Observation 1, the scripts of
Table I and of Figs. 1 to 3, and the check of every printed number.

## Where to find what

| paper | file |
|---|---|
| Section IV, Steps 1 to 4, and Proposition 1 (the certificate) | [`../sec5-certificate/certify_ring32.py`](../sec5-certificate/certify_ring32.py) with [`../sec5-certificate/quadrature_bounds.py`](../sec5-certificate/quadrature_bounds.py) (Lemma 2 and Corollary 1) and [`../sec5-certificate/kkt_krawczyk.py`](../sec5-certificate/kkt_krawczyk.py) (Steps 2 and 3); logs in [`../sec5-certificate/results/`](../sec5-certificate/results/) |
| Section IV, the bracket of the capacity | [`../sec5-certificate/capacity_bracket.py`](../sec5-certificate/capacity_bracket.py), log [`../sec5-certificate/results/rerun_capacity_bracket.log`](../sec5-certificate/results/rerun_capacity_bracket.log) |
| Observation 1 (the separate evaluator) | `scripts/separate_evaluator.py`, outputs `scripts/separate_evaluator_out_h20.json` and `scripts/separate_evaluator_out_h28.json` |
| Table I | `scripts/make_ring_table_fig.py`, outputs `scripts/table_ring.tex` and `scripts/numbers_ring.json` |
| Fig. 1 | `scripts/make_fig_face_B.py` |
| Fig. 2 | `scripts/margin_along_boundary.py` (the values along the boundary), then `scripts/make_fig_margin.py` |
| Fig. 3 | `scripts/make_fig_gap.py` |

The ring itself, to full precision, is the center of the certified box,
[`../sec5-certificate/results/rerun_ring32_polished.json`](../sec5-certificate/results/rerun_ring32_polished.json); the figure
scripts read the same values from [`../checks/inputs/F_work__ring32_R2fix_polished.json`](../checks/inputs/F_work__ring32_R2fix_polished.json).

## Running the scripts

Install the packages of [`../requirements.txt`](../requirements.txt). The certificate is run as described in
[`../sec5-certificate/README.md`](../sec5-certificate/README.md); it takes about four minutes. The separate evaluator takes the center
file and the step, the half-width and the precision:

```
python scripts/separate_evaluator.py sec5-certificate/results/rerun_ring32_polished.json 20 22 256
python scripts/separate_evaluator.py sec5-certificate/results/rerun_ring32_polished.json 28 26 320
```

(from the top folder of the repository; a few seconds each). It writes its result into a file separate_evaluator_out.json next to itself; the two runs quoted
in the paper are kept as `scripts/separate_evaluator_out_h20.json` and `scripts/separate_evaluator_out_h28.json`. It shares no code with the certificate and gives no error bound, so its
values are checked numerically only. `scripts/margin_along_boundary.py 721` evaluates the same function at 921 boundary points in
about a minute; the other scripts run in seconds. Each script finds its inputs through `scripts/layout.py` and writes its outputs
next to itself.
