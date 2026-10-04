# Section V and Appendix A: the certificate of Theorem 1

**Theorem 1.** For the filled ellipse with semi-axes r_p = 1 and r_m = 0.9597, the capacity-achieving input of Y = X + Z
(Z standard Gaussian in the plane) is a symmetric ring of 32 atoms with no atom at the ends of the major axis, (±1, 0).
This contradicts Theorem 4 of Dytso, Barletta and Kramer (arXiv:2401.17084).

## Run it

```
python certify_ring32.py data/start_state_ring32.json ring32
python capacity_bracket.py 1.0 0.9597 4 128 5e-14
```

The first command takes about four minutes (environment variable `THREADS`, default 3) and writes `results/ring32_*.json`.
The second takes about ten seconds and prints an independent two-sided bracket on the capacity.

## What each step proves

| step | file | what it shows |
|---|---|---|
| 1. quadrature with an error bound | `quadrature_bounds.py` | every value of the optimality function D_π is a ball that contains the true value (trapezoid rule, Lemma 1 of the paper) |
| 2. an exact solution | `kkt_krawczyk.py` | a Krawczyk test: a box of radius 6.82e-66 around the computed ring holds exactly one solution of the optimality equations |
| 3. the boundary | `kkt_krawczyk.py` | Taylor enclosures on [0, π/2]: D_π − C < 0 away from the atoms, concave near each atom, uniformly over the box |
| 4. the interior | (argument in the paper) | the largest atom radius is below 1, so D_π is strictly convex and the interior lies below the boundary maximum |
| driver | `certify_ring32.py` | runs steps 1 to 3 and prints the values at the two vertices |
| second check | `capacity_bracket.py` | 0.3906492301481604 ≤ C ≤ 0.3906492301483361 from a smooth input on 128 points, with no use of the ring |

`data/start_state_ring32.json` is the numerical starting point (a 640-node Gauss-Hermite solve). It is not part of the proof:
the Krawczyk test certifies whatever point it is given.

## Results

| file | content |
|---|---|
| `results/original_ring32_cert.log` | the run reported in the paper |
| `results/rerun_ring32_cert.log` | the same run repeated from this folder before publishing; identical numbers |
| `results/*_ring32_krawczyk.json` | box radii of every unknown (largest 6.817e-66) and the contraction norm 8.970e-22 |
| `results/*_ring32_polished.json` | the box midpoint: angles, masses and C |
| `results/*_capacity_sandwich.log`, `results/rerun_capacity_bracket.log` | the two-sided capacity bracket |

Key lines of the log: `Krawczyk attempt 0 ... ok True; ||I - YJ(B)||_inf 8.970e-22`, `curve certificate ok True`, and the vertex
values `[-1.12760361457e-50, -1.12760361456e-50]` at (1, 0) and `-1.41767643390e-48` at (0, 0.9597).

## Settings

Environment variables of `certify_ring32.py` (defaults are the values used in the paper): `HD`, `NN`, `PREC` (fine grid step
1/HD, nodes, bits), `HDC`, `NNC`, `PRECC` (curve grid), `A` (strip half-width, 1.45), `NT`, `MT`, `RHO` (Taylor intervals,
order, Cauchy radius), `DZ` (half-width of the zone around each atom), `THREADS`.
