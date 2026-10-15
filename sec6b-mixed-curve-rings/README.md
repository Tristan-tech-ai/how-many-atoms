# sec6b-mixed-curve-rings

The material behind Section VIII of the paper for the center births on the mixed curves (Table VI, notes d and e). The rings are
followed from the ellipse to the mixed curve with every mass kept positive. The registered center births are then tested on those rings.

- `scripts/` holds the continuation in the curve parameter, the walks in rm and the solver. It also holds the scripts that read
  the center excess and the two verdict scripts of the registered tests. `FILES.md` lists every script with the first sentence
  of its own docstring and the name it had when it ran.
- `logs/` holds the outputs of the continuations, the walks and the two tests.

The NPMLE ring on the mixed curve loses its pair at the ends of the major axis near rm = 1.7393. The ring of 18 atoms that follows
gives the center birth of the registered test. The LFP ring first merges an orbit into a pair at the ends of the minor axis. It then
loses its pair at the ends of the major axis, and the ring of 14 atoms that follows gives the center birth of its registered test.
The registration log entries 2252, 2253 and 2254 in [`../registration-log/`](../registration-log/) state each step with its script
and output.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
