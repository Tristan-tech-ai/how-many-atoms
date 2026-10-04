# checks: how the numbers and figures of the paper were verified

Every number printed in the paper was checked by code against a primary file: a run log, a result file, or an entry of the
registration log. A number passes only if it occurs in such a file to its printed precision, or if a declared formula over such
files reproduces it. This folder holds that code and what it reads.

| folder | content |
|---|---|
| `bindings/` | one entry for each number that no file prints as such: the paper's words around it, and the formula that reproduces it |
| `scripts/` | the number gate (`gate_numbers.py`, with `truth_index.py`, which indexes the numbers of every primary file), the figure gate, and the scripts that recompute derived quantities |
| `figures/` | the figure scripts; they read their data through `scripts/figsrc.py` and hold no typed-in values |
| `outputs/` | what the recomputation scripts wrote, as used for the paper |
| `inputs/` | primary files that the scripts, bindings and figures read and that the section folders do not already hold |

In a binding, `L(n, regex)` reads a number from entry n of the registration log, `J(path, keys)` a value of a JSON file, and
`T(path, regex)` a number of a text file. The scripts expect the folder layout of the original project, whose paths are listed in
[`FILES.md`](FILES.md). Texts of cited papers are not copied; the paper cites them.
