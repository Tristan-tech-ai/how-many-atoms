# checks: how the numbers and figures of the paper were verified

Every number printed in the paper was checked by code against a primary file: a run log, a result file, or an entry of the
registration log. A number passes only if it occurs in such a file to its printed precision, or if a declared formula over such
files reproduces it. This folder holds that code and what it reads.

| folder | content |
|---|---|
| `bindings/` | one entry for each number that no file prints as such: the paper's words around it, and the formula that reproduces it |
| `scripts/` | the number gate (`gate_numbers.py`, with `truth_index.py`, which indexes the numbers of every primary file), the figure gate, the scripts that recompute derived quantities, and the generators of the bindings of the merged paper |
| `figures/` | the figure scripts; they read their data through `scripts/figsrc.py` and hold no typed-in values |
| `outputs/` | what the recomputation scripts wrote, as used for the paper |
| `inputs/` | primary files that the scripts, bindings and figures read and that the section folders do not already hold |

In a binding, `L(n, regex)` reads a number from entry n of the registration log, `J(path, keys)` a value of a JSON file, and
`T(path, regex)` a number of a text file. The bindings of the merged paper are in `bindings/bindings_unified.json` and those of its Supplementary Notes in `bindings/bindings_supplement_notes.json`. It joins the
bindings of the earlier version of the paper, with the paper name changed, to the bindings of the new sections in
`bindings/unified_new_src.json`, which the script `scripts/make_bindings_unified_new.py` writes.

The scripts expect the folder layout of the original project, whose paths are listed in [`FILES.md`](FILES.md). Texts of cited papers
are not copied; the paper cites them.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
