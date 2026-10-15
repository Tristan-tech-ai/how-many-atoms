# sec13-monotonicity

The material behind Section V-A of the paper and Appendix B. It concerns the reduction of the monotonicity question for the number of atoms to two statements about the innermost atoms, and the numerical evidence for those statements.

## What is here

- `scripts/` holds the scripts that read the stored states, form the pairs of amplitudes, compute the majorization and margin quantities, and fit the margins and test their limits. `FILES.md` lists every script with the first sentence of its own docstring and the name it had when it ran.
- `logs/` holds the outputs of those runs.

## How to read it

This folder is a record. The scripts are the ones that ran, and the file names match the registration log. The two statements are numerical here and are not proved. The margin limits reported in the paper come from extrapolations of finitely many rows, and the paper says so.

The log is in [`../registration-log/`](../registration-log/).

## Scope

The folder holds the files that the cited entries name and that the build found in the project tree. Outputs larger than one megabyte are not copied.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
