# appendix-checks

The numerical checks behind statements of the Supplementary Notes and of the appendices: the deformation identity of the planar NPMLE (Supplementary Notes S8), the one-minimum lemma (Appendix C) and the proof of the total-positivity theorem (Appendix D).

## What is here

- `scripts/` holds the scripts named by the registration-log entries that those parts cite: the solvers of the NPMLE on discs and ellipses and of the tilted one-dimensional channel behind the deformation identity. A script that a section also cites is filed under the first section that cites it, so the scripts of Appendix C are in [`../sec13-monotonicity/`](../sec13-monotonicity/), those of Appendix D are in [`../sec12-total-positivity/`](../sec12-total-positivity/), and the check of Theorem 1 (Appendix A) is in [`../sec3-proved-in-one-dimension/`](../sec3-proved-in-one-dimension/). The last column of each `FILES.md` names the sections that cite the script.
- `logs/` holds the outputs of those runs.

## How to read it

This folder is a record. The scripts are the ones that ran, and the file names match the registration log. A numerical check at stored states supports a statement of an appendix; it does not replace the proof in the appendix, and the appendices say which steps rest on a check.

The log is in [`../registration-log/`](../registration-log/).

## Scope

The folder holds the files that the cited entries name and that the build found in the project tree. Outputs larger than one megabyte are not copied.

The terms of this folder are those of the paper (Section I, Terms) and of the top README.
