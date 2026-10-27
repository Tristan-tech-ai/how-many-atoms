# How Many Atoms: code and data for three papers on the capacity-achieving input of amplitude-constrained Gaussian channels

This repository holds the code, the computed optima, the certificates and the run logs behind three papers by
I Made Tristan Hope Firdaus (2026). Each paper has its own folder with a guide to the files it rests on, the scripts of its figures
and tables, and the data they read.

| paper | folder | main result |
|---|---|---|
| "A Lower Bound of Order A^(4/3) on the Number of Mass Points of the Capacity-Achieving Input of the Amplitude-Constrained Gaussian Channel" | [`paperA-lower-bound/`](paperA-lower-bound/) | the capacity-achieving input of the scalar channel with amplitude constraint A has at least 0.1660 A^(4/3) mass points for all large A (proved) |
| "The Capacity-Achieving Input of a Two-by-Two Peak-Constrained Gaussian Channel Need Not Use the Ends of Its Ellipse" | [`paperB-mimo-ring/`](paperB-mimo-ring/) | for the ellipse with semi-axes 1 and 0.9597 the capacity-achieving input is a ring of 32 mass points without mass at the ends of the major axis (computer-assisted proof in ball arithmetic) |
| "Total Positivity and the Monotonicity Question for the Amplitude-Constrained Gaussian Channel" | [`paperC-total-positivity/`](paperC-total-positivity/) | an equivalent local form of the question whether the number of mass points can decrease, strict total positivity of the Gram part of the velocity matrix (proved), and outward motion at 91 optimal states (computer-assisted) |

For the larger computations the paper folders point into shared folders. [`sec5-certificate/`](sec5-certificate/) is the
certificate of paper B, a package that runs on its own. [`sec3-proved-in-one-dimension/`](sec3-proved-in-one-dimension/) holds the
ball-arithmetic step of paper A. [`sec12-total-positivity/`](sec12-total-positivity/) holds the computer-assisted proof of paper C,
and [`sec13-monotonicity/`](sec13-monotonicity/) the data of its observations. [`checks/`](checks/) holds input files that several
scripts share.

Install the packages of [`requirements.txt`](requirements.txt) (`pip install -r requirements.txt`; Python 3.10 or later). Each paper
folder's README says how to rerun its scripts.

## The earlier merged paper

The three papers were written from one earlier working paper, "How Many Atoms, Where, and When", whose PDF and Supplementary Notes
are in [`paper/`](paper/). The folders `sec3-one-dimension/` to `sec15-open-problems/`, `appendix-checks/`, `registration-log/` and
`checks/` were organized for that paper, and the section, theorem and table numbers in their READMEs refer to it. The rest of this
page is the guide to those folders as it was written for the merged paper.

### How Many Atoms, Where, and When: supplementary material

Code, computed states and run logs for the paper

> I Made Tristan Hope Firdaus, *How Many Atoms, Where, and When: Predicting the Discrete Optima of Peak-Power Capacity, Minimax Estimation and Nonparametric Maximum Likelihood Under Gaussian Noise*, working paper, 2026. The paper and the Supplementary Notes are in [`paper/`](paper/).

The paper studies three optimal laws under Gaussian noise: the
capacity-achieving input of a peak-limited channel, the least favorable prior (LFP) of a bounded normal mean, and the
nonparametric maximum-likelihood estimate (NPMLE) of a mixing law. In all three the optimal law is discrete. The paper proves a
lower bound of order A^(4/3) on the number of atoms of the capacity-achieving input on an interval of half-length A. It certifies one
counterexample to a published theorem about the two-by-two MIMO channel, and it predicts where atoms sit and at what amplitude a new
atom appears. It also follows one optimum along the amplitude and reduces the question whether the number of atoms can decrease to
two statements about the innermost atoms.

#### Terms

The paper defines its terms in Section I. In brief: a result is *certified* when a computer-assisted proof establishes it, with every quantity
computed in ball arithmetic, the interval arithmetic in which each number is a midpoint with a radius that bounds all rounding and truncation
errors. A prediction is *registered* when its value and an acceptance band were written into the dated log of
[`registration-log/`](registration-log/) before the optimum was computed, and *blind* when the optimum did not yet exist. A *measured law* is a
relation found on computed optima and tested on later ones; it is not proved. At a *birth* a new atom appears at the center of the interval, and at
a *split* the atom at the center becomes two; the *births law* gives the amplitudes of these events. A *stored state* is a computed optimum kept in
this repository with its Kuhn-Tucker residual. An *amendment* is an entry of the registration log. A *verdict* is the last line that a test
script prints: PASS, FAIL, or another outcome stated in advance; a run that stopped without a verdict line is VOID. The other terms are those of the paper.

#### Where to find what

Each folder holds the material behind one part of the paper. The folder names are fixed identifiers and do not follow the section
numbers; the second column gives the sections and appendices of the paper, and `S1`, `S2`, ... are the sections of the
Supplementary Notes, which are submitted with the paper.

| folder | paper | what it reproduces |
|---|---|---|
| [`sec3-one-dimension/`](sec3-one-dimension/) | III-A | edge constants (Table I) and the forward tests of the count law, fixed before the computation |
| [`sec3-proved-in-one-dimension/`](sec3-proved-in-one-dimension/) | III-B, Appendix A | **Theorem 1** (lower bound of order A^(4/3)): its one numerical input in ball arithmetic and checks of its bounds at certified states |
| [`sec4-closed-curves/`](sec4-closed-curves/) | IV | the formal density, the Szegő rule and the ring losses of Tables II to IV, Fig. 1 |
| [`sec5-certificate/`](sec5-certificate/) | V, Appendix B | **Theorem 2**: the certified 32-atom ring without vertex atoms (runs on its own) |
| [`sec6-filled-regions/`](sec6-filled-regions/) | VI, S3, S4 | births at the center, the first center atom, the continuation past the birth, rings inside rings |
| [`sec7-value-laws/`](sec7-value-laws/) | S1 | optimal values on discs, balls, shells and annuli, close walls |
| [`sec8-center-events/`](sec8-center-events/) | S2 | events at the center of d-balls and the shift law |
| [`sec9-tower-of-events/`](sec9-tower-of-events/) | VII-A | following one optimum as the amplitude grows: the tower of events |
| [`sec10-births-law/`](sec10-births-law/) | VII-B | the births law, the stored coefficients of its series and its blind tests |
| [`sec11-residual/`](sec11-residual/) | S5, S6 | the residual of the series and the structure of its coefficients |
| [`sec12-total-positivity/`](sec12-total-positivity/) | VIII, Appendix D | **Theorem 3**: total positivity of the matrix that governs the motion of the atoms |
| [`sec13-monotonicity/`](sec13-monotonicity/) | IX, Appendix C | the reduction of the monotonicity question and its evidence |
| [`sec14-differences-and-bridges/`](sec14-differences-and-bridges/) | X, S7 | what does not carry over, and the blind test against published rate-distortion and rational-inattention solutions |
| [`sec15-open-problems/`](sec15-open-problems/) | X | the registered tests that the open problems refer to |
| [`appendix-checks/`](appendix-checks/) | S8, Appendices C and D | numerical checks of statements of the appendices |
| [`registration-log/`](registration-log/) | all | the append-only log in which every prediction was written before it was tested |
| [`checks/`](checks/) | all | the code that checked every number and figure of the paper against primary files, with what it reads |
| [`paper/`](paper/) | | the paper and the Supplementary Notes |

Every folder has a `README.md` (what is there and how it maps to the paper) and a `FILES.md` (one line per script: what it does,
the name it had when it ran, and the sections that cite it).

#### Two kinds of folder

**`sec5-certificate/` is a package.** It runs on its own and reproduces the proof of Theorem 2. It was rerun from this folder
before publishing and the output matches the original run digit for digit (`results/rerun_*` against `results/original_*`).

**The other folders are records.** They hold the scripts as they ran and the logs they wrote. File names are the original ones,
so they match the names in the registration log. The scripts expect the folder layout of the original project (for example
`centre_scripts/` for run folders), so rerunning one usually means adjusting a path at the top of the script. The Python files
were reformatted with [black](https://github.com/psf/black) for readability; black changes layout only, never the program.

#### Requirements

Python 3.10 or later and the packages in [`requirements.txt`](requirements.txt):

```
pip install -r requirements.txt
```

`python-flint` provides the ball arithmetic (FLINT/Arb) used by every rigorous step.

#### Quick start: check Theorem 2

```
cd sec5-certificate
python certify_ring32.py data/start_state_ring32.json ring32
```

About four minutes on three threads. The last lines should report `Krawczyk ... ok True`, `curve certificate ok True`, and
`D - C at the major vertex ... [-1.12760361457e-50, -1.12760361456e-50]`. See [`sec5-certificate/README.md`](sec5-certificate/README.md).

#### Quick start: the numerical input of Theorem 1

```
cd sec3-proved-in-one-dimension/scripts
python q1796_lower_bound_constants_arb.py
```

A few seconds. The output shows the ball for J(1.1648), which lies below zero, the ball for J(1.1647), which lies above zero, and the
constants that follow, ending with `CHECK q1796: ALL THREE STATEMENTS HOLD`. The argument itself is Appendix A of the paper.

#### What is proved and what is numerical

Every theorem, proposition, lemma and corollary of the paper has its proof in the paper, or the certificate on which it rests.
Theorem 1 and its proof (Appendix A) concern the capacity-achieving input; Theorem 2 rests on the computer-assisted certificate in
`sec5-certificate/`; Theorem 3 proves total positivity for the part of a matrix built from the Gaussian kernel. Everything else is
numerical and is labeled as a measured law or a conjecture in the paper. A prediction called *registered* was written into a dated,
append-only log before the optimum was computed; the log's time stamps are not certified by a third party. The log is in
[`registration-log/`](registration-log/), and Table VII of the paper and Table S7 of the Supplementary Notes list the registered
tests that did not pass.

#### How the numbers were checked

Every number printed in the paper was checked by code against a primary file: a run log, a result file, or an entry of the
registration log. The figures are drawn by scripts that read their data from such files. [`checks/`](checks/) holds that code,
the formula behind each derived number, and the input files that the other folders do not already hold.

#### License and citation

Code: MIT License (see [`LICENSE`](LICENSE)). If you use this material, please cite the paper above.

Contact: I Made Tristan Hope Firdaus, Independent Researcher, Bali, Indonesia (madetristanfirdaus@gmail.com, ORCID
0009-0002-3048-3624).
