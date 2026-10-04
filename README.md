# How Many Atoms, and Where: supplementary material

Code, computed states and run logs for the paper

> I Made Tristan Hope Firdaus, *How Many Atoms, and Where: Predicting the Discrete Optima of Capacity, Minimax Estimation and
> the NPMLE Under Gaussian Noise*, working paper, 2026. The PDF is in [`paper/`](paper/).

The paper studies three problems in which a probability law on a bounded region is seen through Gaussian noise. They are the
capacity-achieving input of a peak-limited channel, the least favorable prior (LFP) of a bounded normal mean, and the
nonparametric maximum-likelihood estimate (NPMLE) of a mixing law. In all three the optimal law is discrete. The paper predicts
how many atoms it has and where they sit, and certifies one counterexample to a published theorem.

## Where to find what

Each folder holds the material behind one section of the paper.

| folder | paper section | what it reproduces |
|---|---|---|
| [`sec3-one-dimension/`](sec3-one-dimension/) | III | edge constants (Table I) and the forward tests of the count law, fixed before the computation |
| [`sec4-closed-curves/`](sec4-closed-curves/) | IV | the formal density, the Szegő rule and the ring losses of Tables II and III, Fig. 1 |
| [`sec5-certificate/`](sec5-certificate/) | V and Appendix A | **Theorem 1**: the certified 32-atom ring without vertex atoms (runs on its own) |
| [`sec6-filled-regions/`](sec6-filled-regions/) | VI | births at the center (Table V, Fig. 3), the first center atom, rings inside rings (Table VI) |
| [`sec7-value-laws/`](sec7-value-laws/) | VII | optimal values on discs, balls, shells and annuli (Table VII, Fig. 4), close walls |
| [`sec8-center-events/`](sec8-center-events/) | VIII | events at the center of d-balls and the shift law |
| [`sec9-differences-and-bridges/`](sec9-differences-and-bridges/) | IX | what does not carry over, and the blind test against published rate-distortion and rational-inattention solutions |
| [`registration-log/`](registration-log/) | all | the append-only log in which every prediction was written before it was tested |
| [`checks/`](checks/) | all | the code that checked every number and figure of the paper against primary files, with what it reads |
| [`paper/`](paper/) | | the paper and the full table of inner-ring tests |

Every folder has a `README.md` (what is there and how it maps to the paper) and a `FILES.md` (one line per script: what it does,
and the name it had when it ran).

## Two kinds of folder

**`sec5-certificate/` is a package.** It runs on its own and reproduces the proof of Theorem 1. We reran it from this folder
before publishing and the output matches the original run digit for digit (`results/rerun_*` against `results/original_*`).

**The other folders are records.** They hold the scripts as they ran and the logs they wrote. File names are the original ones,
so they match the names in our registration log. The scripts expect the folder layout of the original project (for example
`centre_scripts/` for run folders), so rerunning one usually means adjusting a path at the top of the script. The Python files
were reformatted with [black](https://github.com/psf/black) for readability; black changes layout only, never the program.

## Requirements

Python 3.10 or later and the packages in [`requirements.txt`](requirements.txt):

```
pip install -r requirements.txt
```

`python-flint` provides the ball arithmetic (FLINT/Arb) used by every rigorous step.

## Quick start: check Theorem 1

```
cd sec5-certificate
python certify_ring32.py data/start_state_ring32.json ring32
```

About four minutes on three threads. The last lines should report `Krawczyk ... ok True`, `curve certificate ok True`, and
`D - C at the major vertex ... [-1.12760361457e-50, -1.12760361456e-50]`. See [`sec5-certificate/README.md`](sec5-certificate/README.md).

## What is proved and what is numerical

Only Theorem 1 and Proposition 1 of the paper are proved; the theorem rests on the computer-assisted certificate in
`sec5-certificate/`. Everything else is numerical. A prediction called *registered* was written into a dated, append-only log
before the optimum was computed; the log's time stamps are not certified by a third party. The log is in
[`registration-log/`](registration-log/), and Table VIII of the paper lists the registered tests that did not pass.

## How the numbers were checked

Every number printed in the paper was checked by code against a primary file: a run log, a result file, or an entry of the
registration log. The figures are drawn by scripts that read their data from such files. [`checks/`](checks/) holds that code,
the formula behind each derived number, and the input files that the other folders do not already hold.

## License and citation

Code: MIT License (see [`LICENSE`](LICENSE)). If you use this material, please cite the paper above.

Contact: I Made Tristan Hope Firdaus, Independent Researcher, Bali, Indonesia (madetristanfirdaus@gmail.com, ORCID
0009-0002-3048-3624).
