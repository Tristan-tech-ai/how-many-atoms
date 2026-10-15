# Bridge cases, specifications only (FILE A)

Written 30 Sep 2026. This file holds problem specifications only. No reported count of actions, clusters, support points or phases, and no critical value read off a figure, appears here. Reported outcomes are in `bridge_outcomes_SEALED.md`; do not open it before predictions are frozen.

Every quote below was copied from text I extracted myself (pdftotext or pypdf, or the tag-stripped PMC HTML where noted) or read by me from a PNG I rendered with pdftoppm. Greek letters that the extractors dropped are restored in [brackets] from the rendered page. All source files are in `papers/bridge/`; the full search and download log is `papers/bridge/search_log.txt` and is summarised at the end of this file.

Notation I use when I restate a convention (my wording, not a quote): the rate-distortion Lagrangian with slope beta is `min I(X;Y) + beta * E[(X-Y)^2]`, with I in nats. For a Gaussian test channel this gives a conditional variance of 1/(2 beta).

---

## Core cases: bounded prior with a stated density, quadratic loss, Shannon mutual information

### Case A1. Matejka and Sims, CERGE-EI WP 441 (June 2011), Figure 1

- File: `cerge_wp441_matejka_sims_2011.pdf` (PDF page 15, printed page 12). An earlier copy of the same paper, `sims_wayback_DiscreteTracking2Prop.pdf` (Sims' site, file dated 2010-11-18, via the Wayback Machine), has the same figures.
- Problem (PDF page 7, printed page 4), max over f of E U - alpha^-1 I, with I in nats:
  > "max ∫U(|x− y|) f (x, y) µx(dx) µy(dy) − α−1 (∫ log( f (x, y)) f (x, y) µx(dx) µy(dy) + ∫ log (∫ ... )"
  > "U(|y − x|) is the objective function being maximized, and α is the inverse of the cost of information (or of the Lagrange multiplier on the information constraint)."
- Loss: quadratic. Section 4 sets it up: "Suppose U = −(y − x)2, g is Gaussian with variance σ2, and the 'information per util' parameter is α as in the problem statement (1-3)." It then says the untruncated Gaussian solution holds "when σ2 > 1/(2α)". So alpha plays the role of beta in my notation: E U - alpha^-1 I with U = -(y-x)^2 is equivalent to min I + alpha E(y-x)^2.
- Prior: truncated normal on (-1, 1), discretised:
  > "Figure 1 shows the solution for a case where g is a normal pdf truncated at ±3 standard errors, so that only a tiny fraction of the total probability under the normal density is excluded. The U function is quadratic and has a spread narrower than that of g"
  > "note that the black line is g — a normal density with standard deviation 1/3"
  > "The g distribution is not actually continuous in these exercises, but instead is concentrated on 201 equi-spaced points on (-1,1)."
- Parameter as stated: the figure title reads "Weighted conditional pdf's, normal g, U= a2, κ = 0.33". Here kappa is the mutual information of the optimal solution, in nats:
  > "Notice that the values of κ, the mutual information between x and y, shown at the top of Figures 1 and 2 differ by only .06. These κ values are measured in nats, that is, base-e information units."
- The alpha used for this figure is not stated in the text or the title. The operating point is fixed by kappa = 0.33 nats only.
- Method note: "For this and all the subsequent figures, the displayed solution has been verified to be optimal by searching for a solution with more points of support and finding convergence to the displayed solution."
- Outcome to predict: the number of support points of x. The figure marks them with asterisks on the horizontal axis.

### Case A2. Same paper, Figure 2

- File and page: `cerge_wp441_matejka_sims_2011.pdf`, PDF page 16 (printed page 13).
- Prior, loss and discretisation: as in A1 (normal with standard deviation 1/3, truncated at ±3 sd to (-1, 1), 201 grid points, U = -(y-x)^2).
- Parameter: title "Weighted conditional pdf's, Gaussian g, quadratic U, κ = 0.39" (nats). The text says Figure 2 uses a larger alpha (cheaper information) than Figure 1. The alpha value itself is not stated.
- Outcome to predict: the number of support points.

### Case A3. Same paper, Figure 3 (uniform prior)

- File and page: `cerge_wp441_matejka_sims_2011.pdf`, PDF page 17 (printed page 14). The title is an image; I read it from `papers/bridge/png/wp441_p-17.png`.
- Prior: uniform g on (-1, 1), stored as 201 equispaced points (the grid statement quoted in A1 covers "this and all the subsequent figures"). Text: "In Figures 3 and 4 we see solutions with a uniform g, where the truncation involves a sharply discontinuous dropoff in the density g at the boundaries of its support." The plotted g is flat at about 0.005 per grid point (1/201).
- Loss: quadratic. Title: "Weighted conditional pdf's, uniform g, quadratic U, κ = 1".
- Parameter: kappa = 1 nat of mutual information at the optimum. The alpha value is not stated.
- Convention: problem (1) as in A1, alpha = beta in my notation, with U = -(y-x)^2 (Section 4 convention).
- Outcome to predict: the number of support points of x.

### Case A4. Same paper, Figure 4 (uniform prior)

- File and page: `cerge_wp441_matejka_sims_2011.pdf`, PDF page 18 (printed page 15). Title read from `png/wp441_p-18.png`.
- Prior, loss and grid: as in A3 (uniform on (-1, 1), 201 points, quadratic U).
- Parameter: title "Weighted conditional pdf's, uniform g, quadratic U, κ = 1.2" (nats). The alpha value is not stated.
- Outcome to predict: the number of support points.

### Case A5. Jung, Kim, Matejka and Sims, "Discrete actions in information-constrained decision problems" (RES 2019), truncated-normal tracking example

- Files: `jkms_discrete_actions_cerge_home.pdf` (from home.cerge-ei.cz/matejka, header "Date: January 23, 2019."), Section V.3, PDF pages 16 to 20, Figure 1 on PDF page 20. The same text appears in `sims_wayback_DiscreteTracking2PropX4.pdf` (RES-formatted, "Review of Economic Studies (2020) 01, 1–44", Sims' site via the Wayback Machine) and in the 2015 draft `sims_wayback_DiscreteTracking2PropX2.pdf`. I did not obtain the typeset RES article (OUP returned HTTP 403).
- General problem (PDF pages 6 to 7):
  > "max E[U(X, Y)]− λI(X, Y) , (1)"
  > "U(x, y) is the objective function being maximized, and λ is the the cost of information."
  In the appendix monopolist example the unit is stated: "With a utility cost λ per nat (the unit of measurement for mutual information when log base e is used in deﬁning it)".
- Prior:
  > "Formally the problem is the one-dimensional version of the tracking problem of Theorem 1, with V(z) = −z2 and g either a N(0, σ2 y ) density or that same density truncated at y =±3σy."
  > "Now consider the problem with the prior distribution of Y truncated at±3σy."
  > "For example, suppose EY = 0, σ2 y = 1 and λ = .5. Then the untruncated solution makes ω2 = .5 and gives X a N(0, .5) distribution."
  So Y ~ N(0, 1) truncated to [-3, 3].
- Discretisation (footnote 7): "These numbers are based on solving the problem numerically with a grid of one thousand points between -3 and 3. They may not be accurate to more than about 3 decimal places as approximations to the continuously distributed problem."
- Parameter as stated: lambda = .5. Figure 1 caption: "Weighted conditional pdf's for Y in tracking problem, λ = .5".
- Convention warning: the paper's own normalisations disagree, and I checked all three statements.
  1. The text says V(z) = -z^2, and Corollary 2.1 says "X is normal and continuously distributed unless λ > 2σ2". That fits U = -(x-y)^2 with a cost of lambda per nat, which gives conditional variance lambda/2.
  2. Equation (15) as printed, "max ω<σy ( − 1/2 ω2− λ(log σ2 y− log ω2) )", followed by "Taking ﬁrst-order conditions it is easy to see that the solution is ω2 = λ." Differentiating the printed objective gives omega^2 = 2 lambda, so the printed equation and the stated solution do not match.
  3. The worked numbers pin the operating point: "the untruncated solution makes ω2 = .5"; "If we set X = .5(Y + ε), with ε∼ N(0, 1) and independent of Y, we would be using the same formulas as in the untruncated case, and would achieve almost the same result, E[(Y− X)2] = .49932 instead of .5". The appendix R code for tracking also uses a half: "U2drec <- function (x,y) { -.5 * sum((x-y)^2) }".
  My reading: this example behaves as U = -(1/2)(x-y)^2 with lambda per nat. The untruncated conditional variance is omega^2 = 0.5, so in my notation beta = 1/(2 omega^2) = 1 (nats per unit squared error). That is my derivation from the quoted omega^2 = .5, not a statement in the paper.
- Outcome to predict: the number of support points of X.

### Case A6. Rose, "A mapping approach to rate-distortion computation and analysis", IEEE Trans. IT 40 (1994) 1939, Figure 1 (a sweep over beta)

- File: `rose1994_mapping_approach_RD_TIT.pdf` (from web.ece.ucsb.edu/publications/rose/pubs/pub50-T-IT94.pdf). Figure 1 is on PDF page 7 (journal page 1945). The figure was read from `png/rose94_p-07.png` (250 dpi) and `png/rose94_600-07.png` (600 dpi).
- Functional and slope convention (journal page 1941, read from the rendered page `png/rose94_p03_top.png`):
  > "F(q) = −(1/β) ∫dx p(x) log ∫dy q(y) e^{−βd(x,y)}. (3)"
  > "The choice of the positive parameter β rather than the somewhat more common negative slope parameter s = −β is for reasons that will become obvious when the relation to statistical mechanics is discussed ... Another nonessential change is that we divide the usual functional by β."
  The log and exponential are natural, so beta is the slope in nats per unit distortion. This is the same as beta in my notation.
- Distortion (page 1944): "we restrict our derivation to scalars and to the squared error distortion measure d(x, y) = (x − y)2".
- Source: uniform on [-20, 20]. Caption: "Fig. 1. This phase diagram was produced by simulation on a uniform source [−20, 20]. Distortion versus β on logarithmic scale clearly shows the critical β where phase transitions occur. The cardinality (number of symbols) of the effective reproduction alphabet is marked within the region of the corresponding phase. Note that the apparent discontinuity of the transitions is due to the discrete jumps in β, and to the fact that in the simulation they occur slightly later than they should."
- Axes: the horizontal axis is beta on a log scale, with tick labels 0.001, 0.01, 0.1 and 0.5, covering beta from 0.001 to 0.5. The vertical axis is D on a log scale, with tick labels 1.1, 3, 10, 30 and 100.
- Computation (page 1948, said of "the above example", the same uniform source): "an exponential annealing schedule β(n + 1) = 1.01β(n), in the range 0.001 ≤ β < 0.5, was generated in 14 h." The convergence-threshold exponent printed on that line is illegible in my extraction. The algorithm is Rose's mapping approach: deterministic annealing over a finite set of reproduction symbols, with new symbols added when a split condition is met.
- Theory statement on the same page (a condition, not a reported number): "Phase transitions occur so as to maintain the cluster variance below 1/2β. As we increase β, when a cluster's variance becomes equal to 1/2β, it splits into smaller clusters, and the number of symbols increases."
- Outcomes to predict: (i) the cardinality in each phase across beta in [0.001, 0.5]; (ii) the critical beta values at which the cardinality changes. The figure draws each critical value as a vertical line.

### Case A7. Chen, Wu, Ye, Wu, Zhang, Wu and Bai, "A Constrained BA Algorithm for Rate-Distortion and Distortion-Rate Functions" (arXiv 2305.02650v2, CSIAM Trans. Appl. Math.), Figure 3 left, D = 4

- File: `arxiv_2305.02650.pdf`, Section 4.3 on PDF pages 25 to 26, Figure 3 on PDF page 26 (read from `png/cba_p-26.png`).
- Problem (PDF page 2): "R(D) := min I(X;Y) s.t. EPXY [d(X,Y)] ≤ D. (1.1)" with Lagrangian "L[λ]RD(PY|X) := I(X;Y)+ λ EPXY [d(X,Y)], (1.2) for each fixed multiplier λ∈R+. Geometrically, λ corresponds to the slope of the tangent line of the RD curve." (lambda restored from the pypdf text.) The log base is not stated anywhere I found.
- Source and distortion (Section 4.3): "we concentrate on instances characterized by squared error distortion measures. Specifically, our focus gravitates towards the uniform source ... We consider the uniform source on interval [-8,8] and conduct experiments with different discretization parameters, namely K = 20,40,80,160. The corresponding results are illustrated below in Figure 3."
- Discretisation rule, stated in Section 4.1 for the Gaussian and Laplacian runs: "we truncate the continuous probability distribution of the input source X within an interval [-L,L] and discretize the interval using a set of uniform grid points {xi}K i=1: xi = -L+(i-1/2)Δ, Δ = 2L/K ... yj = -L+(j-1/2)Δ~, Δ~= 2L/N ... We set L = 8, K = 100 and N = K for these sources." Section 4.3 does not restate N for the uniform runs. That N = K there is my assumption (UNCHECKED).
- Parameter as stated: target distortion D = 4. Caption: "Figure 3: The discrete optimal reproduction produced by the CBA algorithm for the cases of distortion D = 4 (Left) and D = 8 (Right)." Curves are shown for K = 20, 40, 80 and 160.
- Reported rate at this D (a rate, not a count), Table 3, CBA at D = 4: 0.7366 (K=20), 0.7363 (K=40), 0.7361 (K=80), 0.7360 (K=160). My unit check: the Shannon lower bound in nats for uniform [-8,8] at D = 4 is ln 16 - 0.5 ln(2 pi e 4) = 0.660, and at D = 2 it is 1.007 against the reported 1.058 to 1.060. The rates are consistent with nats and d = (x-y)^2. That is my computation, not the paper's statement.
- Outcome to predict: the number of mass points of the optimal reproduction.

### Case A8. Same paper, Figure 3 right, D = 8

- Same file, page and conventions as A7. Uniform source on [-8, 8], squared error, target distortion D = 8, K = 20, 40, 80 and 160.
- Reported rates, Table 3, CBA at D = 8: 0.4257 (K=20), 0.4244 (K=40), 0.4244 (K=80), 0.4243 (K=160).
- Outcome to predict: the number of mass points.

### Cases A9 to A12. Chen, Tang, Wu, Wu, Wu and Zhang, "Computing Rate–Distortion Functions of Continuous Memoryless Sources via Discrete Algorithms ..." (Entropy 28:280, 2026, doi 10.3390/e28030280); arXiv v1 is 2405.00474 ("On Convergence of Discrete Schemes for Computing the Rate-Distortion Function of Continuous Source", Chen, Wu, Zhang, Wu and Wu)

- Files: `pmc13025186.html` and `pmc13025186.txt`, the tag-stripped PMC full text. The PMC PDF sits behind a proof-of-work page and MDPI returned 403, so I used the HTML. Figure image: `png/chen2026_entropy_fig1_g001.jpg`. The arXiv v1 PDF, `arxiv_2405.00474.pdf`, has the same experiment in Section V, with Figures 1 and 2 on PDF page 19 (`png/a2405-19.png`).
- Objective (Entropy version, eq. (2) and (4)):
  > "F ( r , β ) ≜ − ∫ log ∫ exp − β ρ ( x , y ) d r ( y ) d p ( x ) − β D . (2)"
  > "f ( r ) ≜ − ∫ log ∫ exp − β ρ ( x , y ) d r ( y ) d p ( x ) , (4)"
  Natural log and exp, so beta is the slope in nats per unit distortion, the same as Rose's (3) up to the 1/beta factor.
- Experiment (Entropy Section 2.3):
  > "We conduct experiments on a uniform source to confirm the convergence shown in this section. We consider the uniform source on interval [ − 8 , 8 ] and solve the corresponding discrete problems with different discretization parameters, with the node number varying according to n = 20 , 40 , 80 , 160 for the discretization of Y . In order to ensure accuracy for evaluating the integrals with respect to p ( x ) , we fix a sufficiently large node number m = 300 for X . We use both the BA and CBA algorithms to solve the discrete problems ( 6 ) and ( 7 ), respectively."
- Distortion: Section 2.3 does not restate rho. The surrounding text says "for continuous sources under the squared-error distortion measure, the optimal reproduction distribution r is usually discrete [ 17 , 24 ]", and Section 4.1 treats "the squared-error distortion ρ ( x , y ) = ( x − y ) 2". That the Section 2.3 runs use rho = (x-y)^2 is my inference (UNCHECKED). The sister paper (A7 and A8) states squared error for the same uniform [-8, 8] source.
- Y grid range for these runs: not stated. The figure axes span [-8, 8].
- Caption (Entropy Figure 1): "The discrete optimal reproduction distribution produced by the BA algorithm for slope β = 0.1 ( upper left ) and β = 0.2 ( upper right ), and by the CBA algorithm for target distortion D = 4 ( lower left ) and D = 3 ( lower right ). For clearer visualization, the log 2 scale of the density is used on the vertical axis."
- Case A9: BA, slope beta = 0.1. Case A10: BA, slope beta = 0.2. Case A11: CBA, target D = 4. Case A12: CBA, target D = 3. In every panel, curves are shown for n (legend "K") = 20, 40, 80 and 160.
- Outcome to predict for each: the number of mass points of the optimal reproduction.

### Case A13. Mao, Gray and Linder, "Rate-Constrained Simulation and Source Coding IID Sources" (arXiv 1008.2008v2, 2011), uniform example

- File: `arxiv_1008.2008.pdf`, Section on test sources, PDF pages 10 to 12, Tables II and IV on PDF page 11 (read from `png/mgl-11.png`).
- Source and distortion: uniform on [0, 1). Table II is titled "UNIFORM [0, 1) EXAMPLE" and Table IV refers to "THE UNIFORM (0, 1) SOURCE". The distortion is mean squared error: "The uniform IID source is of interest because it is simple, there is no exact formula for the rate-distortion function with respect to mean-squared error and hence it must be found by numerical means".
- Parameter as stated: rate R = 1 bit (Table II column "Rate(bits)"). The optimal reproduction distribution was computed with Rose's algorithm ("The Rose algorithm yielded a Shannon optimal distribution ... for R = 1"). The slope is not stated.
- Reported distortion at this rate (a distortion, not a count), Table II: "D_X(R) | 1 | 0.0173 | 6.84" (Rate in bits, MSE, SNR in dB).
- Outcome to predict: the size of the optimal reproduction alphabet at R = 1 bit.

---

## Partially specified cases: bounded prior, but the density is not fully stated

### Case P1. CERGE-EI WP 441, Figure 7 (truncated Cauchy prior)

- File and page: `cerge_wp441_matejka_sims_2011.pdf`, PDF page 19 (printed page 16), read from `png/wp441_p-19.png`.
- Prior: "Figures 7 and 8 show solutions for a truncated Cauchy g with quadratic U." Support (-1, 1) on the 201-point grid, per the statement quoted in A1. The Cauchy scale parameter is NOT stated in the text.
- Loss: quadratic (title "Cauchy g, U = a2"; the minus sign is missing in the title).
- Parameter: kappa = 0.34 nats (title "Weighted conditional pdf's, Cauchy g, U = a2, κ = 0.34"). The alpha value is not stated.
- Outcome to predict: the number of support points.

### Case P2. Same paper, Figure 8

- PDF page 20 (printed page 17). Same prior and loss as P1. Title: "Weighted conditional pdf's, Cauchy g, U = a2, κ = 0.66".
- Outcome to predict: the number of support points.

---

## Auxiliary cases outside the stated scope (loss is not quadratic)

These are recorded in case a universality test wants them. They do not test a squared-error law.

### Case X1. CERGE-EI WP 441, Figure 5

- PDF page 18 (printed page 15). Uniform g on (-1, 1), 201 points. Loss "U(z) = −z1.1" (text: "Figures 5 and 6 display solutions with a uniform g and U(z) = −z1.1. This makes the peak of the objective function sharp"). Title: "Weighted conditional pdf's, uniform g, U= − a1.1, κ = 0.59" (nats).
- Outcome to predict: the number of support points.

### Case X2. Same paper, Figure 6

- PDF page 19 (printed page 16). Same prior and loss as X1. Title: "Weighted conditional pdf's, uniform g, U= − a1.1, κ = 0.7".
- Outcome to predict: the number of support points.

### Case X3. JKMS (2019), Appendix C.2, risk-averse monopolist

- File: `jkms_discrete_actions_cerge_home.pdf`, PDF pages 52 to 55.
- Problem: "max E [ log ( (X− W)q(X) )] − λI(X, W) , (25)" with "a utility cost λ per nat", demand "q(x) = x−θ for x > 0". Prior: "We solve this problem numerically with the distribution of W a Beta(4,4) distribution scaled to cover the interval (0,10)." Parameters: "When θ = 1.5, and λ = .05".
- Outcome to predict: the number of support points of X.

---

## Sources checked that gave no usable case, or could not be obtained

- Rose, Gurewitz and Fox, Phys. Rev. Lett. 65, 945 (1990): NOT OBTAINED. journals.aps.org returned HTTP 401. The Semantic Scholar API reports openAccessPdf status "CLOSED". It is not in CaltechAUTHORS. Whether it contains a 1-D uniform example is UNCHECKED.
- Rose, Gurewitz and Fox, "Vector quantization by deterministic annealing", IEEE Trans. IT 38, 1249 (1992): NOT OBTAINED. The Semantic Scholar API reports "CLOSED". Rose's 1998 Proc. IEEE review reproduces its Figures 1 and 2 ("From [90]", and [90] is this paper). Those show a 2-D mixture of six Gaussians, not a 1-D source. Whether the 1992 paper also has a 1-D example is UNCHECKED.
- Rose, Caltech PhD thesis 1991, "Deterministic annealing, clustering, and optimization" (`rose1991_caltech_thesis.pdf`, a scanned PDF read page by page from PNG renders): the Chapter 3 phase-transition examples (Figures 3.1 to 3.3) are 2-D six-Gaussian data. No 1-D uniform case.
- Rose, Proc. IEEE 86 (1998) 2210 (`rose1998_deterministic_annealing_ProcIEEE.pdf`): all numerical figures are 2-D Gaussian mixtures. The rate-distortion section has theory and no numerical uniform example.
- Yang, Eckstein, Nutz and Mandt, arXiv 2310.18908 (`arxiv_2310.18908.pdf`): examples are Gaussian, circle deconvolution and physics or speech data. No uniform scalar source.
- Jung, Kim, Matejka and Sims: the 2-D tracking examples (circle and square truncation) and the portfolio examples are outside scope (2-D or non-tracking). The unit-circle example with 0/-infinity loss is analytic.
- Matejka, "Rationally Inattentive Seller" (`matejka_RI_seller.pdf`): Example 2 (uniform prior on (0,1), loss -(x-p)^2) compares hand-drawn 2-price and 3-price strategies at an unstated capacity. It is an illustration with no optimum at a stated parameter. The numerical Figure 4 uses a non-quadratic constant-elasticity profit.
- Mackowiak, Matejka and Wiederholt RI survey (`matejka_RIsurvey.pdf`): its Figure 3 is illustrative only.
- Saint-Paul, "A quantized approach to rational inattention" (`saintpaul_tse_wp10-144.pdf`): restricts policies to deterministic partitions (entropy-constrained quantization), which is not the Shannon rate-distortion or RI optimum.
- Bonnaire, Decelle and Aghanim, arXiv 2010.07955: Gaussian-cluster datasets only.
- Wu, Ye et al., arXiv 2212.10098, and arXiv 2307.00246: no uniform-source example.
- Fix (1978, Allerton), Gray and Neuhoff, Chang and Davisson: not found online in the searches listed below. No specific paper with a uniform-source count surfaced for these names.

---

## Search queries and files (full log in `papers/bridge/search_log.txt`)

Web searches (WebSearch tool), in order:
1. Jung Kim Matejka Sims "Discrete actions in information-constrained decision problems" pdf
2. Rose Gurewitz Fox "Statistical mechanics and phase transitions in clustering" pdf
3. sims.princeton.edu yftp "Discrete actions" information-constrained Jung Kim Matejka
4. "Discrete Actions in Information-Constrained Decision Problems" working paper pdf cerge-ei OR nber OR ssrn
5. (extended) "Discrete actions in information-constrained decision problems" pdf download Jung Kim Matějka Sims 2019 Review of Economic Studies uniform prior quadratic
6. Rose "Vector quantization by deterministic annealing" IEEE Transactions on Information Theory 1992 pdf
7. Rose 1994 "A mapping approach to rate-distortion computation and analysis" pdf uniform source
8. "Rose" "Gurewitz" "Fox" 1990 clustering deterministic annealing "critical temperature" pdf
9. (extended) rate-distortion uniform source squared error optimal reproduction distribution discrete number of mass points Blahut-Arimoto slope
10. Kenneth Rose 1991 Caltech thesis "Deterministic annealing, clustering, and optimization" pdf thesis.library.caltech.edu
11. deterministic annealing uniform distribution interval critical temperatures number of clusters one-dimensional "uniform" phase transitions clustering 1-D
12. rational inattention uniform prior quadratic loss number of actions discrete as information cost varies figure tracking problem
13. Fix 1978 "rate distortion functions for squared error distortion measures" discrete reproduction bounded source
14. Saint-Paul "A quantized approach to rational inattention" pdf
15. "clustering" "uniform distribution" "critical temperature" annealing one-dimensional successive splits "free energy" Gaussian mixture phase transition 1D interval number of clusters temperature
16. arXiv rate-distortion optimal transport Sinkhorn uniform source squared error discrete reproduction phase transition slope Wu Ye Zhang
17. Sims 2010 "Rational inattention and monetary economics" handbook pdf uniform discrete support points figure
18. "rate-distortion" "uniform source" "squared error" optimal output distribution "finite number" points slope computed Blahut numerical example support size
19. Chang Davisson rate distortion computation continuous source uniform squared error reproduction alphabet

Direct repository queries: CaltechAUTHORS API (q = Gurewitz; deterministic annealing Rose; Rose vector quantization annealing; Statistical mechanics phase transitions clustering); Semantic Scholar API for DOIs 10.1103/PhysRevLett.65.945 and 10.1109/18.144705; Wayback CDX for sims.princeton.edu/yftp/RIDiscrete/*.

Files obtained (all in `papers/bridge/`):
- `cerge_wp441_matejka_sims_2011.pdf`, from www.cerge-ei.cz/pdf/wp/Wp441.pdf
- `jkms_discrete_actions_cerge_home.pdf`, from home.cerge-ei.cz/matejka/discrete_actions.pdf
- `sims_wayback_DiscreteTracking2Prop.pdf`, `sims_wayback_DiscreteTracking2PropX2.pdf` and `sims_wayback_DiscreteTracking2PropX4.pdf`, from web.archive.org copies of sims.princeton.edu/yftp/RIDiscrete/. The live site did not answer.
- `rose1994_mapping_approach_RD_TIT.pdf`, from web.ece.ucsb.edu/publications/rose/pubs/pub50-T-IT94.pdf. The first download was truncated; the resumed download is complete at 2,732,530 bytes.
- `rose1998_deterministic_annealing_ProcIEEE.pdf`, from www.ece.ucsb.edu/publications/rose/pubs/pub42-Proc11-98.pdf
- `rose1991_caltech_thesis.pdf`, from thesis.caltech.edu/2858/01/Rose_k_1991.pdf (scanned images, no text layer)
- `arxiv_2305.02650.pdf`, `arxiv_2405.00474.pdf`, `arxiv_2310.18908.pdf`, `arxiv_1008.2008.pdf`, `arxiv_2010.07955.pdf`, `arxiv_2212.10098.pdf` and `arxiv_2307.00246.pdf`, from arxiv.org/pdf/<id>
- `pmc13025186.html`, from pmc.ncbi.nlm.nih.gov/articles/PMC13025186/, and the figure `png/chen2026_entropy_fig1_g001.jpg`
- `matejka_RI_seller.pdf` and `matejka_RIsurvey.pdf`, from home.cerge-ei.cz/matejka/
- `saintpaul_tse_wp10-144.pdf`, from publications.ut-capitole.fr/3333/1/10-144.pdf
- Text extractions (`*.txt`, `*_pypdf.txt`) sit next to each PDF. Figure renders are in `png/`.
