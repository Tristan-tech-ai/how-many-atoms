# Bridge cases, reported outcomes (FILE B, SEALED)

Do not open until predictions for `bridge_specs.md` are frozen. Case labels match FILE A.

Quotes are from text I extracted myself (paths in FILE A). Figure readings are mine, taken from PNGs I rendered with pdftoppm, and are marked "reading". Support-point positions read off figures are approximate, about ±0.02 of the axis unit for the WP 441 figures and about ±0.2 for the [-8, 8] figures. Counts were read by counting asterisks or density spikes, and I checked each one against any count the paper states in words.

---

## Core cases

### A1. WP 441 Figure 1: truncated normal (sd 1/3, ±3 sd, on (-1,1)), quadratic, kappa = 0.33 nats
- **Reported count: 3 support points.**
- Quote (PDF page 15): "But the actual solution is supported on three points."
- Reading (`png/wp441_p-15.png`, `png/wp441_axes_stack.png` row 1): three asterisks, at about -0.36, 0.00 and +0.36, with three weighted conditional densities (red, blue, green).

### A2. WP 441 Figure 2: same prior, quadratic, kappa = 0.39 nats
- **Reported count: 5 support points.**
- Quote (PDF page 17, about Figures 3 and 4 in comparison): "the increase in κ in going from 4 to 6 support points is three times as large as that for going from 3 to 5 support points in the normal-g example." Also (PDF page 16): "the solution puts weight on more points, as in Figure 2."
- Reading (`png/wp441_p-16.png`, stack row 2): five asterisks, at about -0.44, -0.21, -0.01, +0.19 and +0.43, with five weighted densities. The paper notes the slight asymmetry is numerical imprecision.

### A3. WP 441 Figure 3: uniform on (-1,1), quadratic, kappa = 1 nat
- **Reported count: 4 support points.**
- Quote (PDF page 17): "the increase in κ in going from 4 to 6 support points is three times as large..." (the uniform-g pair, Figures 3 and 4).
- Reading (`png/wp441_p-17.png`, stack row 3): four asterisks, at about -0.71, -0.22, +0.23 and +0.71, with four weighted densities (black, red, green, blue). A pixel check at 300 dpi gave -0.709, -0.221, +0.226 and +0.715.

### A4. WP 441 Figure 4: uniform on (-1,1), quadratic, kappa = 1.2 nats
- **Reported count: 6 support points.**
- Quote: the same sentence as A3 ("going from 4 to 6 support points").
- Reading (`png/wp441_p-18.png`, stack row 4): six asterisks, at about -0.78, -0.40, -0.11, +0.10, +0.39 and +0.77, with six weighted densities (black, red, green, blue, cyan, magenta).

### A5. JKMS truncated normal N(0,1) on [-3,3], lambda = .5 (untruncated omega^2 = .5)
- **Reported count: 4 support points, at -1.0880, -.1739, .1739 and 1.0880.**
- Quote (`jkms_discrete_actions_cerge_home.pdf`, PDF pages 18 to 19; the same wording is in the RES-formatted X4 copy): "But numerical calculation shows that the optimal solution with this truncation and this cost of information has just four points of support for the X distribution: -1.0880, -.1739, .1739, 1.0880. It achieves almost exactly the same E[(Y− X)2] as in the untruncated case7, while using about 4% less information."
- Figure 1 (PDF page 20) shows four weighted conditional pdfs, consistent with the text.
- Footnote 7 caveat: grid of 1000 points on [-3, 3]; "may not be accurate to more than about 3 decimal places".

### A6. Rose 1994 Figure 1: uniform on [-20, 20], d = (x-y)^2, beta from 0.001 to 0.5
- **Reported phase labels (cardinality of the effective reproduction alphabet), left to right: 1, 2, 3, 4, then "..."** for the last region, from the fourth vertical line to beta = 0.5, which carries no number.
- **Critical beta values (reading of the four vertical lines):**
  - 1 to 2: beta ≈ 0.0039
  - 2 to 3: beta ≈ 0.0116
  - 3 to 4: beta ≈ 0.0274
  - 4 to the "..." region: beta ≈ 0.0725
- Further small unlabelled steps in the D(beta) curve inside the "..." region, at beta ≈ 0.139, 0.228, 0.339 and 0.475 (reading). The figure does not label them. They look like the same kind of step seen at the labelled transitions, but calling them transitions is my interpretation.
- How I read it: I rendered PDF page 7 at 600 dpi (`png/rose94_600-07.png`). I located the axis tick marks by pixel columns crossing the axis line: beta = 0.001 at the origin, 0.01 and 0.1 at ticks 599.5 px and 593.5 px per decade apart, and 0.5 at the predicted log position. I located the vertical transition lines as long dark pixel columns and converted by log interpolation. Pixel uncertainty is about ±3 px, which is about ±1.2% in beta. The caption warns: "the apparent discontinuity of the transitions is due to the discrete jumps in β, and to the fact that in the simulation they occur slightly later than they should." The schedule stated for this source is beta(n+1) = 1.01 beta(n), so each transition can sit up to about 1% late for that reason alone.
- My comparison, not in the paper: the source variance is 40^2/12 = 133.3, so 1/(2 sigma^2) = 0.00375. The first line read at 0.0039 is about 4% later, consistent with the caption. The flat D level left of the first line sits above the 100 tick, consistent with D = 133.3.

### A7. Chen et al. CBA (arXiv 2305.02650) Figure 3 left: uniform [-8, 8], squared error, D = 4
- **Reported count: 3 mass points.**
- Reading (`png/cba_p-26.png`): three spikes of the reproduction pmf (log vertical axis, 10^0 down to 10^-60), at about -4.9, 0.0 and +4.9. The spikes sharpen as K goes from 20 to 160. The text states no count in words.

### A8. Same, Figure 3 right: D = 8
- **Reported count: 2 mass points.**
- Reading: two spikes, at about -3.8 and +3.8. The K = 20 curve has a broad shallow bump in the middle at about 10^-20, which is not a mass point, so I did not count it.

### A9. Chen et al. (Entropy 2026 / arXiv 2405.00474) BA, slope beta = 0.1, uniform [-8, 8]
- **Reported count: 3 mass points.**
- Reading (`png/chen2026_entropy_fig1_g001.jpg` upper left, zoom `png/chen2026_UL_zoom.png`; the same panel is arXiv Fig. 1 left, `png/a2405-19.png`): three spikes, at about -4.8, 0.0 and +4.7.

### A10. Same, BA, slope beta = 0.2
- **Reported count: 4 mass points.**
- Reading (upper right, `png/chen2026_UR_zoom.png`; arXiv Fig. 1 right): four spikes, at about -5.8, -1.8, +1.7 and +5.7. The inner pair carries visibly less mass than the outer pair.

### A11. Same, CBA, target D = 4
- **Reported count: 3 mass points.**
- Reading (lower left; arXiv Fig. 2 left): three spikes, at about -5.0, 0.0 and +5.0.

### A12. Same, CBA, target D = 3
- **Reported count: 4 mass points.**
- Reading (lower right; arXiv Fig. 2 right): four spikes, at about -5.4, -1.3, +1.3 and +5.3.
- The text of both versions states no count in words. It says only: "the convergence of the solutions of discrete problems to discrete reproduction distributions is clearly demonstrated".

### A13. Mao, Gray and Linder: uniform [0, 1), MSE, R = 1 bit
- **Reported count: 3 reproduction points: y = 0.2, 0.5 and 0.8, with p = 0.368, 0.264 and 0.368 (Table IV).**
- Quote (`arxiv_1008.2008_pypdf.txt`, PDF pages 10 to 12): "The Rose algorithm yielded a Shannon optimal distribution with an alphabet of size 3 for R = 1 . The points and their probabilities are shown in Table IV." Table IV: "y 0.2 0.5 0.8 / pY (y) 0.368 0.264 0.368".
- Note: the points are printed to one decimal, so they are probably rounded.

---

## Partially specified cases (Cauchy scale not stated)

### P1. WP 441 Figure 7: truncated Cauchy on (-1,1), quadratic, kappa = 0.34
- **Reported count: 4 support points** (reading). Asterisks at about -0.62, 0.00, +0.50 and +0.77, so the configuration is asymmetric. The four weighted densities are green, purple, blue and red.
- Quote (PDF page 18): "most of the probability distribution for x concentrates on x = 0, even as the information constraint is relaxed, and that as the constraint is relaxed, more small-probability x values in the tails of the distribution appear."

### P2. WP 441 Figure 8: same prior, kappa = 0.66
- **Reported count: 5 support points** (reading). Asterisks at about -0.74, -0.34, 0.00, +0.34 and +0.73.

---

## Auxiliary, non-quadratic cases

### X1. WP 441 Figure 5: uniform on (-1,1), U = -|z|^1.1, kappa = 0.59
- **Reported count: 3 support points** (reading). Asterisks at about -0.63, 0.00 and +0.63.

### X2. WP 441 Figure 6: same, kappa = 0.7
- **Reported count: 5 support points** (reading). Asterisks at about -0.72, -0.37, -0.02, +0.31 and +0.71.

### X3. JKMS risk-averse monopolist: W ~ Beta(4,4) on (0,10), theta = 1.5, lambda = .05 per nat
- **Reported count: 6 support points.**
- Quote (PDF page 54): "We ﬁnd numerically that X is distributed on a support of 6 points: 0.845, 1.790, 3.254, 5.648, 9.796, and 17.061, with probabilities .000028, .000385, ,00334, .025203 .170879, and .800163."
- Caveat quoted on PDF pages 54 to 55: "it is possible, even likely, that the fully optimal solution for the continuous version of the problem has a countable inﬁnity of points of support with a limit" point (the text continues onto the next page).
