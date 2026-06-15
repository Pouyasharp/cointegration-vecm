# Cointegration and VECM — A PhD-Level Deep Dive

> **A focused, runnable treatment of cointegration and Vector Error
> Correction Models (VECM).** Standalone deep-dive of the
> cointegration half of Stage 3 of the
> [`econometrics-deep-research`](https://github.com/Pouyasharp/econometrics-deep-research)
> parent curriculum.
>
> The ARIMA / SARIMAX half of Stage 3 is in the sibling repo
> [`arimax-sarimax-tutorial`](https://github.com/Pouyasharp/arimax-sarimax-tutorial).
> The causal-inference half of Stage 4 is in
> [`causal-inference`](https://github.com/Pouyasharp/causal-inference).

---

## §0 — Meta, learning outcomes, and tool stack

### 0.1 What you should be able to do at the end

By the end, you can:

1. **Diagnose spurious regression** in two I(1) series and explain
   why OLS is inconsistent for the slope.
2. **Test for cointegration** using Engle-Granger 2-step and the
   Johansen trace test, and choose between them based on the data
   structure.
3. **Estimate a VECM** with the right number of cointegrating
   relations, lags, and deterministic terms.
4. **Interpret the cointegrating vector** as a long-run equilibrium
   and the loadings as speeds of adjustment.
5. **Test for Granger causality** within a VECM and the
   corresponding "weak exogeneity" and "strong exogeneity" tests.
6. **Test for structural breaks** in the cointegrating relationship
   (Gregory-Hansen, Hatemi-J).

### 0.2 Tool stack

| Layer | Python | R | Stata |
|---|---|---|---|
| Unit-root tests | `statsmodels.tsa.stattools.adfuller` | `urca::ur.df` | `dfuller` |
| Engle-Granger | `statsmodels` (manual) | `urca::ca.po` | `egranger` |
| Johansen | `statsmodels.tsa.vector_ar.vecm.coint_johansen` | `urca::ca.jo` | `vecrank` |
| VECM fit | `statsmodels.tsa.vector_ar.vecm.VECM` | `tsDyn::VECM`, `vars::VECM` | `vec` |
| Structural break | (manual) | `urca::ur.za` | — |
| IRF / FEVD | `statsmodels.tsa.vector_ar.irf` | `vars::irf` | `irf create` |

The `statsmodels` VECM API is **string-coded for the deterministic
terms**: `"nc"` (no constant), `"ci"` (restricted constant in CEE),
`"co"` (constant outside CEE), `"lo"` (linear trend outside), `"li"`
(linear in CEE). The `R` `urca` package uses integer codes; the
two systems are equivalent but the strings in `statsmodels` are a
frequent source of confusion. See `references/statsmodels-vecm-api-quirks.md`
for the gotcha bank.

---

## §1 — Spurious regression

The classical result (Granger & Newbold 1974, Phillips 1986):

**If `x_t` and `y_t` are independent I(1) random walks, then OLS of
`y_t` on `x_t` gives:**
- `R² → non-degenerate random variable` (NOT going to 0 in large T)
- `t-statistic does NOT have a t-distribution` (the limiting
  distribution is non-standard — the Dickey-Fuller-type distribution)
- **The slope is garbage** — no economic meaning

This is the *spurious regression problem*. It was first documented
empirically (Granger & Newbold 1974 found "too many" t-stats > 2 in
macroeconomic regressions) and the theory was provided by Phillips
(1986).

**Demo 01** demonstrates the spurious case: simulate two independent
random walks and show the OLS slope is huge with a huge t-stat.

**Diagnostic:** the diagnostic is the *cointegrating residual* — if
the OLS residual is I(1), the regression is spurious; if the OLS
residual is I(0), the regression is *not* spurious (it is either
genuine or cointegrated).

---

## §2 — Cointegration

**Definition (Engle & Granger 1987).** Two series `x_t, y_t` are
*cointegrated* of order `(1, 1)` if:
1. Both are I(1) (integrated of order 1)
2. There exists a non-zero vector `β = (1, -b)` such that
   `y_t - b * x_t` is I(0) (stationary)

The vector `β` is called the **cointegrating vector**. The I(0)
combination `y_t - b * x_t` is called the **cointegrating residual**
or the **error-correction term**.

**Intuition:** both series are highly persistent on their own, but
they are *bound together* by a long-run equilibrium. The deviations
from the equilibrium are temporary (stationary).

**Why it matters:**
- **No-arbitrage conditions** — two prices that should move together
- **Demand-supply** — a price-quantity relationship
- **Long-run marketing-macro links** — a brand's ad-spend share and
  the macro ad-spend pool

**Two ways to test for it:**
1. **Engle-Granger 2-step** (residual-based, simple, biased in
   finite samples) — see §3
2. **Johansen** (maximum likelihood, multi-equation, no residual
   bias) — see §4

---

## §3 — Engle-Granger 2-step

**Step 1:** OLS the levels: `y_t = β_1 * x_t + u_t`. Compute
`û_t = y_t - β̂_1 * x_t`.

**Step 2:** ADF on `û_t`:
- Reject unit root → cointegrated
- Fail to reject → either no cointegration or the OLS slope is
  too noisy to detect it

**Properties of the EG estimator:**
- `β̂_1` is **super-consistent** (Engle & Granger 1987): converges
  to the true `β` at rate `T` (not the usual `√T`).
- The bias in finite samples can be substantial for moderate `T`
  (e.g., `T < 200`).
- The ADF test on the residual has **non-standard critical values**
  (the Engle-Granger critical values, which are more negative than
  the standard ADF critical values).
- The EG test **assumes k=1** — if you have more than 2 series, the
  EG test loses power and is inconsistent for the additional
  cointegrating vectors. Use Johansen (§4).

**Engle-Granger critical values (Engle & Yoo 1987):** for 2-variable
case and constant in the regression, the 5% critical value is
approximately **-3.37** (vs the standard ADF 5% of -2.86 for `c`
regression). The `statsmodels` `adfuller` reports the standard
critical values; for EG, you need to manually compare against the
EG critical values.

**Demo 01** runs the full EG procedure on a spurious pair and a
cointegrated pair, showing the contrast.

---

## §4 — Johansen trace test

**Setup:** form a VAR(p) in levels for all K variables. The
VAR(p) in levels is equivalent to a VECM(p-1).

**Two test statistics:**

**Trace test:** H0: rank ≤ r. Compute
`λ_trace(r) = -T * Σ_{i=r+1}^K ln(1 - λ̂_i)`. The largest r for
which `λ_trace(r) > CV(r)` is the rank.

**Max-eigenvalue test:** H0: rank = r. Compute
`λ_max(r) = -T * ln(1 - λ̂_{r+1})`. Decide sequentially: at each r,
if `λ_max(r) > CV(r)`, increment r; otherwise stop.

**Properties:**
- The two tests can disagree. The default in practice is the **trace
  test** (it has better power in small samples).
- The deterministic term choice matters:
  - `det_order=0` (`"nc"` in `statsmodels`): no constant — only valid
    if the cointegrating relation has zero mean (rarely the case)
  - `det_order=1` (`"ci"`): restricted constant IN the cointegrating
    equation (Johansen's default)
  - `det_order=2` (`"co"`): constant OUTSIDE the cointegrating
    equation (use when all series have a non-zero mean but the
    long-run relationship has zero mean)
  - `det_order=-1` (`"lo"`): linear trend outside (use for trending
    data)

**Demo 02** runs the trace + max-eig on a 3-variable system with
k=1 cointegrating relation planted.

**Normalization:** the cointegrating vector is only identified up to
a scale. Normalize by setting `β[0] = 1` (or by the sign convention
that the variable of interest has a positive coefficient).

---

## §5 — VECM estimation

**Representation:** if there are K variables and `r` cointegrating
relations, the VECM(p-1) is:
```
Δy_t = α * β' * y_{t-1} + Σ_{i=1}^{p-1} γ_i * Δy_{t-i} + μ + Φ * D_t + ε_t
```
- `β` is (K × r) — the cointegrating matrix
- `α` is (K × r) — the loadings (speed of adjustment)
- `γ_i` are (K × K) — the short-run coefficients on lagged diffs
- `μ` is a constant (if `det_order` allows)
- `D_t` are deterministic terms (seasonal dummies, etc.)

**Interpretation:**
- `β` is the long-run equilibrium. E.g., for two variables,
  `β = (1, -0.5)` means the long-run relationship is
  `y_{1,t} = 0.5 * y_{2,t}`.
- `α` is the speed of adjustment. E.g., `α[0, 0] = -0.4` means
  that y₁ adjusts toward the long-run equilibrium at rate 40% per
  period (i.e., the disequilibrium is reduced by 40% each period).
- The `α` and `β` products are identified together — only the
  product `α β'` matters for the dynamics.

**Identification:** `β` is identified up to a column scaling. The
`α` is identified once `β` is normalized. Standard normalizations:
- Set `β[0, 0] = 1` (the first variable is the "dependent" one)
- Or: normalize so that the diagonal of `α β'` is `(-1, 0, 0, ...)`
  (the "Procrustes" transformation)

**Demo 03** fits a VECM(1) on a 2-variable system with planted
β = (1, -0.5), α = (-0.4, 0.2), and γ[1,1] = γ[2,2] = 0. Reports
all three sets of coefficients and compares to the true planted
values.

---

## §6 — Interpretation

**Long-run equilibrium:** the row-space of `β` is the long-run
equilibrium. For a single cointegrating relation, the
cointegrating error `ecm_t = β' y_t` is the deviation from
equilibrium.

**Speed of adjustment:** the `α` matrix tells you how each variable
responds to the disequilibrium.
- If `α[i, j] = 0`, variable i does not adjust to the j-th
  cointegrating error.
- If `α[i, j] < 0`, variable i falls when the ecm is positive
  (i.e., the system is moving back toward equilibrium).
- The **half-life of a shock** is `log(0.5) / log(1 - |α * β'[i, i]|)`.

**IRF / FEVD:** the impulse response to an ecm shock shows how the
system returns to equilibrium. The FEVD decomposes the forecast
error variance.

**Demo 03** also computes the autocorrelation of the ecm to estimate
the half-life.

---

## §7 — Granger causality in the VECM

**Three concepts (Engle-Hendry-Richards 1983, Johansen 1992):**

**1. Granger causality in the long run** = whether variable j is in
the cointegrating vector for variable i (i.e., whether the i-th row
of `β` is non-zero for column j).

**2. Weak exogeneity** of variable i = whether the i-th row of `α`
is zero. If `α[i, :] = 0`, variable i does not adjust to the
cointegrating errors; it is weakly exogenous for the cointegrating
parameters.

**3. Strong exogeneity** of variable i = weak exogeneity + variable i
is not Granger-caused by the other variables in the short run (i.e.,
the γ coefficients on variable i are all zero in the other equations).

**Practical implications:**
- If a variable is **strongly exogenous**, you can treat it as
  exogenous in the model — drop its equation and use a
  single-equation ECM.
- If a variable is **NOT weakly exogenous**, it must be in the
  conditional model — you cannot drop its equation.
- The Engle-Hendry-Richards (1983) result: under strong exogeneity,
  a single-equation error-correction model is consistent for the
  long-run parameters even though it ignores the multivariate
  dynamics.

**Demo 04** simulates a 2-variable VECM where y₂ is strongly
exogenous (α₂ = 0 and γ[y₂, y₁] = 0) and y₂ Granger-causes y₁ in
the short run. The demo tests all three concepts.

---

## §8 — Structural breaks in cointegration

**Gregory-Hansen (1996) test:**
- H0: no cointegration (or cointegration with no break)
- H1: cointegration with a LEVEL SHIFT at an UNKNOWN break date
- The break date `t_b` is estimated by minimizing the ADF statistic
  on the regime-by-regime residual
- Three versions: level shift, level + trend shift, regime shift

**Hatemi-J (2008) test:**
- Like Gregory-Hansen but allows a TREND BREAK in addition to the
  level shift
- Useful for variables with a deterministic trend

**Bai-Perron (1998, 2003) multiple-break test:**
- Tests for multiple structural breaks in a (cointegrated) regression
- The break dates are estimated by sequential F-tests

**Demo 05** implements a simplified Gregory-Hansen (level shift
version). It shows the regime-by-regime ADF statistic as a function
of the candidate break date, and the chosen break.

**Caution:** Gregory-Hansen and Bai-Perron can be over-sensitive to
the data range — always plot the cointegrating residual and look
for the break visually.

---

## §9 — Self-assessment rubric

1. You can explain why spurious regression happens and what its
   signature is in OLS output.
2. You can run the Engle-Granger 2-step and interpret the ADF test
   on the residual using the EG-specific critical values.
3. You can run the Johansen trace test, report the rank, and
   explain the choice of `det_order`.
4. You can fit a VECM with the right specification (p, r,
   det_order) and interpret the cointegrating vector.
5. You can run the Granger-causality test within a VECM and
   report weak and strong exogeneity.
6. You can identify the case where a single-equation ECM is
   consistent and defend the choice.
7. You can detect a structural break in the cointegrating
   relationship using Gregory-Hansen.
8. You can defend the choice of EG vs Johansen vs structural-break
   test for a given problem.

---

## §10 — Reading list

1. **Granger & Newbold (1974)** — "Spurious regressions in
   econometrics", *Journal of Econometrics* — the original spurious
   regression paper
2. **Engle & Granger (1987)** — "Co-integration and error
   correction: representation, estimation, and testing",
   *Econometrica* — the foundation
3. **Johansen (1988)** — "Statistical analysis of cointegration
   vectors", *Journal of Economic Dynamics and Control* — the trace
   and max-eigenvalue tests
4. **Johansen & Juselius (1990)** — "Maximum likelihood estimation
   and inference on cointegration — with applications to the
   demand for money", *Oxford Bulletin of Economics and Statistics* —
   the application paper with exogeneity tests
5. **Engle & Yoo (1987)** — "Forecasting and testing in
   cointegrated systems", *Journal of Econometrics* — the EG
   critical values
6. **Gregory & Hansen (1996)** — "Residual-based tests for
   cointegration in models with regime shifts", *Journal of
   Econometrics* — the structural-break test
7. **Lütkepohl (2005)** — *New Introduction to Multiple Time Series
   Analysis* — chapters 6, 7, 8 — the modern textbook

---

## §11 — Refinement log

- **v0.2.0 (2026-06-15):** Promoted from scaffold to full deep-dive.
  Added 4 demos: Johansen, VECM, Granger-in-VECM, Gregory-Hansen.
  Expanded SKILL.md to ~500 lines, added INDEX.md, REFINEMENT.md,
  references/statsmodels-vecm-api-quirks.md.
- **v0.1.0 (2026-06-15):** SCAFFOLD. 1 working demo. Sections 1-7
  were outlines.

---

*Last updated: 2026-06-15 · v0.2.0*
