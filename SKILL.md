# Cointegration and VECM — A PhD-Level Deep Dive (Scaffold)

> **Scaffold / outline version (v0.1.0).** This is the planned structure
> of a full PhD-depth treatment of cointegration and Vector Error
> Correction Models (VECM). Sections marked `⟳ TBD` are
> to-be-developed in v0.2.0+. The single demo that ships with v0.1.0
> is a working Engle-Granger 2-step cointegration test on synthetic
> data.
>
> Standalone deep-dive of the cointegration half of Stage 3 of the
> [`econometrics-deep-research`](https://github.com/Pouyasharp/econometrics-deep-research)
> parent curriculum. (The ARIMA / SARIMAX half is in the sibling
> repo [`arimax-sarimax-tutorial`](https://github.com/Pouyasharp/arimax-sarimax-tutorial).)

---

## §0 — Meta, learning outcomes, and tool stack

### 0.1 What you should be able to do at the end

By the end, you can:

1. **Diagnose spurious regression** in two I(1) series and explain
   why OLS is inconsistent for the slope.
2. **Test for cointegration** using Engle-Granger 2-step and the
   Johansen trace test.
3. **Estimate a VECM** with the right number of cointegrating
   relations, lags, and deterministic terms.
4. **Interpret the cointegrating vector** as a long-run equilibrium
   and the loadings as speeds of adjustment.
5. **Identify Granger causality** within a VECM and the
   corresponding "strong exogeneity" tests.
6. **Test for structural breaks** in the cointegrating relationship
   (Gregory-Hansen, Hatemi-J).

### 0.2 Tool stack

| Layer | Python | R | Stata |
|---|---|---|---|
| Unit-root tests | `statsmodels.tsa.stattools.adfuller`, `kpss` | `urca::ur.df`, `tseries::adf.test` | `dfuller`, `perron` |
| Engle-Granger | `statsmodels` (manual) | `urca::ca.po` | `egranger` |
| Johansen | `statsmodels.tsa.vector_ar.vecm.coint_johansen` | `urca::ca.jo` | `vecrank` |
| VECM fit | `statsmodels.tsa.vector_ar.vecm.VECM` | `tsDyn::VECM`, `vars::VECM` | `vec` |
| Structural break | (manual, `arch.unitroot` has some) | `urca::ur.za` | `zandrews` |
| IRF / FEVD | `statsmodels.tsa.vector_ar.irf` | `vars::irf` | `irf create` |

---

## §1 — Spurious regression ⟳ TBD

The classic Granger-Newbold (1974) and Phillips (1986) result: OLS
of two I(1) series on each other gives nonsense — high R², huge
t-stats, but the slope is garbage. Demonstration in `demo_01`.

**To be developed:** the formal limit theorem, simulation evidence,
and the practical diagnostic (ADF on residuals of the OLS).

---

## §2 — Cointegration ⟳ TBD

**Engle-Granger 2-step:**

1. Test each series for unit root (ADF).
2. If both I(1), OLS them on each other and test the residuals for
   stationarity (ADF on the residual).
3. If the residual is stationary → cointegrated. The OLS cointegrating
   vector is super-consistent (Engle-Granger 1987).

**Johansen trace test:**

- Form a VAR(p) in levels.
- Compute the trace statistic `λ_trace = -T Σ ln(1 - λ̂_i)` for the
  largest eigenvalues.
- The number of cointegrating relations = the number of eigenvalues
  for which the test rejects the null of "at most r relations".

**To be developed:** the VECM representation, identification
restrictions, weak exogeneity, and the Strong-Weak exogeneity tests.

---

## §3 — VECM estimation ⟳ TBD

**To be developed:** the VECM(p-1) representation, the choice of
p and the lag length criterion (AIC, HQIC, BIC), the deterministic
terms (restricted constant, restricted trend, unrestricted), the
estimation algorithm (reduced-rank MLE), and the standard errors.

---

## §4 — Interpretation ⟳ TBD

**To be developed:** what the cointegrating vector means (the
long-run equilibrium), what the adjustment coefficients mean
(speed of return to equilibrium, must be negative for stability),
and how to read the short-run dynamics.

---

## §5 — Structural breaks in cointegration ⟳ TBD

Gregory-Hansen (1996) — test for cointegration in the presence of a
level shift. Hatemi-J (2008) — test for cointegration with an
unknown break in the trend.

**To be developed:** implementation and demo.

---

## §6 — Self-assessment rubric (8 items) ⟳ TBD

1. You can explain why spurious regression happens and what its
   signature is in OLS output.
2. You can run the Engle-Granger 2-step and interpret the ADF
   test on the residuals.
3. You can run the Johansen trace test and report the number of
   cointegrating relations.
4. You can fit a VECM with the right specification and interpret
   the cointegrating vector.
5. You can identify Granger causality in a VECM.
6. You can test for structural breaks in the cointegrating
   relationship.

---

## §7 — Reading list (planned)

1. Engle & Granger (1987), "Co-integration and error correction"
2. Johansen (1988), "Statistical analysis of cointegration vectors"
3. Johansen & Juselius (1990), "Maximum likelihood estimation and
   inference on cointegration"
4. Phillips & Ouliaris (1990), "Asymptotic properties of residual
   based tests for cointegration"
5. Gregory & Hansen (1996), "Residual-based tests for cointegration
   in models with regime shifts"
6. Hatemi-J (2008), "Tests for cointegration with two unknown
   regime shifts"
7. Lütkepohl (2005), *New Introduction to Multiple Time Series
   Analysis* — chapters 6, 7, 8

---

## §8 — Refinement log

- v0.1.0 (2026-06-15): SCAFFOLD. One working demo
  (`demo_01_engle_granger.py`) shipped. Sections 1-7 are outlines
  with `⟳ TBD` markers. 7-paper reading list (subject to expansion).

---

*Last updated: 2026-06-15 · v0.1.0 (scaffold)*
