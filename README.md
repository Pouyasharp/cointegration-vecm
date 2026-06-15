# Cointegration and VECM — A PhD-Level Deep Dive

> **A focused, runnable treatment of cointegration and Vector Error
> Correction Models (VECM):** Engle-Granger 2-step, Johansen trace
> test, VECM estimation and interpretation, Granger causality in the
> VECM, and structural breaks in the cointegrating relationship.
> Standalone deep-dive of the cointegration half of Stage 3 of the
> [`econometrics-deep-research`](https://github.com/Pouyasharp/econometrics-deep-research)
> parent curriculum. (The ARIMA / SARIMAX half is in the sibling
> repo [`arimax-sarimax-tutorial`](https://github.com/Pouyasharp/arimax-sarimax-tutorial).)

The full curriculum lives in **[`SKILL.md`](./SKILL.md)** (~500 lines,
PhD-depth). This README is the front door: it tells you what you'll
learn, how to study it, and what the five runnable demos produce.

---

## What this repo gives you

- **A ~500-line curriculum** in `SKILL.md` covering:
  - **§1 Spurious regression** — the Granger-Newbold (1974) /
    Phillips (1986) result and its signature in OLS output
  - **§2 Cointegration** — definition, intuition, and the
    pre-conditions (both series I(1), a linear combination I(0))
  - **§3 Engle-Granger 2-step** — OLS the levels, ADF on the residual
  - **§4 Johansen trace test** — multi-equation cointegration, the
    rank decision, the choice of `det_order`
  - **§5 VECM estimation** — the long-run β, the loadings α, the
    short-run γ, lag selection, and deterministic terms
  - **§6 Interpretation** — what the cointegrating vector means, the
    speed of adjustment, IRF / FEVD within a VECM
  - **§7 Granger causality in the VECM** — weak exogeneity, strong
    exogeneity, the Engle-Hendry-Richards (1983) result
  - **§8 Structural breaks** — Gregory-Hansen (1996), Hatemi-J (2008)
  - **§9 Self-assessment rubric** (8 items)
  - **§10 Reading list** (7 papers)
  - **§11 Refinement log** (versioned)
- **Five self-contained Python demos** in `demos/`, each ~250-310
  lines, that simulate the relevant DGP and exercise the
  methodology end-to-end. Each demo plants a true effect so you can
  verify the test / estimator recovers it.

---

## Quick start

```bash
git clone https://github.com/Pouyasharp/cointegration-vecm.git
cd cointegration-vecm
pip install -r requirements.txt

# Run any demo (each is self-contained, ~30-90s)
python3 demos/demo_01_engle_granger.py
python3 demos/demo_02_johansen.py
python3 demos/demo_03_vecm.py
python3 demos/demo_04_granger_in_vecm.py
python3 demos/demo_05_structural_breaks.py
```

Each demo prints a step-by-step narrative to stdout and writes a
figure to `figures/`:

| Demo | Topic | Figure |
|------|-------|--------|
| 01 | Engle-Granger 2-step: spurious vs cointegrated series | `figures/demo_01_figure.png` |
| 02 | Johansen trace test on a 3-variable I(1) system | `figures/demo_02_figure.png` |
| 03 | VECM: long-run β, short-run γ, speed-of-adjustment α | `figures/demo_03_figure.png` |
| 04 | Granger causality in the VECM: weak + strong exogeneity | `figures/demo_04_figure.png` |
| 05 | Gregory-Hansen: structural break in the cointegrating vector | `figures/demo_05_figure.png` |

---

## What the figures look like

### Demo 01 — Engle-Granger 2-step
`figures/demo_01_figure.png` shows a 2×2 layout: (A) two independent
random walks (the spurious case), (B) OLS on the two RWs with a
huge t-stat on a garbage slope, (C) two cointegrated I(1) series
(planted β = 0.7), (D) the cointegrating residual (stationary).

### Demo 02 — Johansen trace test
`figures/demo_02_figure.png` shows a 2×2 layout: (A) the 3-variable
system, (B) the cointegrating residual, (C) trace statistic vs 5%
CV for r=0, 1, 2, (D) max-eigenvalue statistic vs 5% CV. The
decision: k=1.

### Demo 03 — VECM
`figures/demo_03_figure.png` shows: (A) the cointegrated series,
(B) the cointegrating error (stationary), (C) ECM autocorrelation
to estimate the half-life, (D) the VECM coefficients vs the true
planted values.

### Demo 04 — Granger causality in the VECM
`figures/demo_04_figure.png` shows: (A) the cointegrated series with
the weakly-exogenous variable, (B) the cointegrating error,
(C) the short-run γ matrix (the y₂→y₁ coefficient is non-zero,
as planted), (D) the α loadings (only y₁ adjusts, y₂ doesn't).

### Demo 05 — Gregory-Hansen
`figures/demo_05_figure.png` shows: (A) the data with the true
break, (B) the break-aware residual (stationary), (C) the naive
residual (visibly shifted), (D) the ADF statistic as a function
of the candidate break date — the minimum is the Gregory-Hansen
estimate.

---

## How to study this

1. **Read `SKILL.md` §0** — get oriented, set up the tool stack.
2. **Read §1-§3 (Engle-Granger)** and run `demo_01`. Spurious
   regression is a probabilistic phenomenon — the same DGP can
   give different results across runs.
3. **Read §4 (Johansen)** and run `demo_02`. Trace and
   max-eigenvalue tests can disagree; report both.
4. **Read §5-§6 (VECM)** and run `demo_03`. The key output is
   β (long-run), α (adjustment), and γ (short-run).
5. **Read §7 (Granger in VECM)** and run `demo_04`. The concepts
   of weak and strong exogeneity are diagnostic — they tell you
   whether a full multivariate model is necessary.
6. **Read §8 (Structural breaks)** and run `demo_05`. Always
   plot the residual — a visible shift is a hint.
7. **Self-assess with §9** rubric.
8. **Read the papers in §10** (7 entries, organized by topic).
9. **Open `REFINEMENT.md`** to see what's still missing (Hatemi-J,
   Bai-Perron, Markov-switching VECM, TVP-VECM).

---

## How to apply to a real project

For **a pair of prices (or a price and a quantity) that should be
related in the long run:**

```python
# 1. Diagnose the order of integration
from statsmodels.tsa.stattools import adfuller
adf_p_x = adfuller(price_x)[1]    # should be > 0.05 (I(1))
adf_p_y = adfuller(price_y)[1]    # should be > 0.05 (I(1))

# 2. Engle-Granger 2-step
import statsmodels.api as sm
X = sm.add_constant(price_x)
fit = sm.OLS(price_y, X).fit()
beta_eg = fit.params[price_x]
resid = price_y - beta_eg * price_x
adf_p_resid = adfuller(resid)[1]  # should be < 0.05 (cointegrated)

# 3. If you have more than 2 series, use Johansen
from statsmodels.tsa.vector_ar.vecm import coint_johansen
res = coint_johansen(df.values, det_order=0, k_ar_diff=2)
# res.lr1 = trace stats, res.cvt = 5% CV
# rank = number of trace stats that exceed the CV

# 4. Estimate the VECM
from statsmodels.tsa.vector_ar.vecm import VECM
vecm = VECM(df, k_ar_diff=2, coint_rank=1, deterministic="ci").fit()
print(vecm.beta)        # cointegrating vector
print(vecm.alpha)       # speed of adjustment
print(vecm.gamma)       # short-run dynamics

# 5. Test Granger causality within the VECM
result = vecm.test_granger_causality(caused="y1", causing=["y2"])
print(f"y2 → y1: p = {result.pvalue}")

# 6. ALWAYS check for a structural break before reporting
from statsmodels.tsa.stattools import adfuller
# Compute the residual and look for a level shift
# If found → use Gregory-Hansen (R `urca` package)
```

For **what to do when each assumption fails**, see `SKILL.md` §3
(super-consistency of EG is reassuring but finite-sample is bad),
§4 (rank selection in small samples), §6 (normalization issues),
§7 (when exogeneity fails → multivariate VECM is required),
§8 (Gregory-Hansen for level shifts, Hatemi-J for trend breaks).

---

## Repository layout

```
cointegration-vecm/
├── README.md             # this file
├── SKILL.md              # ~500-line PhD curriculum
├── INDEX.md              # topic-to-section navigation
├── REFINEMENT.md         # known gaps, versioned checklist
├── LICENSE               # MIT
├── requirements.txt
├── .gitignore
├── references/
│   └── statsmodels-vecm-api-quirks.md
├── demos/
│   ├── demo_01_engle_granger.py          # ~270 lines
│   ├── demo_02_johansen.py               # ~250 lines
│   ├── demo_03_vecm.py                   # ~260 lines
│   ├── demo_04_granger_in_vecm.py        # ~280 lines
│   └── demo_05_structural_breaks.py      # ~310 lines
└── figures/
    ├── demo_01_figure.png
    ├── demo_02_figure.png
    ├── demo_03_figure.png
    ├── demo_04_figure.png
    └── demo_05_figure.png
```

---

## Why this exists

Cointegration is the canonical workhorse for "two series that should
move together in the long run." It is the right framework for
*prices that should obey a no-arbitrage condition*, *a price-quantity
relationship*, *demand and supply that should be linked*, and
*marketing + macro variables that share a long-run equilibrium*. The
tools are well-established but the failure modes (spurious regression,
structural breaks, weak exogeneity ignored) are easy to miss without
a PhD-level treatment.

This repo provides that treatment with runnable code. It is
intentionally a **curriculum**, not a library — the demos are
designed to be read and modified, not imported as a package.

The parent curriculum ([`econometrics-deep-research`][1]) maps the
full 5-stage top-to-bottom treatment at survey depth; this deep-dive
is the natural next stop for anyone applying cointegration / VECM to
a real dataset with two or more I(1) series.

[1]: https://github.com/Pouyasharp/econometrics-deep-research

---

## License

MIT. See `LICENSE`. Demos may be reused as starting points for your
own analyses; attribution appreciated.
