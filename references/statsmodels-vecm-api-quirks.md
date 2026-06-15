# statsmodels VECM — API Quirks

A living bank of the gotchas that bite when you use `statsmodels`'s
`VECM`, `coint_johansen`, and related time-series tools.

---

## 1. `VECM(..., deterministic=...)` requires a STRING

```python
# WRONG — raises "TypeError: deterministic must be a string"
vecm = VECM(data, k_ar_diff=1, coint_rank=1, deterministic=0).fit()

# RIGHT — use one of the string codes
vecm = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
```

**The codes:**

| String | Meaning | When to use |
|---|---|---|
| `"nc"` | No constant | Variables with zero mean (rare) |
| `"ci"` | Restricted constant IN cointegrating equation | Default; variables have a non-zero mean but the long-run relationship has zero mean |
| `"co"` | Constant OUTSIDE cointegrating equation | Variables have a non-zero mean, the long-run relationship also has a non-zero mean (most common in practice) |
| `"lo"` | Linear trend OUTSIDE | Trending data |
| `"li"` | Linear trend INSIDE | Trending data with a non-zero long-run drift |

The `R` `urca` package uses integer codes 0-3 for similar
distinctions — **don't mix them up**.

---

## 2. `coint_johansen` returns eigenvalues in DESCENDING order

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

res = coint_johansen(data.values, det_order=0, k_ar_diff=1)
# res.lr1 = trace stats (longest first)
# res.lr2 = max-eig stats
# res.cvt = 5% (column 1), 1% (column 2) critical values for trace
# res.cvm = critical values for max-eig
# res.eig = sorted descending
# res.evec = columns are the cointegrating vectors, ORDER matches res.eig
```

**Common errors:**
- **Picking the wrong column from `evec`:** the cointegrating
  vector for the LARGEST eigenvalue is `evec[:, 0]` (the first
  column), not `evec[0, :]`.
- **Mixing up `eig` and `evec`:** the eigenvalues are the
  *squared* canonical correlations, not the cointegrating vectors.

---

## 3. VECM `gamma` is 2D for `k_ar_diff=1`, 3D for `k_ar_diff>1`

```python
# With k_ar_diff=1, gamma is (K, K) — the only γ matrix
# With k_ar_diff=2, gamma is (K, K, 2) — γ_1 and γ_2
# With k_ar_diff=k, gamma is (K, K, k)
```

**Robust code:**

```python
def get_gamma_at_lag(vecm_res, eq: int, var: int, lag: int) -> float:
    """Robust: works for both 2D and 3D gamma shapes."""
    g = vecm_res.gamma
    if g.ndim == 3:
        return float(g[eq, var, lag])
    elif lag == 0:
        return float(g[eq, var])
    else:
        raise ValueError(f"Cannot access lag {lag} — k_ar_diff=1 has only lag 0")
```

---

## 4. `vecm.test_granger_causality(caused, causing)` test

The built-in test is per equation, not per pair. For the y₂ → y₁
direction (does y₂ Granger-cause y₁?):

```python
result = vecm.test_granger_causality(caused="y1", causing=["y2"], signif=0.05)
print(f"F = {result.test_statistic:.3f}, p = {result.pvalue:.4g}")
```

**Common errors:**
- **Reversing caused and causing:** the `caused` argument is the
  variable whose equation you're testing (the dependent variable).
- **Passing a list of causing variables** is supported; if you pass
  more than one, the test is "do they jointly Granger-cause?".
- **The `signif` argument doesn't affect the test result** — it
  just changes what's printed.

---

## 5. Weak exogeneity test is a `t`-test on `alpha[i, :]`

```python
# Is variable i weakly exogenous?
from scipy import stats
coef = vecm_res.alpha[i, 0]                  # assume r=1
se = vecm_res.stderr_alpha[i, 0]              # SE from the fit
t_stat = coef / se
p = 2 * (1 - stats.norm.cdf(abs(t_stat)))
```

The `stderr_alpha` attribute is what you need. If your `vecm_res`
doesn't have it, refit the model — older statsmodels versions
may not expose the per-loading SE.

---

## 6. ADF on the residual: USE THE ENGLE-GRANGER CRITICAL VALUES

When running the EG 2-step, the ADF test on the residual is testing
a NULL OF NO COINTEGRATION. The critical values are different from
the standard ADF critical values.

| T | 1% | 5% | 10% |
|---|---|---|---|
| Standard ADF (constant) | -3.43 | -2.86 | -2.57 |
| **Engle-Granger (2-var)** | **-3.90** | **-3.37** | **-3.04** |
| **Engle-Granger (K-var)** | **-4.32** | **-3.78** | **-3.47** |

`statsmodels`'s `adfuller` reports the STANDARD critical values,
not the EG ones. For a valid EG test, manually compare against
the EG critical values, or use the `R` `urca::ur.pp` /
`urca::ca.po` which reports the right ones.

---

## 7. `det_order` for `coint_johansen` is an INTEGER

```python
# Different from VECM!
res = coint_johansen(data.values, det_order=0, k_ar_diff=1)
# det_order=0: no constant
# det_order=1: restricted constant
# det_order=2: constant outside (Johansen's default)
# det_order=-1: linear trend outside
```

**Don't use strings here** — `coint_johansen` accepts only integers,
while `VECM` accepts only strings. The mapping is the same in
intent but the string/integer split is a footgun.

---

## 8. The lag length in `VECM` and `coint_johansen`

`coint_johansen(... , k_ar_diff=p)` runs the test in the context of
a VECM(p-1). So:
- `k_ar_diff=1` → VECM with no lagged diffs (only the ecm term)
- `k_ar_diff=2` → VECM with one lagged diff term
- `k_ar_diff=p` → VECM with p-1 lagged diff terms

The VAR in levels is VAR(p) with p lagged levels, which is VECM(p-1)
in differenced form. The standard AIC-based lag selection in `statsmodels`
is on the VAR in levels, so `var.select_order(maxlags=N).aic` gives you
p, and you pass `k_ar_diff=p-1` to the VECM.

---

*Last updated: 2026-06-15*
