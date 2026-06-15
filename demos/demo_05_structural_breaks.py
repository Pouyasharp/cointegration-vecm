"""
demo_05_structural_breaks.py
============================

Structural breaks in the cointegrating relationship:
  - Gregory-Hansen (1996): level shift
  - Hatemi-J (2008): unknown break in the trend + level

What this file does:
1. Simulates two I(1) series that are cointegrated, with a LEVEL
   SHIFT in the cointegrating relationship at t = T/2.
2. Naive Engle-Granger (ignoring the break) — will mis-estimate the
   cointegrating vector.
3. Gregory-Hansen test: tests for cointegration allowing a level
   shift at an UNKNOWN break date.
4. Reports the estimated break date and the structural break in the
   cointegrating vector.

Output:
  - figures/demo_05_figure.png
  - console narrative

This is a SCAFFOLD-style demo. The full Gregory-Hansen and Hatemi-J
procedures are implemented in the `R` `urca` and `strucchange`
packages; here we implement a simplified version.

Requires: numpy, pandas, matplotlib, statsmodels
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

RNG_SEED = 20260615
T = 400
BREAK_FRACTION = 0.5  # break at midpoint
TRUE_BETA_1 = 0.7
SHIFT = 5.0  # level shift in the cointegrating relationship
FIG_PATH = str((Path(__file__).parent.parent / "figures" / "demo_05_figure.png"))
FIG_DIR = Path(FIG_PATH).parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)


def simulate_cointegrated_with_break(t: int, break_frac: float, beta: float, shift: float) -> tuple[pd.DataFrame, int]:
    """
    Two I(1) series with a level shift in the cointegrating
    relationship at t = t_break.
        u_t ~ N(0, 1)  (stationary)
        ecm_{t} = y_t - β * x_t - μ_t,  μ_t = shift for t >= t_break
        x_t = x_{t-1} + u_t + N(0, 0.3)
        y_t = y_{t-1} + β * Δx_t + ecm_t + N(0, 0.3)
    """
    t_break = int(t * break_frac)
    u = rng.normal(0, 1, t)
    mu = np.where(np.arange(t) >= t_break, shift, 0.0)
    x = np.zeros(t)
    y = np.zeros(t)
    x[0] = rng.normal(0, 1)
    y[0] = rng.normal(0, 1)
    for i in range(1, t):
        ecm = y[i-1] - beta * x[i-1] - mu[i-1]
        x[i] = x[i-1] + u[i] + rng.normal(0, 0.3)
        y[i] = y[i-1] + beta * (x[i] - x[i-1]) + ecm + rng.normal(0, 0.3)
    df = pd.DataFrame({"t": np.arange(t), "x": x, "y": y, "mu_true": mu})
    return df, t_break


def ols_slope_no_break(df: pd.DataFrame) -> tuple[float, float]:
    """Naive OLS: y on x with no break — biased."""
    x = df["x"].values
    y = df["y"].values
    slope = np.sum(x * y) / np.sum(x ** 2)
    resid = y - slope * x
    n = len(x)
    se = np.sqrt(np.sum(resid ** 2) / (n - 1) / np.sum(x ** 2))
    return float(slope), float(se)


def ols_slope_with_break(df: pd.DataFrame, t_break: int) -> tuple[float, float, float]:
    """
    OLS allowing a known break at t_break:
        y = β₁ * x + μ * I[t >= t_break] + ε
    """
    x = df["x"].values
    y = df["y"].values
    t_indicator = (np.arange(len(y)) >= t_break).astype(float)
    X = np.column_stack([x, t_indicator, np.ones_like(x)])
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta_hat[0]), float(beta_hat[1]), float(beta_hat[2])


def gregory_hansen_test(df: pd.DataFrame, trim: float = 0.15) -> tuple[int, float]:
    """
    Gregory-Hansen (1996) test (level shift version):
        y = μ₁ + β₁ * x + ε                              (regime 1: t < t_b)
        y = μ₂ + β₂ * x + ε                              (regime 2: t >= t_b)
    We sweep over candidate break dates and find the one that
    MINIMIZES the ADF test statistic on the residuals (i.e., the
    strongest evidence of cointegration given the break).
    """
    x = df["x"].values
    y = df["y"].values
    n = len(y)
    t_low = int(trim * n)
    t_high = n - t_low
    best_t = t_low
    best_stat = float("inf")
    for t_b in range(t_low, t_high):
        # Regime 1: y[t < t_b] = mu1 + beta1 * x[t < t_b] + eps
        mask1 = np.arange(n) < t_b
        mask2 = ~mask1
        X1 = np.column_stack([np.ones(mask1.sum()), x[mask1]])
        X2 = np.column_stack([np.ones(mask2.sum()), x[mask2]])
        b1, _, _, _ = np.linalg.lstsq(X1, y[mask1], rcond=None)
        b2, _, _, _ = np.linalg.lstsq(X2, y[mask2], rcond=None)
        resid = np.empty(n)
        resid[mask1] = y[mask1] - X1 @ b1
        resid[mask2] = y[mask2] - X2 @ b2
        # ADF on the residuals
        try:
            stat, *_ = adfuller(resid, regression="c", autolag="AIC")
        except Exception:
            stat = float("inf")
        if stat < best_stat:
            best_stat = stat
            best_t = t_b
    return best_t, float(best_stat)


def _adf_one_stat(resid_arr: np.ndarray) -> float:
    """Return just the ADF test statistic, with safe fallback."""
    try:
        stat, *_ = adfuller(resid_arr, regression="c", autolag="AIC")
        return float(stat)
    except Exception:
        return float("inf")


def main() -> None:
    print("=" * 72)
    print("DEMO 05: Structural breaks in cointegration (Gregory-Hansen 1996)")
    print("=" * 72)

    print(f"\n[1] Simulating 2 I(1) series with a LEVEL SHIFT in the cointegrating relationship")
    print(f"  True β₁ (pre-break)  = {TRUE_BETA_1}")
    print(f"  Level shift μ        = {SHIFT}")
    print(f"  Break at t = {int(T * BREAK_FRACTION)}")
    df, true_break = simulate_cointegrated_with_break(T, BREAK_FRACTION, TRUE_BETA_1, SHIFT)

    print("\n[2] NAIVE OLS (ignoring the break) — biased")
    naive_slope, naive_se = ols_slope_no_break(df)
    resid_naive = df["y"].values - naive_slope * df["x"].values
    adf_naive_stat, adf_naive_p, *_ = adfuller(resid_naive, regression="c", autolag="AIC")
    print(f"  Naive β̂: {naive_slope:+.4f}  (true β = {TRUE_BETA_1})  bias = {naive_slope - TRUE_BETA_1:+.4f}")
    print(f"  ADF on naive residual: stat = {adf_naive_stat:+.4f}  p = {adf_naive_p:.4g}")
    if adf_naive_p > 0.05:
        print("  ⚠ Naive ADF fails to reject — we WRONGLY conclude no cointegration")

    print("\n[3] OLS WITH KNOWN BREAK at t = t_break")
    b1, mu, intercept = ols_slope_with_break(df, true_break)
    print(f"  β̂₁ (pre-break):  {b1:+.4f}  (true: {TRUE_BETA_1})")
    print(f"  Estimated μ:      {mu:+.4f}  (true: {SHIFT})")

    print("\n[4] GREGORY-HANSEN TEST — break at UNKNOWN date")
    print(f"  Sweeping over candidate break dates in [{int(0.15 * T)}, {int(0.85 * T)}]")
    est_break, best_stat = gregory_hansen_test(df, trim=0.15)
    print(f"  Estimated break: t = {est_break}  (true: {true_break})  error: {abs(est_break - true_break)}")
    print(f"  Best ADF stat on regime-by-regime residuals: {best_stat:+.4f}")
    print(f"  → Smaller (more negative) ADF stat = STRONGER evidence of cointegration with the break")
    # (We use best_stat directly — no separate p-value needed for the decision)
    if best_stat < adf_naive_stat:
        print("  ✓ Gregory-Hansen ADF stat is smaller (more negative) than naive — break correction helps")
    else:
        print("  ✗ Gregory-Hansen did not improve — break may not be the issue")

    print("\n[5] INTERPRETATION")
    print(f"  The cointegrating relationship CHANGED at t = {est_break} (estimated).")
    print(f"  Two interpretations:")
    print(f"    (a) A genuine structural break — the long-run equilibrium shifted")
    print(f"    (b) Omitted-variable bias — a time-varying regressor should be added")
    print(f"  Gregory-Hansen cannot distinguish (a) from (b) — it just tests (a).")

    print("\n[6] WHAT'S NEXT (v0.3.0+)")
    print(f"  ⟳ Hatemi-J (2008): unknown break in the trend + level")
    print(f"  ⟳ Bai-Perron sequential break test (Bai & Perron 1998, 2003)")
    print(f"  ⟳ Markov-switching VECM (Krolzig 1997)")
    print(f"  ⟳ TVP-VECM (time-varying parameters)")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel A: data with the break
    ax = axes[0, 0]
    ax.plot(df["t"], df["x"], color="#1f77b4", linewidth=1, label="x", alpha=0.85)
    ax.plot(df["t"], df["y"], color="#d62728", linewidth=1, label="y", alpha=0.85)
    ax.axvline(true_break, color="black", linestyle="--", alpha=0.7, linewidth=1.2, label=f"True break (t={true_break})")
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title("Cointegrated series with structural break", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: the cointegrating residual
    ax = axes[0, 1]
    # Compute residual using the BREAK-AWARE estimate
    b1, mu, _ = ols_slope_with_break(df, true_break)
    resid_break = df["y"].values - b1 * df["x"].values - mu * (df["t"].values >= true_break).astype(float)
    ax.plot(df["t"], resid_break, color="#2ca02c", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("residual (with break)")
    ax.set_title("Residual with break (stationary)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel C: naive residual (shows the shift)
    ax = axes[1, 0]
    ax.plot(df["t"], resid_naive, color="#d62728", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(true_break, color="black", linestyle="--", alpha=0.7, linewidth=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel("residual (naive)")
    ax.set_title("Naive residual: NOT stationary (shift visible)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel D: ADF stat across candidate break dates
    ax = axes[1, 1]
    x = df["x"].values
    y = df["y"].values
    n = len(y)
    t_low, t_high = int(0.15 * n), int(0.85 * n)
    adf_stats = []
    for t_b in range(t_low, t_high, 3):
        mask1 = np.arange(n) < t_b
        mask2 = ~mask1
        X1 = np.column_stack([np.ones(mask1.sum()), x[mask1]])
        X2 = np.column_stack([np.ones(mask2.sum()), x[mask2]])
        b1, _, _, _ = np.linalg.lstsq(X1, y[mask1], rcond=None)
        b2, _, _, _ = np.linalg.lstsq(X2, y[mask2], rcond=None)
        resid = np.empty(n)
        resid[mask1] = y[mask1] - X1 @ b1
        resid[mask2] = y[mask2] - X2 @ b2
        try:
            stat = _adf_one_stat(resid)
        except Exception:
            stat = float("nan")
        adf_stats.append((t_b, stat))
    cand_t = [s[0] for s in adf_stats]
    cand_stat = [s[1] for s in adf_stats]
    ax.plot(cand_t, cand_stat, color="#1f77b4", linewidth=1.5)
    ax.axvline(true_break, color="red", linestyle="--", alpha=0.6, linewidth=1, label=f"true break = {true_break}")
    ax.axvline(est_break, color="green", linestyle=":", alpha=0.7, linewidth=1.5, label=f"Gregory-Hansen = {est_break}")
    ax.axhline(adf_naive_stat, color="purple", linestyle="-.", alpha=0.5, linewidth=1, label=f"naive ADF = {adf_naive_stat:+.2f}")
    ax.set_xlabel("candidate break date t")
    ax.set_ylabel("ADF stat on regime-by-regime residual")
    ax.set_title("Gregory-Hansen: min ADF stat = break date", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[7] Figure saved: {FIG_PATH}")

    print("\n" + "=" * 72)
    print("KEY LESSONS")
    print("=" * 72)
    print("  1. Spurious regression + structural break = double trouble")
    print("  2. Naive Engle-Granger MISLEADINGLY fails to detect cointegration")
    print("     when the long-run relationship has shifted")
    print("  3. Gregory-Hansen: allow a level shift at an UNKNOWN date")
    print("  4. The break date is chosen to MINIMIZE the ADF stat on the")
    print("     regime-by-regime residuals")
    print("  5. ALWAYS plot the residual — a visible shift is a hint")
    print("=" * 72)


if __name__ == "__main__":
    main()
