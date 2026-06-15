"""
demo_03_vecm.py
===============

VECM estimation, interpretation of the long-run and short-run
components, and the speed-of-adjustment (loading) coefficients.

What this file does:
1. Simulates a 2-variable VECM with a known cointegrating vector
   and known adjustment matrix.
2. Estimates the VECM with p=2 (k_ar_diff=1).
3. Reports:
   - The estimated cointegrating vector β̂
   - The estimated speed-of-adjustment matrix α̂
   - The short-run coefficients on Δy_{t-1}
4. Compares the VECM fit to a naive VAR in levels (which produces
   a spurious regression in the second-stage estimates).
5. Generates a 4-panel figure showing the data, the cointegrating
   error, the impulse response, and the VECM-vs-VAR coefficient
   comparison.

Output:
  - figures/demo_03_figure.png
  - console narrative

Requires: numpy, pandas, matplotlib, statsmodels
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.tsa.api import VAR
from pathlib import Path

RNG_SEED = 20260615
T = 500
FIG_PATH = str((Path(__file__).parent.parent / "figures" / "demo_03_figure.png"))
FIG_DIR = Path(FIG_PATH).parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)


def simulate_2var_vecm(t: int) -> pd.DataFrame:
    """
    2-variable VECM with 1 cointegrating relation.
        ecm_{t-1} = y_{1,t-1} - 0.5 * y_{2,t-1}            (stationary)
        Δy_{1,t} = -0.4 * ecm_{t-1} + 0.1 * Δy_{1,t-1} + ε_{1,t}
        Δy_{2,t} = +0.2 * ecm_{t-1} - 0.05 * Δy_{2,t-1} + ε_{2,t}
    """
    y = np.zeros((t, 2))
    y[0] = rng.normal(0, 1, 2)
    for i in range(1, t):
        ecm = y[i-1, 0] - 0.5 * y[i-1, 1]
        delta_y1 = -0.4 * ecm + 0.1 * (y[i-1, 0] - y[i-2, 0]) + rng.normal(0, 0.3)
        delta_y2 = 0.2 * ecm - 0.05 * (y[i-1, 1] - y[i-2, 1]) + rng.normal(0, 0.3)
        y[i, 0] = y[i-1, 0] + delta_y1
        y[i, 1] = y[i-1, 1] + delta_y2
    return pd.DataFrame(y, columns=["y1", "y2"])


def fit_vecm(data: pd.DataFrame, k_ar_diff: int = 1, det_order: str = "ci"):
    """Fit a VECM with k_ar_diff lagged differenced terms.

    det_order: "nc" (no constant), "ci" (restricted constant in CEE),
    "co" (constant outside CEE), "lo" (linear trend outside), "li" (linear in CEE)
    """
    return VECM(data, k_ar_diff=k_ar_diff, coint_rank=1, deterministic=det_order).fit()


def fit_var_levels(data: pd.DataFrame, lags: int = 2):
    """Naive VAR in levels — the spurious-regression comparison."""
    return VAR(data).fit(maxlags=lags, ic=None)


def main() -> None:
    print("=" * 72)
    print("DEMO 03: VECM — long-run β, short-run dynamics, speed of adjustment")
    print("=" * 72)

    print(f"\n[1] Simulating 2-variable VECM (T={T})")
    print(f"  True β = (1, -0.5)         (cointegrating vector)")
    print(f"  True α = [(-0.4), (0.2)]ᵀ   (speed of adjustment)")
    df = simulate_2var_vecm(T)

    print("\n[2] VECM fit (k_ar_diff=1, coint_rank=1)")
    vecm = fit_vecm(df, k_ar_diff=1, det_order="ci")
    # The cointegrating vector is the first column of `beta` (k_ar_diff=1 → 1 col)
    beta_est = vecm.beta.flatten()
    alpha_est = vecm.alpha  # K x 1
    print(f"  Estimated β:    ({beta_est[0]:+.3f}, {beta_est[1]:+.3f})  (true: (1, -0.5))")
    print(f"  Estimated α₁:   {alpha_est[0, 0]:+.4f}  (true: -0.4)")
    print(f"  Estimated α₂:   {alpha_est[1, 0]:+.4f}  (true: +0.2)")

    # Normalize β to (1, b2)
    if abs(beta_est[0]) > 1e-6:
        beta_norm = beta_est / beta_est[0]
        print(f"  Normalized β:   (1.0, {beta_norm[1]:+.3f})")

    print("\n[3] Interpretation: speed of adjustment")
    print(f"  α₁ = {alpha_est[0, 0]:+.3f}: y₁ falls by {abs(alpha_est[0, 0]):.2%} of the disequilibrium per period")
    print(f"  α₂ = {alpha_est[1, 0]:+.3f}: y₂ rises by {abs(alpha_est[1, 0]):.2%} of the disequilibrium per period")
    print(f"  → The system restores equilibrium through BOTH variables moving")
    print(f"    back toward the long-run relationship y₁ - 0.5*y₂ = ecm_t (stationary)")

    print("\n[4] Short-run dynamics: coefficient on Δy_{t-1}")
    # The 'gamma' matrix is the short-run coefficient on lagged diffs
    # statsmodels VECM stores it as vecm.gamma
    if hasattr(vecm, "gamma"):
        # gamma is (K, K) for k_ar_diff=1, (K, K, k_ar_diff) for k_ar_diff > 1
        if vecm.gamma.ndim == 3:
            g11 = vecm.gamma[0, 0, 0]
            g22 = vecm.gamma[1, 1, 0]
        else:
            g11 = vecm.gamma[0, 0]
            g22 = vecm.gamma[1, 1]
        print(f"  γ₁₁ (effect of Δy₁_{{t-1}} on Δy₁_t): {g11:+.4f}")
        print(f"  γ₂₂ (effect of Δy₂_{{t-1}} on Δy₂_t): {g22:+.4f}")

    print("\n[5] IMPULSE RESPONSE — shock to ecm dissipates over time")
    # Build a simple IRF: how a unit shock to ecm_{t-1} decays
    ecm_series = df["y1"] - beta_norm[1] * df["y2"]
    ecm_shock = pd.Series(ecm_series.values - ecm_series.mean())
    # Estimate half-life: when does autocorrelation drop below 0.5?
    acf_vals = np.array([ecm_shock.autocorr(lag=k) for k in range(0, 30)])
    if acf_vals[0] > 0.5:
        try:
            half_life = next(k for k in range(1, 30) if acf_vals[k] < 0.5 * acf_vals[0])
        except StopIteration:
            half_life = "n/a"
    else:
        half_life = "n/a (ACF already < 0.5 at lag 0)"
    print(f"  ECM half-life: {half_life} periods")

    print("\n[6] COMPARISON TO NAIVE VAR IN LEVELS")
    var_levels = fit_var_levels(df, lags=2)
    print(f"  VAR in levels coefficient matrix (first lag):")
    A_var = var_levels.coefs[0]
    print(f"  A_VAR = \n  {A_var}")
    print(f"  → The VAR in levels has unit-root-like dynamics; the VECM")
    print(f"    separates the long-run cointegration from the short-run dynamics")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel A: data
    ax = axes[0, 0]
    ax.plot(df.index, df["y1"], color="#1f77b4", linewidth=1, label="y₁", alpha=0.85)
    ax.plot(df.index, df["y2"], color="#d62728", linewidth=1, label="y₂", alpha=0.85)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title("Cointegrated series (2-var)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: cointegrating error
    ax = axes[0, 1]
    ecm_plot = df["y1"] - beta_norm[1] * df["y2"]
    ax.plot(df.index, ecm_plot, color="#2ca02c", linewidth=0.8)
    ax.axhline(ecm_plot.mean(), color="black", linestyle="--", alpha=0.5, linewidth=0.7, label=f"mean = {ecm_plot.mean():.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel(f"ecm = y₁ − {beta_norm[1]:.3f} y₂")
    ax.set_title("Cointegrating error (stationary)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel C: autocorrelation of ecm
    ax = axes[1, 0]
    ax.bar(range(0, 30), acf_vals, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.axhline(0.5 * acf_vals[0], color="red", linestyle="--", alpha=0.6, linewidth=1, label=f"50% of lag-0 = {0.5 * acf_vals[0]:.2f}")
    ax.set_xlabel("lag k")
    ax.set_ylabel(f"ACF(ecm)")
    ax.set_title(f"ECM autocorrelation: half-life ≈ {half_life}", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: coefficient comparison
    ax = axes[1, 1]
    methods = ["VECM\nβ̂ (long-run)", "VECM\nα̂₁ (adj.)", "VECM\nα̂₂ (adj.)"]
    estimates = [beta_norm[1], alpha_est[0, 0], alpha_est[1, 0]]
    colors = ["#1f77b4", "#2ca02c", "#2ca02c"]
    y = np.arange(len(methods))
    ax.barh(y, estimates, color=colors, edgecolor="black", linewidth=0.6)
    trues = [-0.5, -0.4, 0.2]
    for i, (est, t_val) in enumerate(zip(estimates, trues)):
        ax.plot([t_val, t_val], [i - 0.4, i + 0.4], color="red", linewidth=2.5, alpha=0.85)
        ax.text(est + (0.02 if est > 0 else -0.02), i, f"{est:+.3f}", va="center", ha="left" if est > 0 else "right", fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("coefficient")
    ax.set_title("VECM estimates vs true (red lines)", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[7] Figure saved: {FIG_PATH}")

    print("\n" + "=" * 72)
    print("KEY LESSONS")
    print("=" * 72)
    print("  1. VECM separates long-run (β) from short-run (α, γ) dynamics")
    print("  2. The speed-of-adjustment α must be NEGATIVE in the 'wrong' direction")
    print("     (e.g., if y₁ is too high, α₁ < 0 means y₁ falls back)")
    print("  3. The VECM representation is IDENTIFIED up to a normalization of β")
    print("  4. Compare to a naive VAR in levels — the VECM is consistent under")
    print("     cointegration, the VAR is not")
    print("  5. Always report β, α, AND the lag length")
    print("=" * 72)


if __name__ == "__main__":
    main()
