"""
demo_01_engle_granger.py — SCAFFOLD

A runnable demonstration of the Engle-Granger 2-step cointegration test
on synthetic data.

What this file does:
1. Simulates two cointegrated I(1) series (a random walk with a
   cointegrating error)
2. Demonstrates the SPURIOUS REGRESSION problem (Granger-Newbold 1974)
   by simulating two unrelated random walks and showing that OLS gives
   huge t-stats on nonsense
3. Runs the Engle-Granger 2-step:
   a. ADF on each series
   b. OLS the two series on each other
   c. ADF on the residuals — reject unit root → cointegrated
4. Reports the cointegrating vector (the OLS slope)

Output:
  - figures/demo_01_figure.png
  - console narrative

This is a SCAFFOLD demo. v0.2.0 will add:
  - Johansen trace test demo
  - VECM estimation demo
  - Structural break in cointegration demo

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
T = 500
FIG_PATH = str((Path(__file__).parent.parent / "figures" / "demo_01_figure.png"))
FIG_DIR = Path(FIG_PATH).parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)


def simulate_cointegrated(t: int) -> pd.DataFrame:
    """
    Two I(1) series that are cointegrated:
        u_t ~ N(0, 1)  (the cointegrating error, stationary)
        x_t = u_t + x_{t-1} + N(0, 1)  (random walk driven by u)
        y_t = 0.7 * x_t + u_t + y_{t-1} + N(0, 1)  (random walk with the same u)
    The long-run relationship is y - 0.7*x = u (stationary).
    """
    u = rng.normal(0, 1, t)
    x = np.zeros(t)
    y = np.zeros(t)
    for i in range(1, t):
        x[i] = x[i-1] + u[i] + rng.normal(0, 0.3)
        y[i] = y[i-1] + 0.7 * x[i] + u[i] + rng.normal(0, 0.3)
    return pd.DataFrame({"t": np.arange(t), "x": x, "y": y, "u_true": u})


def simulate_spurious(t: int) -> pd.DataFrame:
    """Two INDEPENDENT random walks. Should be I(1) but NOT cointegrated."""
    x = np.cumsum(rng.normal(0, 1, t))
    y = np.cumsum(rng.normal(0, 1, t))
    return pd.DataFrame({"t": np.arange(t), "x": x, "y": y})


def adf_test(series: pd.Series, name: str = "series") -> tuple[float, float]:
    """Return (test_stat, p_value) for ADF with constant."""
    result = adfuller(series, regression="c", autolag="AIC")
    return float(result[0]), float(result[1])


def ols_slope(df: pd.DataFrame) -> tuple[float, float]:
    """OLS of y on x (no intercept — the OLS slope IS the cointegrating vector)."""
    x = df["x"].values
    y = df["y"].values
    slope = np.sum(x * y) / np.sum(x ** 2)
    # Naive SE
    resid = y - slope * x
    n = len(x)
    se = np.sqrt(np.sum(resid ** 2) / (n - 1) / np.sum(x ** 2))
    return float(slope), float(se)


def main() -> None:
    print("=" * 72)
    print("DEMO 01 (SCAFFOLD): Engle-Granger 2-step cointegration test")
    print("=" * 72)

    # ---- Part 1: Spurious regression ----
    print("\n[1] Simulating TWO INDEPENDENT random walks (should be spurious)")
    df_spur = simulate_spurious(T)
    print(f"  Generated {T} observations")
    adf_x, p_x = adf_test(df_spur["x"], "x")
    adf_y, p_y = adf_test(df_spur["y"], "y")
    print(f"  ADF on x: stat = {adf_x:+.4f}  p = {p_x:.4g}  (fail to reject → I(1))")
    print(f"  ADF on y: stat = {adf_y:+.4f}  p = {p_y:.4g}  (fail to reject → I(1))")
    slope_spur, se_spur = ols_slope(df_spur)
    t_stat_spur = slope_spur / se_spur
    print(f"\n  OLS slope (spurious): {slope_spur:+.4f}  SE = {se_spur:.4f}  t = {t_stat_spur:+.4f}")
    if abs(t_stat_spur) > 2:
        print("  ⚠ HUGE t-stat on a SLOPE THAT IS MEANINGLESS — the spurious regression problem")
    else:
        print("  (this run happens to have a small t-stat — spurious regression is *probabilistic*)")

    # ---- Part 2: Cointegrated series ----
    print("\n[2] Simulating TWO COINTEGRATED I(1) series")
    df_coint = simulate_cointegrated(T)
    print(f"  Generated {T} observations")
    adf_x, p_x = adf_test(df_coint["x"], "x")
    adf_y, p_y = adf_test(df_coint["y"], "y")
    print(f"  ADF on x: stat = {adf_x:+.4f}  p = {p_x:.4g}  (fail to reject → I(1))")
    print(f"  ADF on y: stat = {adf_y:+.4f}  p = {p_y:.4g}  (fail to reject → I(1))")

    print("\n[3] Engle-Granger 2-step: OLS the levels, test the residual")
    slope_coint, se_coint = ols_slope(df_coint)
    resid = df_coint["y"] - slope_coint * df_coint["x"]
    adf_r, p_r = adf_test(resid, "residual")
    print(f"  OLS slope:               {slope_coint:+.4f}  (true: 0.7)")
    print(f"  ADF on residual:         stat = {adf_r:+.4f}  p = {p_r:.4g}")
    if p_r < 0.05:
        print("  ✓ Reject unit root in residual → COINTEGRATED")
        print(f"  Cointegrating vector: y - {slope_coint:.3f}*x = stationary")
    else:
        print("  ✗ Fail to reject unit root → NOT cointegrated (or power too low)")

    print("\n[4] COMPARISON")
    print(f"  Spurious:    slope = {slope_spur:+.4f}  (no economic meaning)")
    print(f"  Cointegrated: slope = {slope_coint:+.4f}  (= long-run equilibrium)")

    print("\n[5] WHAT'S NEXT (v0.2.0+)")
    print("  ⟳ Johansen trace test (multi-equation cointegration)")
    print("  ⟳ VECM estimation (short-run dynamics + long-run equilibrium)")
    print("  ⟳ Strong / weak exogeneity tests")
    print("  ⟳ Structural breaks in cointegration (Gregory-Hansen, Hatemi-J)")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Panel A: spurious — x and y
    ax = axes[0, 0]
    ax.plot(df_spur["t"], df_spur["x"], color="#1f77b4", linewidth=1, label="x", alpha=0.85)
    ax.plot(df_spur["t"], df_spur["y"], color="#d62728", linewidth=1, label="y", alpha=0.85)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(f"Spurious: two INDEPENDENT random walks\nOLS slope = {slope_spur:+.3f}, t = {t_stat_spur:+.2f}", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: spurious — y vs x
    ax = axes[0, 1]
    ax.scatter(df_spur["x"], df_spur["y"], s=4, alpha=0.4, color="#7f7f7f")
    x_grid = np.linspace(df_spur["x"].min(), df_spur["x"].max(), 100)
    ax.plot(x_grid, slope_spur * x_grid, color="#d62728", linewidth=2, label=f"OLS y = {slope_spur:+.3f} * x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Spurious: OLS on independent RWs", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel C: cointegrated — x and y
    ax = axes[1, 0]
    ax.plot(df_coint["t"], df_coint["x"], color="#1f77b4", linewidth=1, label="x", alpha=0.85)
    ax.plot(df_coint["t"], df_coint["y"], color="#d62728", linewidth=1, label="y", alpha=0.85)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(f"Cointegrated: same u drives both\nOLS slope = {slope_coint:+.3f} (true 0.7)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel D: cointegrated — residual should be stationary
    ax = axes[1, 1]
    ax.plot(df_coint["t"], resid, color="#2ca02c", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("residual")
    ax.set_title(f"Residual y − {slope_coint:.3f}*x  (ADF p = {p_r:.4g})", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[6] Figure saved: {FIG_PATH}")

    print("\n" + "=" * 72)
    print("KEY LESSONS")
    print("=" * 72)
    print("  1. Spurious regression: OLS on two I(1) series gives huge t-stats on garbage")
    print("  2. Cointegration = both series I(1) but a LINEAR COMBINATION is I(0)")
    print("  3. Engle-Granger 2-step: ADF on the OLS residual")
    print("  4. Cointegrating vector = the long-run equilibrium (here: y - 0.7*x = stationary)")
    print("  5. The OLS cointegrating vector is super-consistent (Engle-Granger 1987)")
    print("  ⟳ v0.2.0 will cover Johansen, VECM, and structural breaks")
    print("=" * 72)


if __name__ == "__main__":
    main()
