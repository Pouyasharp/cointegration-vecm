"""
demo_04_granger_in_vecm.py
==========================

A runnable demonstration of Granger causality within a VECM context.

What this file does:
1. Simulates a 2-variable VECM where:
   - y₂ Granger-causes y₁ IN THE SHORT-RUN: Δy₁ depends on Δy₂_{t-1}
   - y₁ does NOT Granger-cause y₂
   - y₂ is STRONGLY EXOGENOUS (α₂ = 0)
2. Estimates the VECM and uses the built-in Granger causality tests.
3. Demonstrates the two complementary concepts:
   - **Weak exogeneity** (within the VECM): α = 0 means that variable
     does not adjust to disequilibrium
   - **Strong exogeneity**: weak exogeneity + the variable is not
     Granger-caused by the OTHER variables
4. Shows the practical implication: if y₂ is strongly exogenous, you
   can model y₁ as a single-equation ARDL without y₂ as the dependent
   variable.

Output:
  - figures/demo_04_figure.png
  - console narrative

Requires: numpy, pandas, matplotlib, statsmodels
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM
from scipy import stats
from pathlib import Path

RNG_SEED = 20260615
T = 500
FIG_PATH = str((Path(__file__).parent.parent / "figures" / "demo_04_figure.png"))
FIG_DIR = Path(FIG_PATH).parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)


def simulate_2var_vecm_with_granger(t: int) -> pd.DataFrame:
    """
    2-variable VECM with k=1 cointegrating relation.
    The DGP encodes:
      - y₂ Granger-causes y₁ (Δy₁ depends on Δy₂_{t-1})
      - y₁ does NOT Granger-cause y₂
      - y₂ is STRONGLY EXOGENOUS (α₂ = 0)
    """
    y = np.zeros((t, 2))
    y[0] = rng.normal(0, 1, 2)
    for i in range(2, t):
        ecm = y[i-1, 0] - 0.5 * y[i-1, 1]
        # y₁: adjustment to disequilibrium + lags of Δy₂ (Granger cause from y₂)
        delta_y1 = -0.4 * ecm + 0.25 * (y[i-1, 1] - y[i-2, 1]) + rng.normal(0, 0.3)
        # y₂: NO adjustment to disequilibrium (α₂ = 0) + NO lags from y₁
        delta_y2 = 0.0 * ecm + rng.normal(0, 0.3)
        y[i, 0] = y[i-1, 0] + delta_y1
        y[i, 1] = y[i-1, 1] + delta_y2
    return pd.DataFrame(y, columns=["y1", "y2"])


def fit_vecm(data: pd.DataFrame):
    """Fit VECM(k_ar_diff=2, coint_rank=1) so we can test short-run Granger."""
    return VECM(data, k_ar_diff=2, coint_rank=1, deterministic="ci").fit()


def main() -> None:
    print("=" * 72)
    print("DEMO 04: Granger causality in the VECM — short-run + exogeneity")
    print("=" * 72)

    print(f"\n[1] Simulating 2-variable VECM with KNOWN Granger structure (T={T})")
    print(f"  True β = (1, -0.5)")
    print(f"  True α = [(-0.4), (0.0)]ᵀ  ← y₂ is weakly exogenous")
    print(f"  True γ[y₁, y₂] = 0.25      ← y₂ Granger-causes y₁ in the short run")
    print(f"  True γ[y₂, y₁] = 0         ← y₁ does NOT Granger-cause y₂")
    df = simulate_2var_vecm_with_granger(T)

    print("\n[2] Fit VECM(k_ar_diff=2, coint_rank=1)")
    vecm = fit_vecm(df)
    # Cointegrating vector and loadings
    beta_norm = vecm.beta.flatten() / vecm.beta[0, 0]
    alpha = vecm.alpha
    print(f"  β̂ = (1, {beta_norm[1]:+.3f})  (true: (1, -0.5))")
    print(f"  α̂₁ = {alpha[0, 0]:+.3f}  (true: -0.4)")
    print(f"  α̂₂ = {alpha[1, 0]:+.3f}  (true:  0.0)")

    print("\n[3] WEAK EXOGENEITY TEST: H0: α = 0")
    # For each variable, test if its loading on the cointegrating
    # vector is zero
    for i, name in enumerate(["y₁", "y₂"]):
        # Test α[i, 0] = 0
        coef = alpha[i, 0]
        se = vecm.stderr_alpha[i, 0] if hasattr(vecm, "stderr_alpha") else float("nan")
        t_stat = coef / se if se and not np.isnan(se) else float("nan")
        p = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else float("nan")
        decision = "WEAKLY EXOGENOUS" if p > 0.05 else "NOT weak-exog"
        print(f"  α̂[{name}]: coef = {coef:+.4f}  SE = {se:.4f}  t = {t_stat:+.3f}  p = {p:.4g}  → {decision}")

    print("\n[4] GRANGER CAUSALITY (SHORT-RUN) — built-in test")
    print(f"  H0: y₂ does NOT Granger-cause y₁  (γ₁₁ = γ₁₂ = 0 in y₁ equation)")
    try:
        res_y2_to_y1 = vecm.test_granger_causality(caused="y1", causing=["y2"], signif=0.05)
        print(f"  y₂ → y₁:  test_stat = {res_y2_to_y1.test_statistic:+.4f}  p = {res_y2_to_y1.pvalue:.4g}")
        if res_y2_to_y1.pvalue < 0.05:
            print(f"  ✓ Reject H0 — y₂ DOES Granger-cause y₁ (as planted)")
        else:
            print(f"  ✗ Fail to reject H0")
    except Exception as e:
        print(f"  (test_granger_causality not available: {e})")
        # Fallback: report the gamma coefficient
        gamma = vecm.gamma  # (K, K, k_ar_diff)
        for k in range(vecm.k_ar_diff):
            print(f"  γ[y₁, y₂] lag {k+1}: {gamma[0, 1, k]:+.4f}  (true: 0.25)")

    print(f"\n  H0: y₁ does NOT Granger-cause y₂  (γ₂₁ = 0 in y₂ equation)")
    try:
        res_y1_to_y2 = vecm.test_granger_causality(caused="y2", causing=["y1"], signif=0.05)
        print(f"  y₁ → y₂:  test_stat = {res_y1_to_y2.test_statistic:+.4f}  p = {res_y1_to_y2.pvalue:.4g}")
        if res_y1_to_y2.pvalue > 0.05:
            print(f"  ✓ Fail to reject H0 — y₁ does NOT Granger-cause y₂ (as planted)")
        else:
            print(f"  ✗ Reject H0 — TYPE I ERROR")
    except Exception as e:
        gamma = vecm.gamma
        for k in range(vecm.k_ar_diff):
            print(f"  γ[y₂, y₁] lag {k+1}: {gamma[1, 0, k]:+.4f}  (true: 0)")

    print("\n[5] STRONG EXOGENEITY: y₂ is both weakly exogenous AND not Granger-caused")
    print(f"  → y₂ is STRONGLY EXOGENOUS for the cointegrating parameters")
    print(f"  → y₂ can be modeled as a single-equation AR process")
    print(f"  → y₁ can be modeled as a conditional (single-equation) error-correction model")
    print(f"  → You do NOT need a full multivariate VECM — the SINGLE-EQUATION")
    print(f"     conditional model is consistent (Engle-Hendry-Richards 1983)")

    print("\n[6] PRACTICAL IMPLICATIONS")
    print(f"  • If a variable is strongly exogenous, treat it as exogenous in")
    print(f"    the model — this lets you use a single-equation model and")
    print(f"    avoid the curse of dimensionality in small samples")
    print(f"  • If a variable is NOT weakly exogenous, it must be in the")
    print(f"    conditional model — you cannot drop its equation")
    print(f"  • Granger causality is NOT the same as 'cause' in a causal-")
    print(f"    inference sense — see causal-inference §1.5 (LATE)")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel A: data
    ax = axes[0, 0]
    ax.plot(df.index, df["y1"], color="#1f77b4", linewidth=1, label="y₁", alpha=0.85)
    ax.plot(df.index, df["y2"], color="#d62728", linewidth=1, label="y₂ (strongly exogenous)", alpha=0.85)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title("Cointegrated series — y₂ weakly exogenous", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: ecm
    ax = axes[0, 1]
    ecm = df["y1"] - beta_norm[1] * df["y2"]
    ax.plot(df.index, ecm, color="#2ca02c", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("t")
    ax.ylabel if False else ax.set_ylabel(f"ecm = y₁ − {beta_norm[1]:.3f} y₂")
    ax.set_title("Cointegrating error (stationary)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel C: γ coefficients (the short-run matrix)
    ax = axes[1, 0]
    try:
        gamma = vecm.gamma
        # Show γ[y₁, y₂] across lags
        y2_to_y1 = [gamma[0, 1, k] for k in range(vecm.k_ar_diff)]
        y1_to_y2 = [gamma[1, 0, k] for k in range(vecm.k_ar_diff)]
    except AttributeError:
        y2_to_y1 = []
        y1_to_y2 = []
    lags = np.arange(1, len(y2_to_y1) + 1)
    ax.bar(lags - 0.2, y2_to_y1, width=0.4, color="#1f77b4", label="γ[y₁, y₂]  (y₂→y₁)", edgecolor="black", linewidth=0.4)
    ax.bar(lags + 0.2, y1_to_y2, width=0.4, color="#d62728", label="γ[y₂, y₁]  (y₁→y₂)", edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.25, color="#1f77b4", linestyle=":", alpha=0.6, linewidth=1, label="true γ[y₁, y₂] = 0.25")
    ax.set_xlabel("lag")
    ax.set_ylabel("γ coefficient")
    ax.set_title("Short-run γ — y₂→y₁ is non-zero (Granger cause)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: α (loadings)
    ax = axes[1, 1]
    variables = ["y₁", "y₂"]
    alpha_vals = [alpha[0, 0], alpha[1, 0]]
    colors = ["#2ca02c" if abs(a) > 0.05 else "#ff7f0e" for a in alpha_vals]
    y_pos = np.arange(len(variables))
    ax.barh(y_pos, alpha_vals, color=colors, edgecolor="black", linewidth=0.6)
    for i, a in enumerate(alpha_vals):
        ax.text(a + (0.01 if a > 0 else -0.01), i, f"{a:+.3f}", va="center", ha="left" if a > 0 else "right", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel("α (loading)")
    ax.set_title("Speed of adjustment: only y₁ adjusts (weakly exog. y₂)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[7] Figure saved: {FIG_PATH}")

    print("\n" + "=" * 72)
    print("KEY LESSONS")
    print("=" * 72)
    print("  1. Weak exogeneity = α = 0; test it first (it's a t-test)")
    print("  2. Granger causality in the VECM = short-run γ = 0 (F-test)")
    print("  3. Strong exogeneity = weak exog + not Granger-caused by others")
    print("  4. A strongly-exogenous variable can be treated as exogenous —")
    print("     you can drop its equation and use a single-equation ECM")
    print("  5. Granger causality ≠ 'cause' — it's about PREDICTIVE precedence")
    print("=" * 72)


if __name__ == "__main__":
    main()
