"""
demo_02_johansen.py
===================

Johansen trace test for the cointegrating rank on a 3-variable I(1)
system with synthetic data.

What this file does:
1. Simulates a 3-variable VAR(1) in levels with k=1 cointegrating
   relation, where the cointegrating vector is (1, -0.7, -0.4)
   (i.e., one long-run equilibrium between the three series).
2. Computes the optimal lag length via AIC.
3. Runs the Johansen trace test (constant-in-CEE, no trend).
4. Reports the trace statistic and 5% critical value for r=0, 1, 2.
5. Concludes with the rank decision and the estimated cointegrating
   vectors.
6. Also runs the max-eigenvalue test as a robustness check.

Output:
  - figures/demo_02_figure.png
  - console narrative

Requires: numpy, pandas, matplotlib, statsmodels
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from pathlib import Path

RNG_SEED = 20260615
T = 600
FIG_PATH = str((Path(__file__).parent.parent / "figures" / "demo_02_figure.png"))
FIG_DIR = Path(FIG_PATH).parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)


def simulate_vecm(t: int) -> pd.DataFrame:
    """
    Simulate a 3-variable VECM with k=1 cointegrating relation.
        ecm_{t-1} = y_{1,t-1} - 0.7 * y_{2,t-1} - 0.4 * y_{3,t-1}     (stationary)
        Δy_{t} = A * ecm_{t-1} + B * Δy_{t-1} + ε_t
    The α matrix determines the speed of adjustment to equilibrium.
    """
    y = np.zeros((t, 3))
    y[0] = rng.normal(0, 1, 3)
    # Adjustments: -0.3 on y1, +0.1 on y2, +0.05 on y3 (must sum to 0? no — they don't have to)
    A = np.array([[-0.3], [0.10], [0.05]])
    B = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # no lagged diffs for simplicity
    beta = np.array([1.0, -0.7, -0.4])  # cointegrating vector
    for i in range(1, t):
        ecm = y[i-1] @ beta  # the cointegrating error
        delta_y = A.flatten() * ecm + rng.multivariate_normal(np.zeros(3), np.eye(3) * 0.3, 1)
        y[i] = y[i-1] + delta_y.flatten()
    return pd.DataFrame(y, columns=["y1", "y2", "y3"])


def johansen_trace(data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """
    Johansen trace test (constant-in-CEE, no trend).
        det_order=0: no deterministic terms in VECM
        det_order=1: restricted constant (constant enters the
                     cointegrating equation)
        det_order=-1: no deterministic terms
    Returns the trace statistics and 5% critical values for r=0, 1, 2.
    """
    result = coint_johansen(data.values, det_order=det_order, k_ar_diff=k_ar_diff)
    return {
        "trace_stat": result.lr1,           # trace stats (len K)
        "max_eig_stat": result.lr2,         # max-eig stats (len K)
        "trace_cv_5pct": result.cvt[:, 1],  # 5% critical values for trace
        "max_eig_cv_5pct": result.cvm[:, 1],  # 5% CV for max-eig
        "eigenvalues": result.eig,          # sorted descending
        "eigenvectors": result.evec,         # columns = cointegrating vectors
    }


def optimal_lag(data: pd.DataFrame, max_lag: int = 6) -> int:
    """AIC-optimal VAR(p) lag length."""
    model = VAR(data)
    sel = model.select_order(maxlags=max_lag)
    return int(sel.aic)


def main() -> None:
    print("=" * 72)
    print("DEMO 02: Johansen trace test — rank of cointegration in a 3-variable system")
    print("=" * 72)

    print(f"\n[1] Simulating VECM with k=1 cointegrating relation (T={T})")
    print(f"  True cointegrating vector: (1, -0.7, -0.4)")
    print(f"  True adjustment matrix:   [(-0.3), (0.10), (0.05)]ᵀ")
    df = simulate_vecm(T)

    print("\n[2] Selecting optimal VAR lag via AIC")
    p = optimal_lag(df, max_lag=4)
    print(f"  AIC-optimal lag: p = {p}")
    print(f"  → VECM will be fit with p-1 lagged differenced terms")

    print("\n[3] Johansen trace test (det_order=0, k_ar_diff=p-1)")
    res = johansen_trace(df, det_order=0, k_ar_diff=max(1, p - 1))
    n = len(res["trace_stat"])
    print(f"  Number of eigenvalues: {n} (= number of variables)")
    print(f"\n  {'r':>2s}  {'trace stat':>12s}  {'5% CV':>10s}  {'decision':>10s}")
    print("  " + "-" * 50)
    rank = 0
    for i in range(n):
        s = res["trace_stat"][i]
        cv = res["trace_cv_5pct"][i]
        # decision: at step i we test "rank ≤ i"; reject if stat > CV
        decision = "REJECT" if s > cv else "fail to reject"
        # Decision to KEEP: we count the largest i+1 for which we still reject
        print(f"  {i:>2d}  {s:>12.3f}  {cv:>10.3f}  {decision:>15s}")
        if s > cv:
            rank = i + 1
    print(f"\n  → Johansen rank decision: k = {rank} cointegrating relations")

    print("\n[4] Max-eigenvalue test (alternative)")
    print(f"  {'r':>2s}  {'max eig':>10s}  {'5% CV':>10s}  {'decision':>10s}")
    print("  " + "-" * 40)
    rank_me = 0
    for i in range(n):
        s = res["max_eig_stat"][i]
        cv = res["max_eig_cv_5pct"][i]
        decision = "REJECT" if s > cv else "fail to reject"
        print(f"  {i:>2d}  {s:>10.3f}  {cv:>10.3f}  {decision:>15s}")
        if s > cv:
            rank_me += 1
        else:
            break
    print(f"  → Max-eig rank decision: k = {rank_me}")

    print("\n[5] Estimated cointegrating vectors (β, normalized on y1)")
    for i in range(rank):
        v = res["eigenvectors"][:, i]
        v_norm = v / v[0] if v[0] != 0 else v
        print(f"  β̂_{i+1} = ({v_norm[0]:+.3f}, {v_norm[1]:+.3f}, {v_norm[2]:+.3f})")
    print(f"  True β = (1.0, -0.7, -0.4)")

    print("\n[6] INTERPRETATION")
    print(f"  The {rank} cointegrating vector(s) represent(s) the long-run")
    print(f"  equilibrium(s) of the system. The corresponding speed-of-")
    print(f"  adjustment coefficients (the loadings) tell us how fast each")
    print(f"  variable returns to the equilibrium after a shock.")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel A: y1, y2, y3 over time
    ax = axes[0, 0]
    for col, color in zip(df.columns, ["#1f77b4", "#d62728", "#2ca02c"]):
        ax.plot(df.index, df[col], color=color, linewidth=1, label=col)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title("Simulated 3-variable VECM (T = 600)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: the cointegrating residual
    ax = axes[0, 1]
    beta_est = res["eigenvectors"][:, 0] / res["eigenvectors"][0, 0]
    ecm = df.values @ beta_est
    ax.plot(df.index, ecm, color="#1f77b4", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("ecm = y₁ − 0.7 y₂ − 0.4 y₃")
    ax.set_title("Cointegrating residual (should be stationary)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel C: trace stats vs CV
    ax = axes[1, 0]
    x = np.arange(n)
    ax.bar(x - 0.15, res["trace_stat"], width=0.3, color="#1f77b4", edgecolor="black", linewidth=0.5, label="Trace stat")
    ax.bar(x + 0.15, res["trace_cv_5pct"], width=0.3, color="#d62728", edgecolor="black", linewidth=0.5, label="5% critical value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={i}" for i in range(n)])
    ax.set_ylabel("value")
    ax.set_title("Trace statistic vs 5% CV (H0: rank ≤ r)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: max-eig stats vs CV
    ax = axes[1, 1]
    ax.bar(x - 0.15, res["max_eig_stat"], width=0.3, color="#1f77b4", edgecolor="black", linewidth=0.5, label="Max-eig stat")
    ax.bar(x + 0.15, res["max_eig_cv_5pct"], width=0.3, color="#d62728", edgecolor="black", linewidth=0.5, label="5% CV")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={i}" for i in range(n)])
    ax.set_ylabel("value")
    ax.set_title("Max-eigenvalue statistic vs 5% CV (H0: rank = r)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\n[7] Figure saved: {FIG_PATH}")

    print("\n" + "=" * 72)
    print("KEY LESSONS")
    print("=" * 72)
    print("  1. Johansen tests the full RANK of cointegration, not just k=1")
    print("  2. Choose det_order based on economic theory (0 / 1 / -1)")
    print("  3. Trace and max-eigenvalue tests can give different results —")
    print("     report both, default to the more conservative")
    print("  4. The cointegrating vector needs a NORMALIZATION (e.g., β[0]=1)")
    print("  5. The speed-of-adjustment (α) is the second key output")
    print("=" * 72)


if __name__ == "__main__":
    main()
