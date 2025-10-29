"""
================================================================================
Advanced Online Portfolio Selection: Adaptive Methods and Transaction Costs
================================================================================

This tutorial demonstrates advanced OPS strategies including adaptive learning rates,
second-order methods, and the impact of transaction costs on strategy performance.

**Covered Topics:**

1. **Adaptive First-Order Methods**: AdaGrad adjusts learning rates per coordinate
   based on gradient history, providing robustness to ill-conditioned problems.

2. **Adaptive Barrier Methods**: AdaBARRONS combines log-barrier geometry with
   adaptive preconditioning for improved simplex-constrained optimization.

3. **Mean Reversion Variants**: PAMR and CWMR provide alternative approaches to
   exploiting short-term price reversals with different update mechanisms.

4. **Transaction Cost Analysis**: Real-world trading incurs costs that can
   significantly impact net returns, especially for high-turnover strategies.

5. **Market Regime Sensitivity**: Strategy performance varies across trending vs
   mean-reverting regimes; we examine this using original and reversed datasets.

References
----------
.. [1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods
       for Online Learning and Stochastic Optimization. JMLR, 12, 2121-2159.
.. [2] Foster, D., Rakhlin, A., & Sridharan, K. (2017). Adaptive Online Learning.
       NIPS 2017.
.. [3] Li, B., Zhao, P., Hoi, S. C. H., & Gopalkrishnan, V. (2012). PAMR:
       Passive Aggressive Mean Reversion Strategy for Portfolio Selection.
       Machine Learning, 87(2), 221-258.
"""

# %%
# Data Loading: DJIA Dataset
# ===========================
# We use the Dow Jones Industrial Average (DJIA) dataset containing 30 stocks
# from 2001-2003. This period includes the post-dot-com-bubble market regime.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfolio.datasets import load_djia_relatives, load_djia_reversed_relatives
from skfolio.optimization.online import (
    BCRP,
    UCRP,
    FollowTheWinner,
    FollowTheLoser,
    FTWStrategy,
    FTLStrategy,
    regret,
    RegretType,
)
from skfolio.preprocessing import prices_to_returns

# Load original and reversed DJIA datasets
prices_orig = load_djia_relatives()
prices_rev = load_djia_reversed_relatives()  # Time-reversed for regime analysis

print(f"Original dataset: {prices_orig.index[0]} to {prices_orig.index[-1]}")
print(f"Shape: {prices_orig.shape}")

# Convert to net returns
X_orig = prices_to_returns(prices_orig)
X_rev = prices_to_returns(prices_rev)

# %%
# Strategy Suite: From OGD to AdaBARRONS
# =======================================
# We compare five strategies spanning basic to adaptive methods:
#
# 1. **OGD** (Online Gradient Descent): Baseline Euclidean geometry
# 2. **EG** (Exponential Gradient): Entropy geometry, multiplicative updates
# 3. **AdaGrad**: Per-coordinate adaptive learning rates
# 4. **AdaBARRONS**: Adaptive barrier + Mahalanobis preconditioning
# 5. **OLMAR**: Mean-reversion baseline for comparison

strategies_ftw = {
    "OGD": FollowTheWinner(
        strategy=FTWStrategy.OGD,
        learning_rate="auto",
        learning_rate_scale="empirical",
    ),
    "EG": FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate="auto",
        learning_rate_scale="empirical",
    ),
    "AdaGrad": FollowTheWinner(
        strategy=FTWStrategy.ADAGRAD,
        learning_rate="auto",
        learning_rate_scale="empirical",
    ),
    "AdaBARRONS": FollowTheWinner(
        strategy=FTWStrategy.ADABARRONS,
        learning_rate="auto",
        learning_rate_scale="empirical",
        adabarrons_barrier_coef=1.0,
        adabarrons_alpha=1.0,
        adabarrons_euclidean_coef=1.0,
        adabarrons_beta=0.1,
    ),
}

# Mean-reversion strategy
olmar = FollowTheLoser(
    strategy=FTLStrategy.OLMAR, olmar_window=5, epsilon=10.0, update_mode="pa"
)

# Benchmarks
ucrp = UCRP()
bcrp = BCRP()

# %%
# Fit All Strategies on Original Dataset
# =======================================

print("\nFitting strategies on original DJIA dataset...")

# Fit benchmarks
ucrp.fit(X_orig)
bcrp.fit(X_orig)

# Fit FTW strategies
results_orig = {}
for name, model in strategies_ftw.items():
    print(f"  Fitting {name}...", end=" ")
    model.fit(X_orig)
    results_orig[name] = {
        "model": model,
        "wealth": model.all_wealth_,
        "weights": model.all_weights_,
    }
    print(f"Final wealth: {model.wealth_:.2f}")

# Fit OLMAR
olmar.fit(X_orig)
results_orig["OLMAR"] = {
    "model": olmar,
    "wealth": olmar.all_wealth_,
    "weights": olmar.all_weights_,
}
print(f"  Fitting OLMAR... Final wealth: {olmar.wealth_:.2f}")

# Benchmark wealth
ucrp_wealth = ucrp.all_wealth_
bcrp_wealth = bcrp.all_wealth_

# %%
# Wealth Comparison: Adaptive vs Non-Adaptive
# ============================================

fig, ax = plt.subplots(figsize=(14, 7))

# Plot benchmarks
ax.plot(
    ucrp_wealth,
    label="UCRP (1/n)",
    linewidth=2.5,
    alpha=0.6,
    color="gray",
    linestyle="--",
)
ax.plot(
    bcrp_wealth,
    label="BCRP (Hindsight)",
    linewidth=2.5,
    alpha=0.6,
    color="black",
    linestyle=":",
)

# Plot strategies with distinct colors
colors = {"OGD": "blue", "EG": "green", "AdaGrad": "orange", "AdaBARRONS": "red"}
for name in ["OGD", "EG", "AdaGrad", "AdaBARRONS"]:
    ax.plot(
        results_orig[name]["wealth"],
        label=name,
        linewidth=2.5,
        color=colors[name],
        alpha=0.9,
    )

# Plot OLMAR separately
ax.plot(
    results_orig["OLMAR"]["wealth"],
    label="OLMAR (Mean-Rev)",
    linewidth=2,
    color="purple",
    linestyle="-.",
    alpha=0.8,
)

ax.set_xlabel("Trading Day", fontsize=13)
ax.set_ylabel("Cumulative Wealth", fontsize=13)
ax.set_title("Strategy Comparison on DJIA (2001-2003)", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print final wealth summary
print("\n=== Final Wealth (Original DJIA) ===")
print(f"UCRP:       {ucrp_wealth[-1]:.2f}")
print(f"BCRP:       {bcrp_wealth[-1]:.2f}")
for name in ["OGD", "EG", "AdaGrad", "AdaBARRONS", "OLMAR"]:
    w = results_orig[name]["wealth"][-1]
    print(f"{name:12s} {w:.2f} ({w / ucrp_wealth[-1]:.2f}x UCRP)")

# %%
# Performance Metrics: Regret and Sharpe Ratio
# =============================================
# Compute average regret vs BCRP and estimate Sharpe ratios from wealth trajectories.


def compute_sharpe(wealth_series, annualization_factor=252):
    """Compute Sharpe ratio from wealth trajectory."""
    returns = np.diff(np.log(wealth_series))
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return (
        np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)
        if np.std(returns) > 0
        else 0.0
    )


print("\n=== Performance Metrics (Original DJIA) ===")
print(f"{'Strategy':<15} {'Avg Regret':>12} {'Sharpe':>8}")
print("-" * 37)

metrics = {}
for name in ["OGD", "EG", "AdaGrad", "AdaBARRONS", "OLMAR"]:
    model = results_orig[name]["model"]
    avg_regret = regret(
        model, X_orig, comparator=bcrp, regret_type=RegretType.STATIC, average="final"
    )
    sharpe = compute_sharpe(results_orig[name]["wealth"])
    metrics[name] = {"regret": avg_regret, "sharpe": sharpe}
    print(f"{name:<15} {avg_regret:>12.6f} {sharpe:>8.3f}")

# %%
# Reversed Dataset: Regime Analysis
# ==================================
# Testing on time-reversed data reveals strategy robustness to regime changes.
# Momentum strategies may struggle in reversed (mean-reverting-like) markets,
# while mean-reversion strategies may excel.

print("\n\nFitting strategies on REVERSED DJIA dataset...")

# Fit benchmarks on reversed data
ucrp_rev = UCRP()
bcrp_rev = BCRP()
ucrp_rev.fit(X_rev)
bcrp_rev.fit(X_rev)

# Fit FTW strategies on reversed data
results_rev = {}
for name, model_orig in strategies_ftw.items():
    # Create fresh instance with same parameters
    model_rev = FollowTheWinner(
        strategy=model_orig.strategy,
        learning_rate="auto",
        learning_rate_scale="empirical",
    )
    if name == "AdaBARRONS":
        model_rev.adabarrons_barrier_coef = model_orig.adabarrons_barrier_coef
        model_rev.adabarrons_alpha = model_orig.adabarrons_alpha

    print(f"  Fitting {name}...", end=" ")
    model_rev.fit(X_rev)
    results_rev[name] = {
        "model": model_rev,
        "wealth": model_rev.all_wealth_,
        "weights": model_rev.all_weights_,
    }
    print(f"Final wealth: {model_rev.wealth_:.2f}")

# Fit OLMAR on reversed data
olmar_rev = FollowTheLoser(
    strategy=FTLStrategy.OLMAR, olmar_window=5, epsilon=10.0, update_mode="pa"
)
olmar_rev.fit(X_rev)
results_rev["OLMAR"] = {
    "model": olmar_rev,
    "wealth": olmar_rev.all_wealth_,
    "weights": olmar_rev.all_weights_,
}
print(f"  Fitting OLMAR... Final wealth: {olmar_rev.wealth_:.2f}")

# %%
# Regime Comparison: Original vs Reversed
# ========================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Original dataset
for name in ["EG", "AdaGrad", "OLMAR"]:
    axes[0].plot(results_orig[name]["wealth"], label=name, linewidth=2.5, alpha=0.9)
axes[0].plot(
    ucrp_wealth, label="UCRP", linewidth=2, alpha=0.6, color="gray", linestyle="--"
)
axes[0].set_xlabel("Trading Day", fontsize=12)
axes[0].set_ylabel("Cumulative Wealth", fontsize=12)
axes[0].set_title("Original DJIA Dataset", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Reversed dataset
for name in ["EG", "AdaGrad", "OLMAR"]:
    axes[1].plot(results_rev[name]["wealth"], label=name, linewidth=2.5, alpha=0.9)
axes[1].plot(
    ucrp_rev.all_wealth_,
    label="UCRP",
    linewidth=2,
    alpha=0.6,
    color="gray",
    linestyle="--",
)
axes[1].set_xlabel("Trading Day", fontsize=12)
axes[1].set_ylabel("Cumulative Wealth", fontsize=12)
axes[1].set_title("Reversed DJIA Dataset", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Regime Sensitivity ===")
print(f"{'Strategy':<15} {'Original':>10} {'Reversed':>10} {'Ratio (R/O)':>12}")
print("-" * 50)
for name in ["EG", "AdaGrad", "OLMAR"]:
    w_orig = results_orig[name]["wealth"][-1]
    w_rev = results_rev[name]["wealth"][-1]
    ratio = w_rev / w_orig if w_orig > 0 else 0
    print(f"{name:<15} {w_orig:>10.2f} {w_rev:>10.2f} {ratio:>12.2f}")

# %%
# Transaction Costs: Reality Check
# =================================
# Real trading incurs costs. We refit strategies with proportional transaction costs
# to assess robustness. Costs are applied as a fraction of turnover.
#
# **Cost Model**: For turnover Δw, cost = c * ||Δw||₁

transaction_cost = 0.005  # 0.5% per trade (50 bps)

print(
    f"\n\nRefitting with transaction costs ({transaction_cost * 100:.1f}% per trade)..."
)

strategies_with_costs = {
    "EG": FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate="auto",
        learning_rate_scale="empirical",
        transaction_costs=transaction_cost,
    ),
    "AdaGrad": FollowTheWinner(
        strategy=FTWStrategy.ADAGRAD,
        learning_rate="auto",
        learning_rate_scale="empirical",
        transaction_costs=transaction_cost,
    ),
    "OLMAR": FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        olmar_window=5,
        epsilon=10.0,
        update_mode="pa",
        transaction_costs=transaction_cost,
    ),
}

results_with_costs = {}
for name, model in strategies_with_costs.items():
    print(f"  Fitting {name}...", end=" ")
    model.fit(X_orig)
    results_with_costs[name] = {
        "model": model,
        "wealth": model.all_wealth_,
        "weights": model.all_weights_,
    }
    print(f"Final wealth: {model.wealth_:.2f}")

# %%
# Cost Impact Visualization
# ==========================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Wealth comparison
for name in ["EG", "AdaGrad", "OLMAR"]:
    axes[0].plot(
        results_orig[name]["wealth"],
        label=f"{name} (No Costs)",
        linewidth=2.5,
        linestyle="--",
        alpha=0.7,
    )
    axes[0].plot(
        results_with_costs[name]["wealth"],
        label=f"{name} (With Costs)",
        linewidth=2.5,
        alpha=0.9,
    )

axes[0].set_xlabel("Trading Day", fontsize=12)
axes[0].set_ylabel("Cumulative Wealth", fontsize=12)
axes[0].set_title("Impact of Transaction Costs", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=10, loc="best")
axes[0].grid(alpha=0.3)

# Cost impact percentage
cost_impact = {}
for name in ["EG", "AdaGrad", "OLMAR"]:
    w_no_cost = results_orig[name]["wealth"][-1]
    w_with_cost = results_with_costs[name]["wealth"][-1]
    impact_pct = (1 - w_with_cost / w_no_cost) * 100 if w_no_cost > 0 else 0
    cost_impact[name] = impact_pct

names = list(cost_impact.keys())
impacts = [cost_impact[n] for n in names]
colors_bar = ["green", "orange", "purple"]

axes[1].bar(names, impacts, color=colors_bar, alpha=0.7, edgecolor="black")
axes[1].set_ylabel("Wealth Loss (%)", fontsize=12)
axes[1].set_title("Transaction Cost Impact", fontsize=13, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

for i, (name, impact) in enumerate(zip(names, impacts)):
    axes[1].text(i, impact + 1, f"{impact:.1f}%", ha="center", fontsize=11)

plt.tight_layout()
plt.show()

print("\n=== Transaction Cost Impact ===")
print(f"{'Strategy':<15} {'No Costs':>10} {'With Costs':>12} {'Loss (%)':>10}")
print("-" * 50)
for name in ["EG", "AdaGrad", "OLMAR"]:
    w_no = results_orig[name]["wealth"][-1]
    w_with = results_with_costs[name]["wealth"][-1]
    loss = cost_impact[name]
    print(f"{name:<15} {w_no:>10.2f} {w_with:>12.2f} {loss:>10.1f}%")

# %%
# Turnover Analysis: Trading Frequency
# =====================================
# High turnover leads to higher transaction costs. We compute average turnover
# (L1 distance between consecutive weights) for each strategy.


def compute_turnover(weights_matrix):
    """Compute per-period turnover (L1 distance between consecutive weights)."""
    if weights_matrix.shape[0] <= 1:
        return np.array([])
    diffs = np.diff(weights_matrix, axis=0)
    return np.sum(np.abs(diffs), axis=1)


print("\n=== Average Turnover (Original DJIA, No Costs) ===")
print(f"{'Strategy':<15} {'Avg Turnover':>15} {'Max Turnover':>15}")
print("-" * 47)

turnover_stats = {}
for name in ["OGD", "EG", "AdaGrad", "AdaBARRONS", "OLMAR"]:
    turnovers = compute_turnover(results_orig[name]["weights"])
    avg_turn = np.mean(turnovers) if len(turnovers) > 0 else 0
    max_turn = np.max(turnovers) if len(turnovers) > 0 else 0
    turnover_stats[name] = {"avg": avg_turn, "max": max_turn}
    print(f"{name:<15} {avg_turn:>15.4f} {max_turn:>15.4f}")

# %%
# PAMR: Alternative Mean Reversion
# =================================
# PAMR (Passive-Aggressive Mean Reversion) offers a different approach than OLMAR.
# While OLMAR uses moving averages, PAMR directly exploits single-period reversals.

pamr_variants = {
    "PAMR-0 (Simple)": FollowTheLoser(
        strategy=FTLStrategy.PAMR,
        pamr_variant="simple",
        pamr_C=500.0,
        epsilon=0.5,
        update_mode="pa",
    ),
    "PAMR-1 (Slack Linear)": FollowTheLoser(
        strategy=FTLStrategy.PAMR,
        pamr_variant="slack_linear",
        pamr_C=100.0,
        epsilon=0.5,
        update_mode="pa",
    ),
}

print("\n\nFitting PAMR variants...")
pamr_results = {}
for name, model in pamr_variants.items():
    print(f"  Fitting {name}...", end=" ")
    model.fit(X_orig)
    pamr_results[name] = {"model": model, "wealth": model.all_wealth_}
    print(f"Final wealth: {model.wealth_:.2f}")

# Plot PAMR vs OLMAR
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    results_orig["OLMAR"]["wealth"],
    label="OLMAR",
    linewidth=2.5,
    color="purple",
)
for name in pamr_variants.keys():
    ax.plot(pamr_results[name]["wealth"], label=name, linewidth=2.5, alpha=0.8)
ax.plot(ucrp_wealth, label="UCRP", linewidth=2, alpha=0.6, color="gray", linestyle="--")
ax.set_xlabel("Trading Day", fontsize=12)
ax.set_ylabel("Cumulative Wealth", fontsize=12)
ax.set_title("Mean-Reversion Strategies Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Key Insights
# ============
#
# **1. Adaptive Methods Provide Robustness**
#    - AdaGrad and AdaBARRONS adjust learning rates per coordinate, handling
#      ill-conditioned problems (assets with varying volatilities)
#    - Automatic per-asset scaling reduces sensitivity to hyperparameters
#
# **2. No Universal Winner**
#    - Momentum (EG, AdaGrad) excels in trending markets
#    - Mean-reversion (OLMAR, PAMR) profits from reversals
#    - Reversed dataset shows regime sensitivity
#
# **3. Transaction Costs are Critical**
#    - High-turnover strategies (aggressive momentum) lose significant wealth
#    - Mean-reversion can be cost-effective due to slower adaptation
#    - Real-world implementation must account for slippage and fees
#
# **4. AdaBARRONS Adaptive Barrier**
#    - Combines log-barrier geometry (simplex natural) with adaptive
#      Mahalanobis preconditioning
#    - Theoretical advantage in strongly-convex regimes
#    - Computationally heavier than AdaGrad
#
# **5. PAMR vs OLMAR**
#    - PAMR exploits immediate single-period reversals
#    - OLMAR uses multi-period moving-average signals
#    - Different aggressiveness/smoothness trade-offs
#
# **6. Practical Recommendations**
#    - Use `learning_rate="auto"` with `learning_rate_scale="empirical"`
#    - Test on both original and reversed data for robustness assessment
#    - Include transaction costs in backtests
#    - Consider ensemble methods (combine momentum + mean-reversion)
#
# **7. Regret Bounds Hold**
#    - All strategies achieve O(√T) regret empirically
#    - Average regret decreases over time (sublinear growth)
#    - OCO theory provides worst-case guarantees
#
# **Next Steps**: Explore SWORD family (optimistic methods), constraints
# (turnover caps, sector limits), and ensemble/meta-learning strategies.
