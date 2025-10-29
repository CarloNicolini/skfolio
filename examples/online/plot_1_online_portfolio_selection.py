"""
========================================================================================
Introduction to Online Portfolio Selection: Follow-the-Winner vs Follow-the-Loser
========================================================================================

This tutorial introduces Online Portfolio Selection (OPS) through practical examples,
comparing momentum-based (Follow-the-Winner) and mean-reversion (Follow-the-Loser)
strategies.

**What is Online Portfolio Selection?**

Unlike traditional portfolio optimization that relies on historical data to compute
optimal weights once, Online Portfolio Selection updates weights sequentially after
observing each period's returns. This online learning framework is well-suited for:

- **Non-stationary markets** where distribution of returns changes over time
- **High-frequency rebalancing** where computation must be fast
- **Adaptive strategies** that respond to recent market behavior

**Two Families of Strategies**

1. **Follow-the-Winner (Momentum)**:
   Increases allocation to recently successful assets. Examples: Exponential Gradient (EG),
   Online Gradient Descent (OGD), AdaGrad.

2. **Follow-the-Loser (Mean Reversion)**:
   Shifts wealth from recent winners to recent losers, betting on price reversals.
   Examples: OLMAR, PAMR, CWMR.

**OCO Framework**

Both families fit within Online Convex Optimization (OCO), which provides theoretical
regret guarantees: the cumulative difference between the online algorithm's performance
and the best constant portfolio in hindsight grows sublinearly (typically O(√T)).

References
----------
.. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
       arXiv:1212.2129.
.. [2] Hazan, E. (2016). Introduction to Online Convex Optimization.
       Foundations and Trends in Optimization, 2(3-4), 157-325.
"""

# %%
# Data Loading
# ============
# We use the Toronto Stock Exchange (TSE) dataset from Li & Hoi's OLPS benchmarks.
# This dataset contains daily price relatives (gross returns) for 88 stocks
# from 1994 to 1998.

import numpy as np
import matplotlib.pyplot as plt
from skfolio.datasets import load_tse_relatives
from skfolio.optimization.online import (
    BCRP,
    UCRP,
    BestStock,
    FollowTheWinner,
    FollowTheLoser,
    FTWStrategy,
    FTLStrategy,
    regret,
    RegretType,
)
from skfolio.preprocessing import prices_to_returns

# Load TSE dataset as price relatives (gross returns: 1 + net_returns)
prices = load_tse_relatives()
print(f"Dataset shape: {prices.shape}")
print(f"Period: {prices.index[0]} to {prices.index[-1]}")

# Convert to net returns for skfolio estimators
X = prices_to_returns(prices)

# %%
# Benchmark Portfolios
# ====================
# We first establish baseline performance using three classic benchmarks:
#
# 1. **Uniform CRP (UCRP)**: Rebalances to equal weights (1/n) each period
# 2. **Best Stock**: Invests all wealth in the single best-performing asset
# 3. **BCRP**: Best Constant Rebalanced Portfolio in hindsight (upper bound)

# Fit benchmarks
ucrp = UCRP()
ucrp.fit(X)

best_stock = BestStock()
best_stock.fit(X)

bcrp = BCRP()  # Maximizes cumulative log-wealth
bcrp.fit(X)

# Extract wealth trajectories
ucrp_wealth = ucrp.all_wealth_
best_stock_wealth = best_stock.all_wealth_
bcrp_wealth = bcrp.all_wealth_

# Plot wealth evolution
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ucrp_wealth, label="UCRP (1/n)", linewidth=2)
ax.plot(best_stock_wealth, label="Best Stock", linewidth=2, linestyle="--")
ax.plot(bcrp_wealth, label="BCRP (Hindsight)", linewidth=2, linestyle=":")
ax.set_xlabel("Trading Day", fontsize=12)
ax.set_ylabel("Cumulative Wealth", fontsize=12)
ax.set_title("Benchmark Portfolio Performance", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"UCRP final wealth: {ucrp_wealth[-1]:.2f}")
print(f"Best Stock final wealth: {best_stock_wealth[-1]:.2f}")
print(f"BCRP final wealth: {bcrp_wealth[-1]:.2f}")

# %%
# Follow-the-Winner: Exponential Gradient
# ========================================
# Exponential Gradient (EG) uses entropy mirror map for multiplicative weight updates.
# It increases weights on assets with positive gradients (recent gains) exponentially.
#
# **Why EG?**
# - Guarantees O(√T log n) regret against best constant portfolio
# - Natural for simplex constraints (weights stay positive, sum to 1)
# - Aggressive momentum: small weights can quickly grow

eg = FollowTheWinner(
    strategy=FTWStrategy.EG,
    learning_rate="auto",  # Automatically tuned for dataset size
    learning_rate_scale="empirical",  # Use empirically-validated scaling
)
eg.fit(X)

# %%
# Follow-the-Loser: OLMAR (Online Moving Average Reversion)
# ==========================================================
# OLMAR exploits mean reversion by betting that prices will revert to their moving
# average. It shifts weight from assets above their MA to those below.
#
# **Parameters:**
# - `olmar_window`: Moving average window (default=5)
# - `epsilon`: Margin threshold for detecting reversion opportunities
# - `update_mode`: "pa" (Passive-Aggressive, exact) or "md" (Mirror Descent, flexible)

olmar = FollowTheLoser(
    strategy=FTLStrategy.OLMAR,
    olmar_window=5,  # 5-day moving average
    epsilon=10.0,  # Aggressive reversion threshold
    update_mode="pa",  # Closed-form PA update
)
olmar.fit(X)

# %%
# Wealth Comparison: Winner vs Loser
# ===================================
# Compare the two strategies against benchmarks.

eg_wealth = eg.all_wealth_
olmar_wealth = olmar.all_wealth_

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ucrp_wealth, label="UCRP (Benchmark)", linewidth=2, alpha=0.7, color="gray")
ax.plot(
    bcrp_wealth,
    label="BCRP (Hindsight)",
    linewidth=2,
    linestyle=":",
    alpha=0.7,
    color="black",
)
ax.plot(eg_wealth, label="EG (Follow-the-Winner)", linewidth=2.5, color="blue")
ax.plot(olmar_wealth, label="OLMAR (Follow-the-Loser)", linewidth=2.5, color="red")
ax.set_xlabel("Trading Day", fontsize=12)
ax.set_ylabel("Cumulative Wealth", fontsize=12)
ax.set_title(
    "Follow-the-Winner vs Follow-the-Loser on TSE Dataset",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nFinal Wealth Comparison:")
print(f"  UCRP:  {ucrp_wealth[-1]:.2f}")
print(f"  EG:    {eg_wealth[-1]:.2f} ({eg_wealth[-1] / ucrp_wealth[-1]:.2f}x UCRP)")
print(
    f"  OLMAR: {olmar_wealth[-1]:.2f} ({olmar_wealth[-1] / ucrp_wealth[-1]:.2f}x UCRP)"
)
print(f"  BCRP:  {bcrp_wealth[-1]:.2f} ({bcrp_wealth[-1] / ucrp_wealth[-1]:.2f}x UCRP)")

# %%
# Regret Analysis
# ===============
# **Regret** measures the difference in cumulative loss between the online algorithm
# and a comparator. We compute:
#
# - **Static regret**: Comparison to best constant portfolio (BCRP)
# - **Average regret**: Normalized by number of periods
#
# Lower regret indicates better performance relative to the hindsight benchmark.

# Compute static regret (averaged per period)
eg_regret = regret(
    eg,
    X,
    comparator=bcrp,
    regret_type=RegretType.STATIC,
    average="final",  # Average over all periods
)

olmar_regret = regret(
    olmar,
    X,
    comparator=bcrp,
    regret_type=RegretType.STATIC,
    average="final",
)

print(f"\nAverage Regret vs BCRP:")
print(f"  EG:    {eg_regret:.6f}")
print(f"  OLMAR: {olmar_regret:.6f}")

# %%
# Cumulative Regret Curves
# ========================
# Visualize how regret accumulates over time. Ideally, regret grows sublinearly
# (curve flattens), indicating the algorithm is "learning" the market.

eg_regret_curve = regret(
    eg, X, comparator=bcrp, regret_type=RegretType.STATIC, average=False
)

olmar_regret_curve = regret(
    olmar, X, comparator=bcrp, regret_type=RegretType.STATIC, average=False
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(eg_regret_curve, label="EG", linewidth=2, color="blue")
ax.plot(olmar_regret_curve, label="OLMAR", linewidth=2, color="red")
ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Trading Day", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title("Regret Accumulation vs BCRP", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Weight Evolution: Observing Strategy Behavior
# ==============================================
# Examine how strategies adapt weights over time. We plot the top 5 assets by
# final allocation for each strategy.

# Get final weights
eg_final_weights = eg.weights_
olmar_final_weights = olmar.weights_

# Top 5 assets by final weight
eg_top5_idx = np.argsort(eg_final_weights)[-5:][::-1]
olmar_top5_idx = np.argsort(olmar_final_weights)[-5:][::-1]

# Plot EG weight evolution
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# EG
for idx in eg_top5_idx:
    axes[0].plot(eg.all_weights_[:, idx], label=f"Asset {idx}", linewidth=1.5)
axes[0].set_xlabel("Trading Day", fontsize=12)
axes[0].set_ylabel("Weight", fontsize=12)
axes[0].set_title("EG: Top 5 Asset Weights", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# OLMAR
for idx in olmar_top5_idx:
    axes[1].plot(olmar.all_weights_[:, idx], label=f"Asset {idx}", linewidth=1.5)
axes[1].set_xlabel("Trading Day", fontsize=12)
axes[1].set_ylabel("Weight", fontsize=12)
axes[1].set_title("OLMAR: Top 5 Asset Weights", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Portfolio Concentration
# =======================
# Measure concentration using the Herfindahl-Hirschman Index (HHI).
# HHI = Σ w_i² ranges from 1/n (uniform) to 1 (single asset).
#
# Higher HHI indicates more concentrated portfolios.


def herfindahl_index(weights):
    """Compute Herfindahl-Hirschman Index (portfolio concentration)."""
    return np.sum(weights**2, axis=1)


eg_hhi = herfindahl_index(eg.all_weights_)
olmar_hhi = herfindahl_index(olmar.all_weights_)
ucrp_hhi = 1 / X.shape[1]  # Constant for uniform portfolio

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(eg_hhi, label="EG", linewidth=2, color="blue")
ax.plot(olmar_hhi, label="OLMAR", linewidth=2, color="red")
ax.axhline(
    ucrp_hhi,
    color="gray",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"UCRP (1/n = {ucrp_hhi:.3f})",
)
ax.set_xlabel("Trading Day", fontsize=12)
ax.set_ylabel("HHI (Concentration)", fontsize=12)
ax.set_title("Portfolio Concentration Over Time", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nAverage Portfolio Concentration (HHI):")
print(f"  UCRP:  {ucrp_hhi:.4f} (uniform baseline)")
print(f"  EG:    {np.mean(eg_hhi):.4f}")
print(f"  OLMAR: {np.mean(olmar_hhi):.4f}")

# %%
# Key Takeaways
# =============
#
# 1. **Market Regime Matters**:
#    - Follow-the-Winner excels in trending markets (momentum)
#    - Follow-the-Loser profits from mean-reverting markets
#
# 2. **Sublinear Regret**:
#    - Both strategies achieve O(√T) regret, meaning average per-period loss
#      relative to BCRP decreases as 1/√T
#
# 3. **Concentration vs Diversification**:
#    - EG tends to concentrate on winners (momentum)
#    - OLMAR can maintain broader diversification (reversion spreads bets)
#
# 4. **No Free Lunch**:
#    - Neither strategy dominates universally
#    - BCRP is the theoretical upper bound (computed with perfect hindsight)
#
# 5. **Practical Considerations**:
#    - Transaction costs favor less aggressive strategies
#    - Learning rates require tuning (use `learning_rate="auto"` for convenience)
#    - Constraints (turnover, sector caps) are supported via projections
#
# **Next Steps**: Explore advanced strategies (AdaGrad, SWORD), transaction costs,
# and dynamic regret in the companion example.
