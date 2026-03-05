"""
=================================
Online Portfolio Selection Basics
=================================

This tutorial introduces Online Portfolio Selection (OPS) through practical examples,
comparing momentum-based (Follow-the-Winner) and constant rebalanced portfolios.

**What is Online Portfolio Selection?**

Unlike traditional offline portfolio optimization that relies on a historical training set
to compute optimal weights once, Online Portfolio Selection updates weights sequentially
after observing each period's returns. This online learning framework is well-suited for:

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
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition. Prices are transformed into linear returns.
# To keep the example execution fast, we focus on the data from 2010 onwards.

from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.online import BCRP, UCRP, FTWStrategy, FollowTheWinner
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)["2010":]

# %%
# Benchmark Portfolios
# ====================
# We first establish baseline performance using two classic benchmarks:
#
# 1. **Uniform CRP (UCRP)**: Rebalances to equal weights (1/n) each period
# 2. **BCRP**: Best Constant Rebalanced Portfolio in hindsight (upper bound)

ucrp = UCRP()
ucrp.fit(X)

bcrp = BCRP()
bcrp.fit(X)

# %%
# Online Portfolio Model
# ======================
# We then instantiate our online model using the **Exponential Gradient** strategy.
# It is a Follow-the-Winner strategy that updates the portfolio weights using an
# exponential gradient method (also known as the Hedge algorithm).

eg = FollowTheWinner(strategy=FTWStrategy.EG)
eg.fit(X)

# %%
# Sequential Evaluation
# =====================
# In traditional offline optimization, estimators fit on a training set and predict on
# a test set. Calling `predict` applies the *final* learned weights uniformly across
# the test periods.
#
# In online learning, the model processes data sequentially. To properly evaluate the
# model without look-ahead bias, the `fit_predict` method in skfolio's online module
# has been specifically designed to return a :class:`~skfolio.portfolio.MultiPeriodPortfolio`.
# This object represents the true chronological online performance, evaluating each period
# using the weights computed *prior* to observing that period's returns.

ptf_ucrp = ucrp.fit_predict(X)
ptf_ucrp.name = "UCRP (1/n)"

ptf_bcrp = bcrp.fit_predict(X)
ptf_bcrp.name = "Best In Hindsight (BCRP)"

ptf_eg = eg.fit_predict(X)
ptf_eg.name = "Exponential Gradient"

# %%
# Analysis
# ========
# We can now group these trajectories into a :class:`~skfolio.population.Population`
# and analyze their performance. In online portfolio selection, we typically focus on
# compounded cumulative returns (which measures capital growth).

population = Population([ptf_ucrp, ptf_bcrp, ptf_eg])
population.set_portfolio_params(compounded=True)

fig = population.plot_cumulative_returns(log_scale=True)
show(fig)

# %%
# The important point to note here is that while the Best in Hindsight portfolio (BCRP)
# outperforms the other two strategies, it requires full future knowledge. The Exponential
# Gradient strategy (EG) did not see all the data in advance; instead, returns were revealed
# sequentially one day at a time. The performance of the EG strategy is therefore expected
# to be inferior to the BCRP strategy but often remains competitive or better than the Uniform
# Constant Rebalanced Portfolio (UCRP).

# %%
# We print the summary of the population to inspect their metrics:
print(population.summary())
