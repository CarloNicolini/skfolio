"""
=================================
Online Portfolio Selection basics
=================================

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
# We use the Toronto Stock Exchange (TSE) dataset from Li & Hoi's OLPS benchmarks. This dataset contains daily price relatives (gross returns) for 88 stocks from 1994 to 1998.

from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.online import BCRP, UCRP, FTWStrategy, FollowTheWinner
from skfolio.population import Population
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)

# %%
# Benchmark Portfolios
# ====================
# We first establish baseline performance using three classic benchmarks:
#
# 1. **Uniform CRP (UCRP)**: Rebalances to equal weights (1/n) each period
# 2. **BCRP**: Best Constant Rebalanced Portfolio in hindsight (upper bound)
# 3. **Exponential Gradient**: A Follow-the-Winner strategy that updates the portfolio weights using an exponential gradient method also called the Hedge algorithm.


population = []
population.append(
    BCRP(portfolio_params={"name": "Maximum LogWealth in Hindsight"}).fit_predict(X)
)
population.append(
    UCRP(
        portfolio_params={"name": "Uniform Constant Rebalanced Portfolio"}
    ).fit_predict(X)
)
population.append(
    FollowTheWinner(
        strategy=FTWStrategy.EG, portfolio_params={"name": "Exponential Gradient"}
    ).fit_predict(X)
)

# Here we create a Population of the three portfolios and plot the cumulative returns.
# To better show the performance of the strategies, we use a log scale and set the compounded parameter to True since typically in online portfolio selection, the returns are compounded (the objective function is cumulative wealth maximization).
population = Population(population)
population.set_portfolio_params(compounded=True)
fig = population.plot_cumulative_returns(log_scale=True)
show(fig)


# The important point to note here is that while the Best in Hindsight portfolio (BCRP) outperforms the other two strategies, the Exponential Gradient strategy (EG) did not see all the data, instead returns are revealed sequentially one day at a time.
# The performance of the EG strategy is therefore expected to be inferior to the BCRP strategy but it's still better than the Uniform Constant Rebalanced Portfolio (UCRP).

# %%
# We print the summary of the population:
print(population.summary())
