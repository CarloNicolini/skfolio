"""
===================================
Online Portfolio Selection Advanced
===================================

This tutorial explores advanced Online Portfolio Selection (OPS) strategies, comparing
adaptive momentum methods against mean-reversion algorithms on a longer history.

We will evaluate the following algorithms:
- **AdaGrad (Follow The Winner)**: An adaptive gradient method that adjusts the learning rate
  individually for each asset based on historical gradient variations.
- **OLMAR (Follow The Loser)**: Online Moving Average Reversion, which shifts wealth toward
  assets that have recently underperformed their moving average, betting on price reversals.

References
----------
.. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
       arXiv:1212.2129.
.. [2] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for
       Online Learning and Stochastic Optimization. JMLR.
"""

# %%
# Data Loading
# ============
# We use the S&P 500 :ref:`dataset <datasets>` and extract the returns from 2000 to 2022.
# We focus on the tech-heavy or volatile subset by dropping the least volatile assets,
# as mean-reversion strategies often perform best in more dynamic environments.

import numpy as np
from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.online import (
    BCRP,
    UCRP,
    FTWStrategy,
    FollowTheLoser,
    FollowTheWinner,
)
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)["2000":]

# %%
# Model Initialization
# ====================
# We initialize our advanced online models alongside standard hindsight benchmarks.

# Benchmarks
ucrp = UCRP()
bcrp = BCRP()

# Follow The Winner: AdaGrad
# AdaGrad uses an adaptive Euclidean geometry that makes it robust to varying market conditions.
adagrad = FollowTheWinner(
    strategy=FTWStrategy.ADAGRAD,
    learning_rate="auto",
    learning_rate_scale="empirical",
)

# Follow The Loser: OLMAR
# OLMAR bets on short-term price reversals based on a moving average predictor.
olmar = FollowTheLoser(
    strategy="olmar",
    olmar_window=5,  # 5-day moving average
    epsilon=10.0,  # Reversion threshold
)

# %%
# Sequential Fitting
# ==================
# We fit the estimators on the sequential data.

ucrp.fit(X)
bcrp.fit(X)
adagrad.fit(X)
olmar.fit(X)

# %%
# Online Trajectory Evaluation
# ============================
# To evaluate these online learning algorithms correctly without look-ahead bias, we
# reconstruct the chronological trajectory of the portfolios. We do this by calling
# the `fit_predict` method, which for online models is designed to return a
# :class:`~skfolio.portfolio.MultiPeriodPortfolio` representing the sequence of
# portfolios traded prior to observing each period's returns.

ptf_bcrp = bcrp.fit_predict(X)
ptf_bcrp.name = "Best In Hindsight (BCRP)"

ptf_ucrp = ucrp.fit_predict(X)
ptf_ucrp.name = "UCRP (1/n)"

ptf_adagrad = adagrad.fit_predict(X)
ptf_adagrad.name = "AdaGrad (Momentum)"

ptf_olmar = olmar.fit_predict(X)
ptf_olmar.name = "OLMAR (Mean Reversion)"

population = Population(
    [
        ptf_bcrp,
        ptf_ucrp,
        ptf_adagrad,
        ptf_olmar,
    ]
)

# Use compounded returns for online portfolio analysis (capital growth)
population.set_portfolio_params(compounded=True)

# %%
# Visualization
# =============
# Let's plot the cumulative wealth over time.

fig = population.plot_cumulative_returns(log_scale=True)
show(fig)

# %%
# In this environment, mean-reversion strategies like OLMAR can capture significant
# short-term volatility, occasionally even surpassing the best constant rebalanced
# portfolio (BCRP) over specific timeframes (which is theoretically possible since BCRP
# is a *constant* weight portfolio, whereas OLMAR is dynamically rebalanced and can change
# weights aggressively).

# %%
# Summary
# =======
# We print the summary of the population.
print(population.summary())

# %%
# Weight Allocation
# =================
# Finally, let's observe how the AdaGrad strategy adapted its weights sequentially.
# We extract the `MultiPeriodPortfolio` corresponding to AdaGrad and plot its composition
# over time.
mpp_adagrad = population[2]
fig_weights = mpp_adagrad.plot_weights_per_observation()
show(fig_weights)
