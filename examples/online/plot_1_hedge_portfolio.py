"""
=================
Hedge (Multiplicative Weights)
=================

This example demonstrates the :class:`~skfolio.optimization.HedgePortfolio`
estimator, which implements the multiplicative weights (Hedge) update.
We compare it to simple baselines like equally-weighted and inverse volatility.
"""

# %%
# Data
# ====
# Load the S&P 500 dataset, convert prices to arithmetic returns, and split
# into train/test without shuffling.

from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import EqualWeighted, ExponentialGradient, InverseVolatility
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Models
# ======
# Fit HedgePortfolio and baselines on the training data.

hedge = ExponentialGradient(portfolio_params=dict(name="Hedge"))
ew = EqualWeighted(portfolio_params=dict(name="Equal Weight"))
iv = InverseVolatility(portfolio_params=dict(name="Inverse Vol"))

hedge.fit(X_train)
ew.fit(X_train)
iv.fit(X_train)

# %%
# Prediction
# ==========
# Predict on the test set and analyze the resulting portfolios.

ptf_hedge = hedge.predict(X_test)
ptf_ew = ew.predict(X_test)
ptf_iv = iv.predict(X_test)

population = Population([ptf_hedge, ptf_ew, ptf_iv])

# Plot cumulative returns
fig = population.plot_cumulative_returns()
show(fig)

# Display summaries
population.summary()
