"""
===================
Universal Portfolio
===================

This example demonstrates Thomas Cover's Universal Portfolio algorithm using
the :class:`~skfolio.optimization.UniversalPortfolio` estimator.

We show:
- Single fit/predict over a holdout set
- Different expert weighting schemes (wealth, uniform, top_k)
- Walk-forward evaluation (multi-period portfolio)
"""

# %%
# Data
# ====
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population
from skfolio.datasets import load_sp500_dataset, load_ftse100_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import EqualWeighted, Universal
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset().loc["2018":]
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Universal Portfolio (single period, wealth weighting)
up_wealth = Universal(
    n_samples=50_000, random_state=0, weight_scheme="wealth", save_experts=True
)
up_wealth.fit(X_train)
ptf_wealth = up_wealth.predict(X_test)
ptf_wealth.name = "Universal Portfolio (wealth)"
print("Weights (wealth):", ptf_wealth.weights.round(3))

# %%
# Uniform expert weighting
up_uniform = Universal(
    n_samples=50_000, random_state=0, weight_scheme="uniform", save_experts=True
)
up_uniform.fit(X_train)
ptf_uniform = up_uniform.predict(X_test)
ptf_uniform.name = "Universal Portfolio (uniform)"
print("Weights (uniform):", ptf_uniform.weights.round(3))

# %%
# Top-k expert weighting
up_topk = Universal(
    n_samples=50_000, random_state=0, weight_scheme="top_k", top_k=5, save_experts=True
)
up_topk.fit(X_train)
ptf_topk = up_topk.predict(X_test)
ptf_topk.name = "Universal Portfolio (top_k)"
print("Weights (top_k):", ptf_topk.weights.round(3))

# %%
# Benchmark: Equal-Weighted
benchmark = EqualWeighted()
benchmark.fit(X_train)
ptf_bench = benchmark.predict(X_test)

pop = Population([ptf_wealth, ptf_uniform, ptf_topk, ptf_bench])
print(pop.summary())
fig = pop.plot_cumulative_returns()
show(fig)

# %%
# Walk-forward evaluation
cv = WalkForward(train_size=252, test_size=60)
ptf_wf = cross_val_predict(Universal(n_samples=10_000, random_state=0), X, cv=cv)
ptf_wf.summary()
