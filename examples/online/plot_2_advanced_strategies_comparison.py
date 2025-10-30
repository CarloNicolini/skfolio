"""
===================================
Online Portfolio Selection Advanced
===================================

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
from skfolio.measures import RiskMeasure
from skfolio.optimization.convex import MeanRisk, ObjectiveFunction
from skfolio.optimization.online import BCRP, UCRP, FTWStrategy, FollowTheWinner
from skfolio.population import Population
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)

# The following code creates and compares various advanced online portfolio selection strategies:
# - BCRP (Best Constant Rebalanced Portfolio): An oracle strategy that finds the constant-weight portfolio in hindsight that would have achieved the best cumulative return, here minimizing Conditional Value at Risk (CVaR).
# - UCRP (Uniform Constant Rebalanced Portfolio): A naive approach that rebalances constantly to equal weights.
# - FollowTheWinner (with Exponential Gradient): An efficient, adaptive algorithm that updates allocations using the Exponential Gradient (EG) optimization, emphasizing assets with recent strong performance.
# - FollowTheWinner (with AdaGrad): A variation using the AdaGrad algorithm, which adapts each asset's learning rate individually based on historical gradient information, offering robustness to different market conditions.
# These methods are assembled into a Population object for cumulative return comparison and visualization.

# Here we show that it's possible to achieve a higher CVaR ratio than the BCRP strategy by using the AdaGrad algorithm.

population = []
population.append(
    BCRP(
        objective_measure=RiskMeasure.CVAR,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        portfolio_params={"name": "Min CVaR in Hindsight"},
    ).fit_predict(X)
)
population.append(
    UCRP(
        portfolio_params={"name": "Uniform Constant Rebalanced Portfolio"}
    ).fit_predict(X)
)
population.append(
    FollowTheWinner(
        strategy=FTWStrategy.EG,
        objective=RiskMeasure.CVAR,
        portfolio_params={"name": "Exponential Gradient"},
        learning_rate="auto",
        learning_rate_scale="empirical",
    ).fit_predict(X)
)

population.append(
    FollowTheWinner(
        strategy=FTWStrategy.ADAGRAD,
        objective=RiskMeasure.CVAR,
        portfolio_params={"name": "AdaGrad"},
        learning_rate="auto",
        learning_rate_scale="theory",
    ).fit_predict(X)
)

population = Population(population)
population.set_portfolio_params(compounded=True)
fig = population.plot_cumulative_returns(log_scale=True)
show(fig)

# %%
# We print the summary of the population:
print(population.summary())

# %%
# We plot the composition of the population:
population.plot_composition(display_sub_ptf_name=False)
