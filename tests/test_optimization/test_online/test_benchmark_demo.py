"""Demo script showing the new BCRP capabilities.

This demonstrates how to use BCRP with different objective measures.
"""

import numpy as np

from skfolio.datasets import load_sp500_dataset
from skfolio.measures import PerfMeasure, RiskMeasure
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.optimization.online import BCRP
from skfolio.preprocessing import prices_to_returns


def demo_bcrp_objectives():
    """Demonstrate BCRP with different objectives."""
    print("=" * 70)
    print("BCRP Multi-Objective Demo")
    print("=" * 70)

    # Load data
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_subset = X.iloc[:100, :5]  # Use subset for speed

    print(f"\nData shape: {X_subset.shape}")
    print(f"Assets: {list(X_subset.columns)}\n")

    # 1. Log-wealth maximization (default, Kelly criterion)
    print("1. Log-Wealth Maximization (Kelly Criterion)")
    print("-" * 70)
    bcrp_log = BCRP()
    bcrp_log.fit(X_subset)
    print(f"Weights: {bcrp_log.weights_}")
    print(f"Sum: {np.sum(bcrp_log.weights_):.4f}\n")

    # 2. Variance minimization
    print("2. Variance Minimization")
    print("-" * 70)
    bcrp_var = BCRP(
        objective_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
    )
    bcrp_var.fit(X_subset)
    print(f"Weights: {bcrp_var.weights_}")
    print(f"Sum: {np.sum(bcrp_var.weights_):.4f}\n")

    # 3. CVaR minimization
    print("3. CVaR Minimization (95% confidence)")
    print("-" * 70)
    bcrp_cvar = BCRP(
        objective_measure=RiskMeasure.CVAR,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        cvar_beta=0.95,
    )
    bcrp_cvar.fit(X_subset)
    print(f"Weights: {bcrp_cvar.weights_}")
    print(f"Sum: {np.sum(bcrp_cvar.weights_):.4f}\n")

    # 4. Mean-variance utility
    print("4. Mean-Variance Utility (risk_aversion=2.0)")
    print("-" * 70)
    bcrp_utility = BCRP(
        objective_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_aversion=2.0,
    )
    bcrp_utility.fit(X_subset)
    print(f"Weights: {bcrp_utility.weights_}")
    print(f"Sum: {np.sum(bcrp_utility.weights_):.4f}\n")

    # 5. CDaR minimization
    print("5. CDaR Minimization (Conditional Drawdown at Risk)")
    print("-" * 70)
    bcrp_cdar = BCRP(
        objective_measure=RiskMeasure.CDAR,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        cdar_beta=0.95,
    )
    bcrp_cdar.fit(X_subset)
    print(f"Weights: {bcrp_cdar.weights_}")
    print(f"Sum: {np.sum(bcrp_cdar.weights_):.4f}\n")

    # Compare performance metrics
    print("=" * 70)
    print("Performance Comparison")
    print("=" * 70)

    relatives = 1.0 + X_subset.values
    cov = np.cov(X_subset.values.T)

    portfolios = {
        "Log-Wealth": bcrp_log.weights_,
        "Min-Variance": bcrp_var.weights_,
        "Min-CVaR": bcrp_cvar.weights_,
        "Utility": bcrp_utility.weights_,
        "Min-CDaR": bcrp_cdar.weights_,
    }

    for name, weights in portfolios.items():
        log_wealth = np.sum(np.log(relatives @ weights))
        variance = weights @ cov @ weights
        mean_return = np.mean(X_subset.values @ weights)

        print(f"\n{name}:")
        print(f"  Cumulative Log-Wealth: {log_wealth:.6f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Mean Return: {mean_return:.6f}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_bcrp_objectives()
