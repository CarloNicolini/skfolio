.. _online:

.. currentmodule:: skfolio.optimization.online

=============================
Online Portfolio Selection
=============================

Online Portfolio Selection (OPS) is a sequential decision-making framework where an investor must allocate wealth to a set of assets at each period without knowledge of future returns. Unlike traditional portfolio optimization, which solves a single static problem over historical data, OPS algorithms face a stream of price observations and must continuously rebalance to maximize cumulative wealth or minimize risk.

This module implements state-of-the-art online learning algorithms grounded in `Online Convex Optimization (OCO) <https://arxiv.org/abs/1912.13213>`_ theory. The framework unifies two competing paradigms:

- **Follow-The-Winner (FTW)**: Momentum-based strategies that exploit trending asset performance (e.g., Exponentiated Gradient, AdaGrad)
- **Follow-The-Loser (FTL)**: Mean-reversion strategies that bet on underperformers recovering (e.g., OLMAR, PAMR, CWMR)

All algorithms follow the sklearn estimator API and support flexible constraints, transaction costs, and custom objectives, in the same way they are supported in the MeanRisk estimator.
Important: The online learning algorithms are designed to work with gross returns or price relatives, which are the prices of the assets at the end of the period divided by the prices at the beginning of the period, in other words :math:`P_t/P_{t-1}`, differently from the standard net returns (or linear returns). Nonetheless all the estimators exposed in the skfolio online module accept net returns and convert to gross returns internally.

Another observation: in skfolio when one is using a convex optimization, one is implicitly impliying continuous rebalancing. In other words any strategy fitted on past data is called in online learning literature a "static" strategy or "hindsight" strategy as it would require maintaing the predicted, unique weights over the entire period of estimation and inference.
This is instead the basic assumption of the online learning framework, where the algorithm is expected to make decisions in a sequential manner, one at a time, and not to maintain a static portfolio over the entire period of estimation and inference, but to change them dynamically at each time step. This is why the online estimators have post-fit attribute :math:`all_weights_` that contains the weights at each time step.

**Key References:**

- Li, B., & Hoi, S. C. H. (2018). Online Portfolio Selection: A Survey. *ACM Computing Surveys*, 49(2), 1-36.
- Hazan, E. (2023). Introduction to Online Convex Optimization (2nd ed.). MIT Press.
- Orabona, F. (2020+). A Modern Introduction to Online Learning. MIT Press (draft).

Introduction
============

The Online Portfolio Selection Problem
---------------------------------------

At each time step :math:`t = 1, 2, \ldots, T`, an investor must:

1. Choose a portfolio weight vector :math:`\mathbf{b}_t \in \Delta^n` (sum to 1, non-negative)
2. Observe the gross return vector :math:`\mathbf{x}_t \in \mathbb{R}^n_+` (price relatives from yesterday's close to today's)
3. Realize log-wealth: :math:`\log(\mathbf{b}_t^\top \mathbf{x}_t)`

The cumulative log-wealth after :math:`T` periods is:

.. math::

    S_T = \sum_{t=1}^T \log(\mathbf{b}_t^\top \mathbf{x}_t)

The learner's performance is evaluated using **regret** against a comparator strategy:

.. math::

    \text{Regret}_T = \max_{\mathbf{u} \in \mathcal{U}} \sum_{t=1}^T \log(\mathbf{u}^\top \mathbf{x}_t) - \sum_{t=1}^T \log(\mathbf{b}_t^\top \mathbf{x}_t)

- **Static Regret**: :math:`\mathcal{U} = \Delta^n` (Best Constant Rebalanced Portfolio in hindsight)
- **Dynamic Regret**: :math:`\mathcal{U}` allows time-varying portfolios with controlled complexity

Theoretical Framework: Online Convex Optimization
---------------------------------------------------

The module unifies OPS algorithms under a single OCO framework. At each round, the algorithm solves:

.. math::

    \mathbf{w}_{t+1} = \arg\min_{\mathbf{w} \in \mathcal{K}} \left\langle \mathbf{g}_{1:t}, \mathbf{w} \right\rangle + \lambda R(\mathbf{w}) + C(\mathbf{w}, \mathbf{w}_{t-1})

where:

- :math:`\mathbf{g}_{1:t}` = cumulative loss gradients
- :math:`R(\mathbf{w})` = convex regularizer (KL divergence, squared norm, entropy barrier)
- :math:`C(\mathbf{w}, \mathbf{w}_{t-1})` = turnover/cost penalties
- :math:`\mathcal{K}` = feasible set (simplex, box bounds, groups, tracking error)

**Regularizer Choices Determine Algorithm:**

- KL divergence → Exponentiated Gradient (EG) / Entropic Mirror Descent
- Euclidean norm → Online Gradient Descent (OGD)
- Adaptive squared norm → AdaGrad (element-wise adaptive learning rates)
- Barrier + Euclidean + adaptive → Ada-BARRONS

**Key Theoretical Result:** Convexity of the loss determines regret rate:

- **Exp-concave losses** (e.g., :math:`-\log(\mathbf{b}^\top \mathbf{x})`): :math:`O(\log T)` regret
- **Strongly convex losses**: :math:`O(\log T)` regret
- **Convex losses** (e.g., hinge loss for mean reversion): :math:`O(\sqrt{T})` regret

Data Format: Net Returns to Gross Relatives
--------------------------------------------

The module expects **net returns** as input (e.g., arithmetic returns in :math:`[-1, +\infty)`), which are automatically converted to **gross relatives** (price ratios) for algorithm computations:

.. math::

    \mathbf{x}_t = 1 + \mathbf{r}_t

This ensures proper log-wealth calculations and simplifies gradient derivations. See :func:`~skfolio.preprocessing.prices_to_returns` for data preparation.

Algorithms Overview
====================

Follow-The-Winner (FTW): Momentum-Based Strategies
---------------------------------------------------

These algorithms exploit trend-following behavior by increasing allocations to recent outperformers.

**Exponentiated Gradient (EG)**
  - **Paradigm**: Follow-the-winner via entropy regularization
  - **Update**: :math:`\mathbf{b}_t \propto \mathbf{b}_{t-1} \odot \exp(-\eta \mathbf{g}_t)` (multiplicative)
  - **Regret**: :math:`O(\log T)` (optimal for exp-concave losses)
  - **Use when**: Markets trend strongly, portfolio turnover acceptable
  - **Drawback**: Can become concentration-risk heavy over time

**Online Gradient Descent (OGD)**
  - **Paradigm**: Follow-the-winner via Euclidean regularization
  - **Update**: :math:`\mathbf{b}_t = \text{Proj}_{\mathcal{K}}(\mathbf{b}_{t-1} - \eta \mathbf{g}_t)` (additive)
  - **Regret**: :math:`O(\sqrt{T})` generically, :math:`O(\log T)` for strongly convex
  - **Use when**: Diversification important, want to avoid concentration
  - **Advantage**: Works for any convex loss, not just exp-concave

**AdaGrad**
  - **Paradigm**: Follow-the-winner with adaptive per-asset learning rates
  - **Geometry**: :math:`H_i = \sqrt{\sum_{s=1}^t g_{s,i}^2}` (element-wise squared-gradient accumulation)
  - **Update**: :math:`b_{t,i} = \text{Proj}_{\Delta}(b_{t-1,i} - \eta g_{t,i} / H_i)`
  - **Regret**: :math:`O(\log T)` for strongly convex, universal dimension-free bounds
  - **Use when**: Asset volatilities vary dramatically, need acceleration on large-gradient assets

**Ada-BARRONS (Damped Online Newton Step for Portfolio Selection)**
  - **Paradigm**: Composite geometry combining entropy barrier, Euclidean, and adaptive quadratic terms
  - **Advantages**: Balances adaptivity, numerical stability, and concentration control
  - **Regret**: :math:`O(\log T)` with dimension-independent constants
  - **Use when**: Extreme market conditions, need state-of-the-art stability

**SWORD Variants (Stochastic Variance Reduction and Optimistic Online Learning)**
  - **Paradigm**: Meta-learning over multiple expert specialists
  - **Use when**: Uncertain which base strategy (EG, OGD, AdaGrad) will perform best
  - **Advantage**: Automatic expert mixing, can switch between paradigms in hindsight

**PROD (Soft-Bayes Prod)**
  - **Paradigm**: Multiplicative weights over a grid of constant expert strategies
  - **Use when**: Need probabilistic hedge over a pre-defined set of expert portfolios

**Optimistic Updates (Smooth Prediction)**
  - **Idea**: Predict next gradient as a smooth function of past gradients
  - **Benefit**: Faster adaptation to trending markets, :math:`O(\text{path-length})` adaptive regret
  - **Example**: :math:`\hat{g}_t = g_{t-1}` (assume smooth gradient evolution)

Follow-The-Loser (FTL): Mean-Reversion Strategies
---------------------------------------------------

These algorithms exploit mean reversion by betting against recent winners.

**OLMAR (Online Moving Average Reversion)**
  - **Idea**: Maintain a moving average of price relatives, bet on reverting to mean
  - **Predictors**:
    
    - **OLMAR-1 (SMA)**: Simple moving average of inverse cumulative products (window-based)
    - **OLMAR-2 (EWMA)**: Exponentially weighted moving average, recursive update
  
  - **Update**: Passive-aggressive margin constraint or mirror descent
  - **Regret**: :math:`O(\sqrt{T})`
  - **Use when**: Strong mean reversion signal, e.g., cryptocurrency, high-frequency trading
  - **Drawback**: Slow drift recovery in trending markets

**PAMR (Passive-Aggressive Mean Reversion)**
  - **Idea**: Enforce a margin constraint on current price relatives, be aggressive when violated
  - **Variants**:
    
    - **PAMR-0 (Simple)**: Lagrange multiplier set exactly to constraint violation
    - **PAMR-1 (Linear slack)**: Allow slack with linear penalty
    - **PAMR-2 (Quadratic slack)**: Allow slack with quadratic regularization
  
  - **Parameter**: Aggressiveness :math:`C` (larger = more aggressive reversion)
  - **Regret**: :math:`O(\sqrt{T})`
  - **Use when**: Explicit threshold-based reversion control needed

**CWMR (Confidence-Weighted Mean Reversion)**
  - **Idea**: Maintain Gaussian belief over portfolio weights, enforce probabilistic margin constraint
  - **Updates**: KL-proximal with second-order statistics (:math:`\mu`, :math:`\Sigma`)
  - **Parameter**: Confidence level :math:`\eta` (higher = tighter constraint)
  - **Regret**: :math:`O(\sqrt{T})`, but with tighter constants for smooth data
  - **Advantage**: Second-order information reduces variance, better for choppy markets
  - **Use when**: Have limited budget and need robust uncertainty quantification

**RMR (Robust Median Reversion)**
  - **Idea**: Use L1-median of recent price relatives instead of mean, robust to outliers
  - **Algorithm**: Weiszfeld's algorithm for L1-median computation
  - **Regret**: :math:`O(\sqrt{T})`, with improved constants under outlier presence
  - **Use when**: Data has significant outliers (flash crashes, gaps, data errors)

Benchmark Comparators
----------------------

To evaluate online algorithms, the module provides reference baselines for computing regret.

**Uniform Constant Rebalanced Portfolio (UCRP)**
  - **Weights**: :math:`\mathbf{b} = (1/n, \ldots, 1/n)`
  - **Purpose**: Naive diversification baseline
  - **Regret computation**: Often called "Buy-and-Hold"

**Constant Rebalanced Portfolio (CRP)**
  - **Weights**: Fixed user-specified or optimized
  - **Purpose**: Fixed-portfolio reference
  - **Use case**: Compare against manually tuned allocations

**Best Constant Rebalanced Portfolio (BCRP) in Hindsight**
  - **Optimization**: Solves a static convex problem over all :math:`T` periods
  - **Objectives**: Log-wealth (Kelly criterion), variance minimization, CVaR, etc.
  - **Purpose**: Strongest baseline for static regret (lower bound on achievable performance)
  - **Regret formula**: :math:`\text{Regret}_T = \sum_t \log(u^\top x_t) - \sum_t \log(b_t^\top x_t)`

**BestStock**
  - **Strategy**: Allocate 100% to the single best-performing asset
  - **Purpose**: Detect if market has a clear trend (best-stock wins easily)
  - **Use case**: Diagnostic for market regime detection

Regret Analysis
---------------

The module provides comprehensive regret computation via :func:`regret`:

.. math::

    \text{Regret}_T(b) = \sum_{t=1}^T \log(\mathbf{u}^\top \mathbf{x}_t) - \sum_{t=1}^T \log(\mathbf{b}_t^\top \mathbf{x}_t)

**Regret Types**:

- **Static**: Compare to fixed best portfolio in hindsight
- **Dynamic**: Compare to time-varying portfolios with bounded path length (e.g., sparse portfolio swaps)
- **Dynamic Worst-Case**: Compare to per-round optimal (most stringent, typically :math:`> 0`)
- **Dynamic Universal**: Compare to time-varying with explicit budget on cumulative drift

Getting Started
================

Basic Example: Follow-The-Winner (Momentum)
--------------------------------------------

.. code-block:: python

    from skfolio.datasets import load_sp500_relatives_dataset
    from skfolio.optimization.online import FollowTheWinner, FTWStrategy, BCRP, regret, RegretType
    import numpy as np

    # Load data (net returns)
    X = load_sp500_relatives_dataset(net_returns=True)
    print(f"Data shape: {X.shape}")  # (T, n) = (periods, assets)

    # 1. Train a momentum (EG) strategy
    model = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate="auto")
    model.fit(X)
    print(f"Final weights shape: {model.weights_.shape}")
    print(f"Final weights (top 5): {np.sort(model.weights_)[-5:]}")

    # 2. Evaluate regret against BCRP
    comparator = BCRP()  # Best Constant Rebalanced Portfolio
    regret_array = regret(model, X, comparator=comparator, regret_type=RegretType.STATIC)
    print(f"Final static regret: {regret_array[-1]:.4f} nats")

    # 3. Inspect wealth trajectory
    print(f"Initial wealth: {model.all_wealth_[0]:.2f}")
    print(f"Final wealth: {model.wealth_:.2f}")
    print(f"Cumulative return: {model.wealth_ / model.all_wealth_[0] - 1:.2%}")

Mean Reversion Example: PAMR
-----------------------------

.. code-block:: python

    from skfolio.optimization.online import FollowTheLoser, FTLStrategy, BCRP
    from skfolio.measures import PerfMeasure

    # 1. Train a mean-reversion strategy
    model = FollowTheLoser(
        strategy=FTLStrategy.PAMR,
        pamr_variant="slack_quadratic",  # More stable PAMR-2
        pamr_C=500.0,  # Aggressiveness parameter
        epsilon=1.0,   # Margin threshold
        update_mode="pa"  # Passive-Aggressive (original algorithm)
    )
    model.fit(X)

    # 2. Compare weights evolution
    print(f"Weights shape over time: {model.all_weights_.shape}")  # (T, n)
    print(f"Average turnover per period: {np.mean(np.abs(np.diff(model.all_weights_, axis=0)), axis=1).mean():.4f}")

    # 3. Analyze portfolio metrics
    portfolio = model.predict(X)
    print(f"Annualized Sharpe ratio: {portfolio.annualized_sharpe_ratio:.3f}")
    print(f"Max drawdown: {portfolio.max_drawdown:.2%}")

Comparing Strategies
---------------------

.. code-block:: python

    from skfolio.optimization.online import (
        FollowTheWinner, FollowTheLoser, FTWStrategy, FTLStrategy,
        BCRP, CRP, UCRP, regret, RegretType
    )
    import pandas as pd

    # Define strategies to compare
    strategies = {
        "EG": FollowTheWinner(strategy=FTWStrategy.EG),
        "OGD": FollowTheWinner(strategy=FTWStrategy.OGD),
        "AdaGrad": FollowTheWinner(strategy=FTWStrategy.ADAGRAD),
        "OLMAR-1": FollowTheLoser(strategy=FTLStrategy.OLMAR, olmar_predictor="sma"),
        "PAMR": FollowTheLoser(strategy=FTLStrategy.PAMR, update_mode="pa"),
        "CWMR": FollowTheLoser(strategy=FTLStrategy.CWMR, update_mode="pa"),
    }

    # Fit each strategy
    comparator = BCRP()
    results = {}

    for name, model in strategies.items():
        model.fit(X)
        regrets = regret(model, X, comparator=comparator, regret_type=RegretType.STATIC)
        results[name] = {
            "final_wealth": model.wealth_,
            "cumulative_return": model.wealth_ / model.all_wealth_[0] - 1,
            "final_regret": regrets[-1],
            "max_drawdown": model.predict(X).max_drawdown,
        }

    df = pd.DataFrame(results).T
    print(df)

Advanced Features
=================

Transaction Costs and Turnover Control
---------------------------------------

Real-world trading incurs costs. The module supports:

- **Proportional transaction costs**: Fixed fee per unit of traded volume
- **Turnover constraints**: Cap the L1 distance between consecutive weight vectors
- **Adaptive fee scaling**: Apply fees only to mean-reversion predictors

.. code-block:: python

    model = FollowTheWinner(
        strategy=FTWStrategy.EG,
        transaction_costs=0.001,  # 10 bps per trade
        max_turnover=0.1,         # Max 10% turnover per period
    )
    model.fit(X)

    # Compare wealth with/without costs
    model_free = FollowTheWinner(strategy=FTWStrategy.EG)
    model_free.fit(X)
    print(f"Wealth (with costs): {model.wealth_:.2f}")
    print(f"Wealth (without costs): {model_free.wealth_:.2f}")
    print(f"Cost drag: {(model_free.wealth_ - model.wealth_) / model_free.wealth_:.2%}")

Multiple Objectives and Risk Measures
--------------------------------------

The module supports optimizing various objectives:

.. code-block:: python

    from skfolio.measures import RiskMeasure, PerfMeasure

    # Risk minimization (CVaR)
    model_risk = FollowTheWinner(
        strategy=FTWStrategy.OGD,
        objective=RiskMeasure.CVAR,
        learning_rate=0.1,
    )
    model_risk.fit(X)
    
    # Log-wealth maximization (default, most theoretical support)
    model_wealth = FollowTheWinner(
        strategy=FTWStrategy.EG,
        objective=PerfMeasure.LOG_WEALTH,
    )
    model_wealth.fit(X)

Constraints: Box, Budget, Groups, Tracking Error
-------------------------------------------------

Like the convex optimization module, online methods support rich constraints:

.. code-block:: python

    model = FollowTheWinner(
        strategy=FTWStrategy.ADAGRAD,
        min_weights=0.0,           # Long-only
        max_weights=0.5,           # No position > 50%
        budget=1.0,                # Fully invested
        groups={"tech": ["AAPL", "MSFT"], "energy": ["XLE"]},
        linear_constraints=["tech >= 0.3", "energy <= 0.2"],
    )

Warm Start and Sequential Fitting
----------------------------------

For online learning, data often arrives incrementally. Use :meth:`partial_fit`:

.. code-block:: python

    model = FollowTheWinner(strategy=FTWStrategy.EG, warm_start=True)

    # Fit incrementally as new data arrives
    for t in range(len(X)):
        X_new = X.iloc[[t], :]
        model.partial_fit(X_new)

    # Or fit all at once (auto-vectorized):
    model.fit(X)

    # Next batch (warm_start=True preserves state)
    X_new_batch = load_new_data()
    model.partial_fit(X_new_batch)

Learning Rate Schedules and Auto-Tuning
----------------------------------------

Learning rates control convergence speed. The module offers:

- **Constant**: :math:`\eta_t = \eta` (simplest, requires manual tuning)
- **Scheduled**: :math:`\eta_t = \eta / \sqrt{t}` (standard OCO decay)
- **Callables**: :math:`\eta_t = f(t)` (custom schedules)
- **Auto**: Automatically tuned based on strategy and data properties

.. code-block:: python

    # Manual constant
    model1 = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.5)

    # Manual schedule (callable)
    def my_schedule(t):
        return 1.0 / np.sqrt(t + 1)
    model2 = FollowTheWinner(strategy=FTWStrategy.OGD, learning_rate=my_schedule)

    # Auto-tuned (recommended for new users)
    model3 = FollowTheWinner(strategy=FTWStrategy.ADAGRAD, learning_rate="auto")
    model3.fit(X)

Mirror Maps and Geometry
-------------------------

The underlying optimization geometry is controlled by mirror maps:

.. code-block:: python

    from skfolio.optimization.online._mirror_maps import (
        EuclideanMirrorMap, EntropyMirrorMap, BurgMirrorMap, AdaptiveMahalanobisMap
    )

    # Entropy geometry (EG-style multiplicative updates)
    model_eg = FollowTheWinner(
        strategy=FTWStrategy.EG,  # Uses entropy mirror map internally
    )

    # Euclidean geometry (OGD-style additive updates)
    model_ogd = FollowTheWinner(
        strategy=FTWStrategy.OGD,  # Uses Euclidean mirror map internally
    )

    # Adaptive Mahalanobis geometry (AdaGrad-style)
    model_ada = FollowTheWinner(
        strategy=FTWStrategy.ADAGRAD,  # Element-wise adaptive geometry
    )

Optimistic Predictions for Smooth Markets
-------------------------------------------

When gradients vary slowly (smooth environments), optimistic updates can achieve faster convergence:

.. code-block:: python

    model = FollowTheWinner(
        strategy=FTWStrategy.EG,
        grad_predictor="smooth",  # Predict next gradient as smoothly varying
        smooth_epsilon=1.0,       # Smoothness strength (larger = assume smoother)
    )
    model.fit(X)

Practical Guidance
==================

Which Strategy Should I Use?
----------------------------

**When markets are trending (momentum dominates):**
  - Use Follow-The-Winner: EG or AdaGrad
  - Higher expected log-wealth growth
  - Watch for trend reversals; consider dynamic regret for market changes

**When markets mean-revert (reversals dominate):**
  - Use Follow-The-Loser: OLMAR, PAMR, or CWMR
  - Exploit short-term price overshoots
  - Choose CWMR for markets with high outlier probability (flash crashes)

**When uncertain about market regime:**
  - Use SWORD (meta-learner over experts)
  - Or compare all strategies on historical data and backtest
  - Adaptive methods (AdaGrad, Ada-BARRONS) often robust across regimes

**When transaction costs are high (>50 bps):**
  - High-turnover strategies (FTW, PAMR) may be unprofitable
  - Use low-turnover methods with turnover constraints
  - Consider CRP (fixed allocation) as strong competitor

**For crypto, commodities, or thin-traded assets:**
  - Mean reversion often stronger due to lack of fundamental anchoring
  - CWMR or RMR recommended for outlier resistance
  - Use management fees parameter to model slippage

**For equity indices (S&P 500, etc.):**
  - Weak mean reversion, slight momentum trending
  - EG and AdaGrad often competitive
  - BCRP strong baseline; hard to beat without excess transaction costs

Parameter Tuning Heuristics
----------------------------

- **Learning rate** (:math:`\eta`):
  - Start with "auto" (recommended)
  - If want faster adaptation: increase :math:`\eta`
  - If concerned about overfitting: decrease :math:`\eta`
  - Typical range: 0.01 to 10.0

- **Mean-reversion epsilon** (:math:`\varepsilon`):
  - Larger → fewer active trades (more conservative)
  - Smaller → more active mean reversion
  - Typical range: 0.5 to 2.0
  - Default: 1.0 (exploit margin up to 100% underperformance)

- **PAMR aggressiveness** (:math:`C`):
  - Larger → stronger reversion push
  - Smaller → weaker reversion (closer to EG)
  - Default: 500.0 (quite aggressive)
  - Try 100-1000 depending on reversion strength

- **CWMR confidence** (:math:`\eta`):
  - Larger (closer to 1.0) → tighter probabilistic bounds, stronger reversion
  - Smaller (closer to 0.5) → looser bounds, milder reversion
  - Default: 0.95 (95% confidence level)

- **OLMAR window** (OLMAR-1):
  - Larger → smoother reversion predictor, slower adaptation
  - Smaller → noisier predictor, faster changes
  - Typical: 3-10 (default: 5)

- **OLMAR alpha** (OLMAR-2):
  - Closer to 1.0 → weight recent observations more
  - Closer to 0.0 → give equal weight to all past
  - Default: 0.5 (balanced)

Complete Workflow Example
---------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from skfolio.datasets import load_sp500_relatives_dataset
    from skfolio.optimization.online import (
        FollowTheWinner, FollowTheLoser, FTWStrategy, FTLStrategy,
        BCRP, UCRP, regret, RegretType
    )
    from skfolio.measures import RiskMeasure

    # 1. Load and split data
    X = load_sp500_relatives_dataset(net_returns=True)
    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]

    # 2. Define strategies
    strategies = {
        "EG": FollowTheWinner(strategy=FTWStrategy.EG, learning_rate="auto"),
        "AdaGrad": FollowTheWinner(strategy=FTWStrategy.ADAGRAD, learning_rate="auto"),
        "OLMAR": FollowTheLoser(strategy=FTLStrategy.OLMAR, update_mode="pa"),
        "PAMR": FollowTheLoser(strategy=FTLStrategy.PAMR, update_mode="pa"),
    }

    # 3. Train on train set
    for name, model in strategies.items():
        model.fit(X_train)
        print(f"{name} trained, final weight concentration: {model.weights_.max():.2%}")

    # 4. Evaluate on test set with regret
    comparator = BCRP()
    comparator.fit(X_test)

    results = []
    for name, model in strategies.items():
        regrets = regret(model, X_test, comparator=comparator, regret_type=RegretType.STATIC)
        port = model.predict(X_test)
        results.append({
            "Strategy": name,
            "Cumulative Return (%)": 100 * (port.wealth[-1] - 1),
            "Annualized Sharpe": port.annualized_sharpe_ratio,
            "Max Drawdown (%)": 100 * port.max_drawdown,
            "Final Regret (nats)": regrets[-1],
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # 5. Deep dive: Analyze best performer
    best_strategy = strategies["EG"]  # Example
    print(f"\nBest strategy 'EG':")
    print(f"  - All weights shape: {best_strategy.all_weights_.shape}")
    print(f"  - Weight std across time: {best_strategy.all_weights_.std(axis=0).mean():.4f}")
    print(f"  - Average position concentration: {best_strategy.all_weights_.max(axis=1).mean():.2%}")

References and Further Reading
===============================

**Core Theory**

- Hazan, E. (2023). *Introduction to Online Convex Optimization* (2nd ed.). MIT Press.
  `[PDF available at elad.hazan.org] <https://www.cs.princeton.edu/~ehazan/>`_

- Orabona, F. (2020+). *A Modern Introduction to Online Learning* (2nd ed.). MIT Press.
  `[Draft available at francescorabona.eu] <https://francescorabona.eu/>`_

**OPS Surveys and Foundations**

- Li, B., & Hoi, S. C. H. (2018). Online Portfolio Selection: A Survey.
  *ACM Computing Surveys*, 49(2), 1-36.
  `[arXiv:1212.2129] <https://arxiv.org/abs/1212.2129>`_

- Li, B., Hoi, S. C. H., Zhao, P., & Gopalkrishnan, V. (2012). Confidence Weighted
  Mean Reversion Strategy for Online Portfolio Selection.
  *In Proceedings of AISTATS*.

**Specific Algorithms**

- Li, B., Zhao, P., Hoi, S. C. H., & Gopalkrishnan, V. (2012). PAMR: Passive Aggressive
  Mean Reversion strategy for portfolio selection. *Machine Learning*, 87(2), 221-258.

- Huang, D., Zhou, J., Li, B., Hoi, S. C. H., & Zhou, S. (2013). Robust Median Reversion
  Strategy for Online Portfolio Selection. *In Proceedings of IJCAI*.

- Orseau, L., Lattimore, T., & Legg, S. (2017). Soft-Bayes: Product Rule for Mixtures
  of Experts with Log-Loss. *In Algorithmic Learning Theory (PMLR 76)*, 73-90.

**OCO and Mirror Descent**

- McMahan, H. B. (2011). Follow-the-Regularized-Leader and Mirror Descent: Equivalence
  Theorems and L1 Regularization. *In Proceedings of AISTATS*.

- Chiang, C. K., Yang, T., Lee, C. J., Mahdavi, M., & Zhu, C. J. (2012). Online
  Optimization with Gradual Variations. *In Proceedings of COLT*.

**Regret Analysis**

- Awerbuch, B., & Kleinberg, R. M. (2008). Adaptive Routing with End-to-End
  Feedback: Distributed Learning and Geometric Approaches.
  *In Proceedings of STOC*, 45-53.

**Benchmarking Datasets**

- OLPS Codebase: `http://olps.stevenhoi.org <http://olps.stevenhoi.org>`_
  Includes canonical datasets: NYSE-O, TSE, DJIA, MSCI, S&P 500, CMC20.

See Also
========

- :ref:`convex` — Batch (offline) portfolio optimization
- :class:`~skfolio.optimization.RiskBudgeting` — Risk parity methods
- :class:`~skfolio.optimization.HierarchicalRiskParity` — Clustering-based hierarchical allocation
- :func:`~skfolio.optimization.online.regret` — Compute regret curves
