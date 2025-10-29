.. _online_examples:

Online Portfolio Selection Examples
====================================

This section demonstrates Online Portfolio Selection (OPS) strategies implemented
in the `skfolio.optimization.online` module.

**What is Online Portfolio Selection?**

Unlike traditional batch optimization that assumes stationary returns and computes
weights once, Online Portfolio Selection updates weights sequentially after observing
each period's returns. This online learning framework provides:

- **Adaptivity**: Responds to non-stationary market conditions
- **Efficiency**: Fast updates suitable for high-frequency rebalancing
- **Theoretical guarantees**: Sublinear regret bounds via Online Convex Optimization

**Strategy Families**

1. **Follow-the-Winner (Momentum)**
   Transfer wealth to recently successful assets:
   
   - OGD (Online Gradient Descent): Euclidean geometry
   - EG (Exponential Gradient): Entropy mirror map, multiplicative weights
   - AdaGrad: Per-coordinate adaptive learning rates
   - AdaBARRONS: Adaptive barrier with Mahalanobis preconditioning
   - SWORD: Optimistic methods with gradient prediction
   - PROD: Soft-Bayes mixture of experts

2. **Follow-the-Loser (Mean Reversion)**
   Shift wealth from winners to losers, betting on reversals:
   
   - OLMAR (Online Moving Average Reversion): Multi-period MA signals
   - PAMR (Passive-Aggressive Mean Reversion): Single-period reversals
   - CWMR (Confidence-Weighted): Distributional updates with uncertainty
   - RMR (Robust Median Reversion): L1-median, outlier-robust

3. **Benchmarks**
   
   - UCRP (Uniform CRP): Equal-weighted rebalancing (1/n)
   - BCRP (Best CRP): Hindsight-optimal constant portfolio
   - BestStock: All wealth in single best performer

**Regret Analysis**

Regret measures cumulative loss difference between online algorithm and comparator:

- **Static Regret**: vs best constant portfolio (BCRP)
- **Dynamic Regret**: vs time-varying optimal sequence
- **Universal Dynamic**: vs path-length-bounded sequence

Theory guarantees O(√T) regret for convex losses, O(log T) for exp-concave losses.

**Key Features**

- **Automatic Learning Rates**: `learning_rate="auto"` with empirical tuning
- **Transaction Costs**: Proportional costs, drift-aware turnover modeling
- **Management Fees**: Per-period fees affecting net returns
- **Rich Constraints**: Box, budget, groups, linear, turnover, tracking error
- **Projection Methods**: Fast simplex projection or convex fallback (cvxpy)

**References**

.. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
       arXiv:1212.2129.
.. [2] Hazan, E. (2016). Introduction to Online Convex Optimization.
       Foundations and Trends in Optimization, 2(3-4), 157-325.
.. [3] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods
       for Online Learning and Stochastic Optimization. JMLR, 12, 2121-2159.

