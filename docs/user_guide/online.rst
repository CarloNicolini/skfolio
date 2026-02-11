.. _online:

.. currentmodule:: skfolio.optimization.online

=================================
Online Portfolio Selection (OLPS)
=================================

Traditional portfolio optimization methods (Mean-Variance, Risk Parity, HRP, etc.)
are **offline**: they process an entire historical window at once, estimate moments,
solve a convex program, and output a fixed allocation. Whenever new data arrives the
model must be completely re-fitted.

**Online Portfolio Selection** takes a fundamentally different approach. The investor
chooses portfolio weights *before* observing each period's returns, then updates the
weights incrementally as new data streams in. No distributional assumption is made
about the market -- the framework provides rigorous **regret** guarantees that hold
even against an adversarial sequence of returns.

This module implements the major families of Online Portfolio Selection algorithms
from the Online Convex Optimization (OCO) literature, using the same scikit-learn
``fit`` / ``partial_fit`` API as the rest of skfolio.

.. note::

   **Offline vs Online -- when to use which?**

   * Use :ref:`offline optimization <optimization>` (``MeanRisk``, ``RiskBudgeting``,
     etc.) when you have a representative training set and want to solve a
     single-period or multi-period allocation with rich risk constraints.
   * Use **online optimization** (``FollowTheWinner``, ``FollowTheLoser``) when you
     need a streaming, assumption-free rebalancing policy with formal regret
     guarantees -- for example in high-frequency rebalancing, live trading systems,
     or as robust baselines for backtesting studies.


Quick Start
***********

Every online estimator follows the standard skfolio API: call ``fit(X)`` with a
matrix of **net returns** (not prices). The estimator internally converts to gross
relatives, sequentially updates weights, and exposes the final weights in
``weights_`` and the full trajectory in ``all_weights_``.

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization.online import FollowTheWinner
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = FollowTheWinner(strategy="eg")
    model.fit(X)
    print(model.weights_)

    portfolio = model.predict(X)
    print(portfolio.annualized_sharpe_ratio)

For incremental (streaming) usage, use ``partial_fit`` one row at a time:

.. code-block:: python

    model = FollowTheWinner(strategy="eg", warm_start=True)
    for t in range(len(X)):
        model.partial_fit(X.iloc[[t]])
    print(model.weights_)


Benchmarks
**********

The module provides hindsight benchmarks commonly used in the OLPS literature for
performance comparison and regret calculation:

    * :class:`CRP` -- Constant Rebalanced Portfolio (fixed user-supplied weights)
    * :class:`UCRP` -- Uniform CRP (equal-weighted, also known as :math:`1/n`)
    * :class:`BestStock` -- Best single asset in hindsight
    * :class:`BCRP` -- Best Constant Rebalanced Portfolio in hindsight

The :class:`BCRP` is the standard comparator for **static regret**: it finds the single fixed allocation that would have maximized log-wealth over the entire history.
Any online algorithm with sublinear regret is guaranteed to approach BCRP performance as the horizon grows.

**Example:**

.. code-block:: python

    from skfolio.optimization.online import BCRP, UCRP

    ucrp = UCRP()
    ucrp.fit(X)

    bcrp = BCRP()
    bcrp.fit(X)

    print(f"UCRP final wealth: {ucrp.wealth_:.4f}")
    print(f"BCRP final wealth: {bcrp.wealth_:.4f}")


Follow-the-Winner Strategies
*****************************

:class:`FollowTheWinner` implements a unified engine for **first-order Online Convex
Optimization** algorithms. These methods allocate more to assets that have recently
performed well by following the gradient of the log-wealth objective.

The family is parameterized by a ``strategy``:

    * ``"eg"`` -- **Exponentiated Gradient** (entropy mirror map, multiplicative updates)
    * ``"ogd"`` -- **Online Gradient Descent** (Euclidean mirror map)
    * ``"prod"`` -- **PROD / Soft-Bayes** (Burg log-barrier mirror map)
    * ``"adagrad"`` -- **AdaGrad** (adaptive diagonal preconditioning)
    * ``"adabarrons"`` -- **Ada-BARRONS** (adaptive barrier + Online Newton Step)
    * ``"sword_var"`` -- **SWORD-Var** (variation-adaptive, dynamic regret)
    * ``"sword_small"`` -- **SWORD-Small** (small-loss adaptive)
    * ``"sword_best"`` -- **SWORD-Best** (meta-aggregation of SWORD experts)
    * ``"sword_pp"`` -- **SWORD++** (meta with EG expert)

Under the hood, the engine supports both **Online Mirror Descent (OMD)** and
**Follow-the-Regularized-Leader (FTRL)** update modes, configurable through
``update_mode``.

**Example -- Exponentiated Gradient:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization.online import FollowTheWinner
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = FollowTheWinner(
        strategy="eg",
        learning_rate="auto",
    )
    model.fit(X)
    print(model.weights_)

**Example -- AdaGrad with optimistic predictions:**

Optimistic OMD uses a predictor for the next gradient. When gradients are temporally
smooth (common in financial data), this can significantly reduce regret.

.. code-block:: python

    model = FollowTheWinner(
        strategy="adagrad",
        grad_predictor="last",   # predict next gradient = last gradient
    )
    model.fit(X)
    print(model.weights_)

Learning Rates
==============

When ``learning_rate="auto"``, the module selects a learning rate based on:

    * **Strategy geometry** (entropy for EG, Euclidean for OGD, adaptive for AdaGrad)
    * **Domain diameter** (computed from box and budget constraints)
    * **Gradient bound** (estimated from the objective's convexity class)

The ``learning_rate_scale`` parameter controls the aggressiveness:

    * ``"theory"`` (default) -- Worst-case OCO rate :math:`\sqrt{\log n / (t+1)}`
      for EG.  Safe on all datasets, never more than 1% worse than UCRP.
    * ``"moderate"`` -- :math:`2\sqrt{2}` boost (Hazan's book constant).
    * ``"empirical"`` -- :math:`\sqrt{2}` boost over theory.  All three scales
      preserve the same :math:`O(\sqrt{T \log n})` regret guarantee.

Gradient Enhancements
=====================

Two optional parameters can improve FTW performance on real financial data:

**Discounted FTRL** (``discount``): replaces the cumulative gradient sum with an
exponentially decaying sum :math:`G_t = \gamma G_{t-1} + g_t`, giving recent
observations more influence. Useful for non-stationary markets.

**Momentum lookback** (``gradient_lookback``): averages the last *W* gradients
before passing them to the OCO engine, capturing medium-term momentum signals
(Jegadeesh & Titman, 1993).

.. code-block:: python

    # Momentum-enhanced EG with 60-day lookback
    model = FollowTheWinner(
        strategy="eg",
        gradient_lookback=60,
    )
    model.fit(X)

.. note::

   ``gradient_lookback`` in [60, 120] consistently turns EG from a UCRP-equivalent
   into a modest momentum strategy on real datasets.  ``discount`` on its own
   compresses weights toward 1/n; combine it with a constant ``learning_rate``
   for best results.


Supported Constraints
=====================

All online estimators support the same rich constraint set as offline methods:

    * Weight Bounds (``min_weights``, ``max_weights``)
    * Budget Constraint (``budget``)
    * Turnover Constraint (``max_turnover``)
    * Group Constraints (``groups``, ``linear_constraints``)
    * Transaction Costs (``transaction_costs``)
    * Management Fees (``management_fees``)

Constraints are enforced at every rebalancing via projection. Simple constraints
(box + budget + turnover) use a fast bisection projector; complex constraints
(groups, linear, variance bounds) automatically fall back to a ``cvxpy`` solver.


Follow-the-Loser Strategies (Mean Reversion)
*********************************************

:class:`FollowTheLoser` implements **mean-reversion** strategies that bet on
short-term price reversals. These methods move wealth from recent winners to recent
losers, exploiting the empirical observation that asset prices often revert toward
their moving averages.

The family is parameterized by a ``strategy``:

    * ``"olmar"`` -- **Online Moving Average Reversion**. Uses a moving-average
      predictor to estimate next-period relatives and tilts the portfolio toward
      undervalued assets with a passive-aggressive update.
    * ``"pamr"`` -- **Passive-Aggressive Mean Reversion**. Enforces a margin
      constraint :math:`w^\top x_t \leq \varepsilon` and finds the minimum-change
      portfolio that satisfies it.
    * ``"cwmr"`` -- **Confidence-Weighted Mean Reversion**. Maintains a Gaussian
      distribution over weights and updates it via a KL-proximal step under a
      probabilistic margin constraint.
    * ``"rmr"`` -- **Robust Median Reversion**. Like OLMAR but uses the L1-median
      (geometric median) for outlier-robust price prediction.

Each strategy supports two ``update_mode`` values:

    * ``"pa"`` -- The original **passive-aggressive** closed-form update from
      Li and Hoi (2012). Reproduces the OLPS reference implementation.
    * ``"md"`` -- A modern **mirror descent** formulation using surrogate convex
      losses (hinge, squared hinge, or softplus) and a configurable mirror map.

**Example -- OLMAR with SMA predictor:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization.online import FollowTheLoser
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=5,
        epsilon=10.0,
    )
    model.fit(X)
    print(model.weights_)

**Example -- PAMR with slack:**

.. code-block:: python

    model = FollowTheLoser(
        strategy="pamr",
        pamr_variant="slack_quadratic",
        pamr_C=500.0,
        epsilon=0.5,
    )
    model.fit(X)
    print(model.weights_)

**Example -- CWMR (second-order, distributional):**

CWMR maintains a full covariance belief over portfolio weights and provides
confidence-weighted updates -- a natural second-order extension of PAMR.

.. code-block:: python

    model = FollowTheLoser(
        strategy="cwmr",
        cwmr_eta=0.95,        # confidence level
        cwmr_sigma0=1.0,      # initial variance
        epsilon=1.0,
    )
    model.fit(X)
    print(model.weights_)


Regret Analysis
***************

The :func:`regret` function computes **regret curves** that measure how an online
strategy compares to a benchmark over time. Regret is the core performance metric
in Online Convex Optimization:

.. math::

   R_T = \sum_{t=1}^{T} \ell_t(w_t) - \sum_{t=1}^{T} \ell_t(w^*)

where :math:`\ell_t(w) = -\log(w^\top x_t)` is the per-period log-wealth loss and
:math:`w^*` is the comparator (e.g., the best fixed portfolio in hindsight).

Sublinear regret (:math:`R_T / T \to 0`) means the online strategy asymptotically
matches the comparator's performance.

The module supports several regret types:

    * ``RegretType.STATIC`` -- Against the best fixed portfolio (BCRP). This is the
      standard regret notion in universal portfolio theory.
    * ``RegretType.DYNAMIC_UNIVERSAL`` -- Against the best *changing* portfolio
      sequence with bounded path length. Captures non-stationarity.
    * ``RegretType.DYNAMIC_WORST_CASE`` -- Against the per-round best asset. The
      strongest (and hardest to beat) benchmark.

**Example -- static regret curve:**

.. code-block:: python

    from skfolio.optimization.online import FollowTheWinner, regret, RegretType

    model = FollowTheWinner(strategy="eg")
    r = regret(model, X, regret_type=RegretType.STATIC, average=True)

    # Plot with plotly
    from skfolio.optimization.online._regret import plot_regret_curve
    fig = plot_regret_curve(r, average=True, label="EG vs BCRP")
    fig.show()

**Example -- dynamic regret with path-length budget:**

.. code-block:: python

    r_dyn = regret(
        model, X,
        regret_type=RegretType.DYNAMIC_UNIVERSAL,
        dynamic_config={"path_length": 5.0, "norm": "l1"},
        average=True,
    )


Comparing Online and Offline Strategies
***************************************

One of the strengths of the skfolio API is that online and offline estimators share
the same interface. You can directly compare them using the :ref:`Population <population>`
tools:

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.optimization.online import FollowTheWinner, FollowTheLoser
    from skfolio.population import Population
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = X["2014":"2019"], X["2020":]

    # Offline: Mean-Variance on training set, predict on test set
    mv = MeanRisk()
    mv.fit(X_train)
    ptf_mv = mv.predict(X_test)

    # Online: fit on full test set (streaming)
    eg = FollowTheWinner(strategy="eg")
    eg.fit(X_test)
    ptf_eg = eg.predict(X_test)

    olmar = FollowTheLoser(strategy="olmar")
    olmar.fit(X_test)
    ptf_olmar = olmar.predict(X_test)

    pop = Population([ptf_mv, ptf_eg, ptf_olmar])
    pop.plot_cumulative_returns()

.. note::

   Offline methods use a **train/test split** and predict on unseen data. Online
   methods process data sequentially -- each period's weights are chosen *before*
   seeing that period's returns, so there is no look-ahead bias even when
   ``fit`` and ``predict`` use the same ``X``.


Wealth Tracking and Transaction Costs
**************************************

All online estimators track cumulative wealth via ``wealth_`` and ``all_wealth_``,
accounting for transaction costs and management fees:

.. code-block:: python

    model = FollowTheWinner(
        strategy="eg",
        transaction_costs=0.001,     # 10 bps per trade
        management_fees=0.0001,      # 1 bp per period
    )
    model.fit(X)
    print(f"Final wealth: {model.wealth_:.4f}")
    print(f"Wealth history shape: {model.all_wealth_.shape}")


Mathematical Background
***********************

The theoretical foundation draws from two references:

    * Hazan, E. (2016). *Introduction to Online Convex Optimization.*
    * Orabona, F. (2023). *A Modern Introduction to Online Learning.*

The key result is that **Follow-the-Regularized-Leader** (FTRL) with an entropy
regularizer on the simplex gives the Exponentiated Gradient algorithm, achieving

.. math::

   R_T \leq O\!\left(\sqrt{T \log n}\right)

For the log-wealth loss :math:`-\log(w^\top x_t)`, which is *exp-concave*
(Hazan, Ch. 6), second-order methods such as the **Online Newton Step** can achieve
the tighter bound

.. math::

   R_T \leq O\!\left(n \log T\right)

Adaptive methods (AdaGrad, SWORD) further improve by adapting to the observed
gradient sequence rather than worst-case assumptions.

The **mean-reversion** family (OLMAR, PAMR, CWMR) operates outside the pure OCO
framework: it uses domain-specific predictors based on moving averages and margin
constraints, which can be formulated as passive-aggressive online learning or,
equivalently, as convex surrogate losses under mirror descent.


Going Further
*************

For detailed API documentation, see:

    * :class:`FollowTheWinner` -- Follow-the-Winner strategies
    * :class:`FollowTheLoser` -- Follow-the-Loser (mean reversion) strategies
    * :class:`BCRP` -- Best Constant Rebalanced Portfolio
    * :func:`regret` -- Regret computation

For the offline optimization methods, see :ref:`optimization`.
