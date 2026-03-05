"""Sequential walk-forward prediction for online portfolio selection.

This module provides :func:`online_walk_forward_predict`, which iterates folds
in chronological order, carries ``previous_weights`` across folds, and assembles
a :class:`~skfolio.portfolio.MultiPeriodPortfolio`.

Execution is always sequential (no parallelism) to guarantee correct state
propagation for online convex optimization estimators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn as sk
import sklearn.base as skb
import sklearn.model_selection as sks
import sklearn.utils as sku

from skfolio.model_selection._walk_forward import WalkForward
from skfolio.portfolio import MultiPeriodPortfolio
from skfolio.utils.tools import safe_split


def online_walk_forward_predict(
    estimator: skb.BaseEstimator,
    X,
    y=None,
    cv: WalkForward | sks.BaseCrossValidator | None = None,
    method: str = "predict",
    portfolio_params: dict[str, Any] | None = None,
) -> MultiPeriodPortfolio:
    """Sequential walk-forward prediction carrying previous weights.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted or unfitted estimator. A single clone is created and reused
        across all folds so that learned state can persist when the estimator
        supports ``warm_start``.

    X : array-like of shape (n_observations, n_assets)
        Asset net returns.

    y : array-like or None
        Optional target (ignored by most portfolio estimators).

    cv : WalkForward | BaseCrossValidator | None
        Cross-validation splitter. Folds are sorted by the first test index to
        guarantee chronological order regardless of the splitter implementation.

    method : str, default="predict"
        Estimator method called on each test fold. Common choices:

        * ``"predict"`` — returns a static :class:`Portfolio` per fold.
        * ``"predict_online"`` — returns a :class:`MultiPeriodPortfolio`
          per fold (the sequential trajectory); results are automatically
          flattened into a single continuous trajectory.

    portfolio_params : dict or None
        Extra keyword arguments forwarded to the final
        :class:`MultiPeriodPortfolio`.

    Returns
    -------
    MultiPeriodPortfolio
        One ``Portfolio`` per test observation (when ``method="predict"``) or
        the flattened continuous trajectory (when ``method="predict_online"``).

    Notes
    -----
    **Expanding-window caveat for online (OCO) estimators.**
    With expanding-window CV (e.g. ``WalkForward``), training windows overlap:
    fold *k*'s training set is a superset of fold *k-1*'s. Because the
    estimator clone persists across folds with ``warm_start=True``, calling
    ``fit`` on fold *k*'s full training window re-processes observations that
    were already seen in fold *k-1*. This is mathematically impure for
    algorithms whose state depends on seeing each observation exactly once
    (e.g. AdaGrad's gradient accumulators).

    For a true prequential (test-then-train) evaluation where each observation
    is processed exactly once, use ``fit`` on the initial training window and
    then call ``predict_online`` on successive test folds directly, or use the
    ``partial_fit`` streaming API.
    """
    portfolio_params = {} if portfolio_params is None else portfolio_params.copy()

    X, y = sku.indexable(X, y)
    cv = sks.check_cv(cv, y)

    splits = list(cv.split(X, y))
    sorted_fold_id = np.argsort(
        [test[0] if len(test) > 0 else -1 for _, test in splits]
    )

    predictions: list = []
    est = sk.clone(estimator)
    func = getattr(est, method)

    last_weights = None
    for fold_id in sorted_fold_id:
        train, test = splits[fold_id]
        X_train, y_train = safe_split(X, y, indices=train, axis=0)
        X_test, _ = safe_split(X, y, indices=test, axis=0)

        if hasattr(est, "previous_weights"):
            est.previous_weights = last_weights

        if y_train is None:
            est.fit(X_train)
        else:
            est.fit(X_train, y_train)

        p = func(X_test)

        if isinstance(p, MultiPeriodPortfolio):
            predictions.extend(p.portfolios)
            last_weights = (
                p.portfolios[-1].weights if p.portfolios else None
            )
        else:
            predictions.append(p)
            try:
                last_weights = p.weights
            except AttributeError:
                last_weights = None

    return MultiPeriodPortfolio(
        portfolios=predictions, check_observations_order=False, **portfolio_params
    )
