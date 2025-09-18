"""Sequential walk-forward prediction for online portfolio selection.

This module provides `online_walk_forward_predict`, which iterates folds in
chronological order, carries `previous_weights` across folds, and assembles a
`MultiPeriodPortfolio`. It avoids parallel execution to ensure correct state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn as sk  # type: ignore
import sklearn.base as skb  # type: ignore
import sklearn.model_selection as sks  # type: ignore
import sklearn.utils as sku  # type: ignore

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

    Notes
    -----
    - Folds are evaluated strictly in chronological order (no parallelism).
    - If the estimator exposes `previous_weights`, it is updated with the
      latest predicted weights before the next fold.
    - The returned object is a `MultiPeriodPortfolio` with one `Portfolio` per fold.
    """
    portfolio_params = {} if portfolio_params is None else portfolio_params.copy()

    X, y = sku.indexable(X, y)
    cv = sks.check_cv(cv, y)

    splits = list(cv.split(X, y))
    # Order folds by first test index to guarantee chronology
    sorted_fold_id = np.argsort(
        [test[0] if len(test) > 0 else -1 for _, test in splits]
    )

    predictions = []
    est = sk.clone(estimator)
    func = getattr(est, method)

    last_weights = None
    for fold_id in sorted_fold_id:
        train, test = splits[fold_id]
        X_train, y_train = safe_split(X, y, indices=train, axis=0)
        X_test, _ = safe_split(X, y, indices=test, axis=0)

        # Carry previous weights if estimator supports it
        if hasattr(est, "previous_weights"):
            setattr(est, "previous_weights", last_weights)

        if y_train is None:
            est.fit(X_train)
        else:
            est.fit(X_train, y_train)

        p = func(X_test)
        predictions.append(p)

        # Update last_weights for next fold when available
        try:
            last_weights = p.weights
        except Exception:
            last_weights = None

    return MultiPeriodPortfolio(
        portfolios=predictions, check_observations_order=False, **portfolio_params
    )
