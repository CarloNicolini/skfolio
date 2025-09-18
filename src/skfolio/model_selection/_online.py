"""Sequential walk-forward prediction for online portfolio selection.

This module provides `online_walk_forward_predict`, which iterates folds in
chronological order, carries `previous_weights` across folds, and assembles a
`MultiPeriodPortfolio`. It avoids parallel execution to ensure correct state.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import sklearn as sk  # type: ignore
import sklearn.base as skb  # type: ignore
import sklearn.model_selection as sks  # type: ignore
import sklearn.utils as sku  # type: ignore

from skfolio.model_selection._combinatorial import BaseCombinatorialCV
from skfolio.model_selection._walk_forward import WalkForward
from skfolio.population import Population
from skfolio.portfolio import BasePortfolio, MultiPeriodPortfolio
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
            est.previous_weights = last_weights

        # Prefer incremental learning when available
        if hasattr(est, "partial_fit"):
            if y_train is None:
                est.partial_fit(X_train)
            else:
                est.partial_fit(X_train, y_train)
        else:
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


def online_combinatorial_predict(
    estimator: skb.BaseEstimator,
    X,
    y=None,
    cv: BaseCombinatorialCV | None = None,
    method: str = "predict",
    portfolio_params: dict[str, Any] | None = None,
) -> Population:
    """Sequential CPCV prediction carrying previous weights per recombined path.

    Notes
    -----
    - Iterates splits in increasing order and constructs each recombined test path.
    - For each path, folds are evaluated sequentially; if the estimator exposes
      `previous_weights`, it is updated with the last predicted weights along that path.
    - Returns a `Population` of `MultiPeriodPortfolio`, one per recombined path.
    """
    if cv is None:
        raise ValueError("cv must be a BaseCombinatorialCV instance")

    portfolio_params = {} if portfolio_params is None else portfolio_params.copy()

    X, y = sku.indexable(X, y)

    # Collect all splits once
    # In CPCV, each split yields (train_indices, list_of_test_indices)
    splits = list(cast(Iterable[tuple[np.ndarray, list[np.ndarray]]], cv.split(X, y)))

    # Mapping from (split i, test j) -> path_id
    path_ids = cv.get_path_ids()
    n_paths = int(np.max(path_ids)) + 1 if path_ids.size else 0

    paths_result: list[MultiPeriodPortfolio | None] = [None for _ in range(n_paths)]

    # For each path, walk sequentially over splits
    for path_id in range(n_paths):
        est = sk.clone(estimator)
        func = getattr(est, method)
        last_weights = None

        path_portfolios = []
        for i, pair in enumerate(splits):
            train, tests = pair
            # Identify the test block index belonging to this path for split i
            j_candidates = np.where(path_ids[i] == path_id)[0]
            if j_candidates.size != 1:
                # No test block for this split in this path (robustness)
                continue
            j = int(j_candidates[0])
            # Narrow types for the linter
            test: np.ndarray = tests[j]

            X_train, y_train = safe_split(X, y, indices=train, axis=0)
            X_test, _ = safe_split(X, y, indices=test, axis=0)

            if hasattr(est, "previous_weights"):
                # Direct attribute set; estimator may define this dynamically
                est.previous_weights = last_weights  # type: ignore[attr-defined]

            # Prefer incremental learning when available
            if hasattr(est, "partial_fit"):
                if y_train is None:
                    est.partial_fit(X_train)
                else:
                    est.partial_fit(X_train, y_train)
            else:
                if y_train is None:
                    est.fit(X_train)
                else:
                    est.fit(X_train, y_train)

            p = func(X_test)
            path_portfolios.append(p)

            try:
                last_weights = p.weights
            except Exception:
                last_weights = None

        name = portfolio_params.get("name", f"path_{path_id}")
        mpp = MultiPeriodPortfolio(
            name=name,
            portfolios=path_portfolios,
            check_observations_order=False,
            **portfolio_params,
        )
        paths_result[path_id] = mpp
    # Filter out any None paths (should not happen in normal CPCV)
    finalized = [p for p in paths_result if p is not None]
    finalized_bp: list[BasePortfolio] = cast(list[BasePortfolio], finalized)
    return Population(finalized_bp)
