from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from skfolio.utils.equations import equations_to_matrix
from skfolio.utils.tools import input_to_array

CLIP_EPSILON = 1e-12


def net_to_relatives(X: ArrayLike) -> np.ndarray:
    """Convert net returns to price relatives robustly.

    Accepts 1D or 2D inputs and always returns a 2D array of shape (T, n).
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    return np.maximum(1.0 + X_arr, CLIP_EPSILON)


def validate_weights(
    w: np.ndarray,
    min_weights: ArrayLike | None,
    max_weights: ArrayLike | None,
    budget: float | None,
    linear_constraints: list[str] | None,
    groups: ArrayLike | None,
    left_inequality: ArrayLike | None,
    right_inequality: ArrayLike | None,
    max_turnover: float | None,
    previous_weights: ArrayLike | None,
) -> list[str]:
    """Return list of violated constraints for w (empty if none)."""
    n = w.shape[0]
    violations: list[str] = []
    # Box - using input_to_array with proper handling for scalars and None
    if min_weights is None:
        lo = None
    elif np.isscalar(min_weights):
        lo = np.full(n, min_weights, dtype=float)  # mypy: ignore
    else:
        lo = input_to_array(min_weights, n, 0.0, 1, None, "min_weights")

    if max_weights is None:
        up = None
    elif np.isscalar(max_weights):
        up = np.full(n, max_weights, dtype=float)
    else:
        up = input_to_array(max_weights, n, 1.0, 1, None, "max_weights")
    if lo is not None:
        bad = np.where(w < lo - 1e-12)[0]
        if bad.size:
            violations.append(f"min_weights violated at indices {bad.tolist()}")
    if up is not None:
        bad = np.where(w > up + 1e-12)[0]
        if bad.size:
            violations.append(f"max_weights violated at indices {bad.tolist()}")
    # Budget
    if budget is not None:
        s = float(np.sum(w))
        if abs(s - float(budget)) > 1e-10:
            violations.append(
                f"budget violated: sum(w)={s:.12g} != {float(budget):.12g}"
            )
    # Linear constraints (string equations)
    if linear_constraints is not None and len(linear_constraints) > 0:
        if groups is None:
            violations.append(
                "linear_constraints provided but groups is None (cannot parse)"
            )
        else:
            groups_np = np.asarray(groups)
            a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
                groups=groups_np,
                equations=linear_constraints,
                raise_if_group_missing=False,
            )
            if len(a_eq) != 0:
                r = a_eq @ w - b_eq
                bad = np.where(np.abs(r) > 1e-10)[0]
                if bad.size:
                    violations.append(
                        f"linear equality violated at rows {bad.tolist()}"
                    )
            if len(a_ineq) != 0:
                r = a_ineq @ w - b_ineq
                bad = np.where(r > 1e-12)[0]
                if bad.size:
                    violations.append(
                        f"linear inequality violated at rows {bad.tolist()}"
                    )
    # Matrix inequalities A w <= b
    if left_inequality is not None and right_inequality is not None:
        A = np.asarray(left_inequality, dtype=float)
        b = np.asarray(right_inequality, dtype=float)
        if A.ndim != 2 or b.ndim != 1 or A.shape[1] != n or A.shape[0] != b.shape[0]:
            violations.append(
                "left/right_inequality shapes are invalid for asset count"
            )
        else:
            r = A @ w - b
            bad = np.where(r > 1e-12)[0]
            if bad.size:
                violations.append(f"matrix inequality violated at rows {bad.tolist()}")
    # Turnover
    if max_turnover is not None:
        if previous_weights is None:
            violations.append("max_turnover provided but previous_weights is None")
        else:
            l1 = float(np.sum(np.abs(w - previous_weights)))
            if l1 > float(max_turnover) + 1e-12:
                violations.append(
                    f"max_turnover violated: L1={l1:.12g} > {float(max_turnover):.12g}"
                )
    return violations


def integer_simplex_grid(n_assets: int, m: int) -> np.ndarray:
    r"""
    Generate all compositions of integer m into n_assets nonnegative parts.

    This function generates a grid of integer points that sum to m, distributed
    across n_assets components. Each point represents a valid allocation where
    all components are non-negative integers.

    Parameters
    ----------
    n_assets : int
        Number of assets (components) in each composition.
    m : int
        Target sum for each composition.

    Returns
    -------
    np.ndarray of shape (n_compositions, n_assets)
        Array where each row is a composition of m into n_assets parts.
        All entries are non-negative integers that sum to m.

    Notes
    -----
    This function implements the "stars and bars" combinatorial method to
    enumerate all ways to distribute m identical items into n_assets distinct
    bins. The total number of compositions is C(m + n_assets - 1, n_assets - 1).

    Used primarily for generating expert portfolios in Universal Portfolio
    algorithms, where each composition represents a discrete allocation
    strategy on the probability simplex.

    Examples
    --------
    >>> integer_simplex_grid(2, 3)
    array([[3., 0.],
           [2., 1.],
           [1., 2.],
           [0., 3.]])
    """
    if n_assets <= 0 or m <= 0:
        return np.zeros((0, 0), dtype=float)
    comb = np.fromiter(
        (
            c
            for bars in combinations(range(m + n_assets - 1), n_assets - 1)
            for c in bars
        ),
        dtype=np.int64,
    )
    if comb.size == 0:
        # n_assets == 1 -> single point (m)
        return np.array([[float(m)]], dtype=float)
    comb = comb.reshape(-1, n_assets - 1)
    padded = np.empty((comb.shape[0], n_assets + 1), dtype=np.int64)
    padded[:, 0] = -1
    padded[:, 1:-1] = comb
    padded[:, -1] = m + n_assets - 1
    points = np.diff(padded, axis=1) - 1
    return points.astype(float)
