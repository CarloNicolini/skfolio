from enum import StrEnum, auto
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike

from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.online._benchmark import BCRP
from skfolio.optimization.online._mixins import RegretType as LegacyRegretType
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives

_LEGACY_TO_REGRET: dict[Any, "RegretType"] = {}


class RegretType(StrEnum):
    STATIC = auto()
    # Legacy dynamic mode (prefix BCRP). Kept for backward compatibility.
    DYNAMIC = auto()
    DYNAMIC_UNIVERSAL = (
        auto()
    )  # universal dynamic regret with path-length budget or penalty
    DYNAMIC_WORST_CASE = auto()  # per-round minimizers (one-hot on argmax relative)
    DYNAMIC_LEGACY = auto()  # explicit alias for legacy


if isinstance(LegacyRegretType, type):
    _LEGACY_TO_REGRET = {
        LegacyRegretType.STATIC: RegretType.STATIC,
        LegacyRegretType.DYNAMIC: RegretType.DYNAMIC,
    }
    if hasattr(LegacyRegretType, "DYNAMIC_UNIVERSAL"):
        _LEGACY_TO_REGRET[LegacyRegretType.DYNAMIC_UNIVERSAL] = (
            RegretType.DYNAMIC_UNIVERSAL
        )
    if hasattr(LegacyRegretType, "DYNAMIC_WORST_CASE"):
        _LEGACY_TO_REGRET[LegacyRegretType.DYNAMIC_WORST_CASE] = (
            RegretType.DYNAMIC_WORST_CASE
        )
    if hasattr(LegacyRegretType, "DYNAMIC_LEGACY"):
        _LEGACY_TO_REGRET[LegacyRegretType.DYNAMIC_LEGACY] = RegretType.DYNAMIC_LEGACY


def plot_regret_curve(
    regret: ArrayLike,
    *,
    fig: Any | None = None,
    average: bool = True,
    label: str | None = None,
):
    """Plot a regret curve over time using plotly.

    Parameters
    ----------
    regret : array-like of shape (T,)
        Regret values per time step.
    fig : plotly figure, optional
        Figure to plot on; created if None.
    average : bool, default=True
        If True, the label defaults to 'Average regret'; otherwise 'Regret'.
    label : str, optional
        Custom label for the curve.
    """
    import plotly.graph_objects as go

    r = np.asarray(regret, dtype=float)
    if fig is None:
        fig = go.Figure()
    if label is None:
        label = "Average regret" if average else "Regret"

    fig.add_trace(go.Scatter(x=np.arange(1, r.size + 1), y=r, mode="lines", name=label))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=label,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.3)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.3)"),
        showlegend=True,
    )

    return fig


def _losses_from_weights(
    relatives: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorized per-period negative log-wealth losses.

    relatives : (T, n), weights : (T, n) or (n,)
    Returns: (T,)
    """
    R = np.asarray(relatives, dtype=float)
    if weights.ndim == 1:
        w = np.asarray(weights, dtype=float)
        dots = R @ w
    else:
        W = np.asarray(weights, dtype=float)
        dots = np.sum(W * R, axis=1)
    dots = np.maximum(dots, CLIP_EPSILON)
    return -np.log(dots)


def _running_regret(
    online_losses: np.ndarray,
    comp_losses: np.ndarray,
    *,
    average: bool | str = False,
    window: int | None = None,
) -> np.ndarray:
    """Compute running (or windowed) regret curve with optional averaging.

    average can be:
    - False or "none": no averaging (default)
    - True or "running": divide by t (or by window size when windowed)
    - "final": return a constant array equal to the final average value
    """
    T = online_losses.size
    avg_mode = (
        "running"
        if average is True
        else (average if isinstance(average, str) else "none")
    )

    if window is not None:
        if not (1 <= int(window) <= T):
            raise ValueError("window must be between 1 and T.")
        wr_raw = np.zeros(T, dtype=float)
        wsize = int(window)
        for t in range(wsize - 1, T):
            o = float(np.sum(online_losses[t - wsize + 1 : t + 1]))
            c = float(np.sum(comp_losses[t - wsize + 1 : t + 1]))
            wr_raw[t] = o - c
        if avg_mode == "running":
            return wr_raw / float(wsize)
        if avg_mode == "final":
            final_avg = wr_raw[-1] / float(wsize)
            return np.full(T, final_avg, dtype=float)
        return wr_raw

    # No window: cumulative
    rr_raw = np.cumsum(online_losses - comp_losses)
    if avg_mode == "running":
        return rr_raw / np.arange(1, T + 1, dtype=float)
    if avg_mode == "final":
        final_avg = rr_raw[-1] / float(T)
        return np.full(T, final_avg, dtype=float)
    return rr_raw


def _worst_case_dynamic_weights(relatives: np.ndarray) -> np.ndarray:
    """One-hot comparators on argmax asset per round for log-wealth loss.

    relatives: (T, n) strictly positive gross relatives.
    Return W*: (T, n) with one-hot rows. Ensures per-round minimal loss.
    """
    R = np.asarray(relatives, dtype=float)
    if R.ndim != 2:
        raise ValueError("relatives must be (T, n)")
    idx = np.argmax(R, axis=1)
    T, n = R.shape
    W = np.zeros((T, n), dtype=float)
    W[np.arange(T), idx] = 1.0
    return W


def _universal_dynamic_weights(
    relatives: np.ndarray,
    *,
    path_length: float | None = None,
    path_penalty: float | None = None,
    norm: str = "l1",
    solver: str | None = "CLARABEL",
    solver_params: dict | None = None,
) -> np.ndarray:
    r"""
    Solve universal dynamic comparator sequence u_1..u_T on the simplex.

        maximize     sum_t log( r_t^T u_t )
        subject to   u_t >= 0, sum_i u_{t,i} = 1  for each t
                     sum_{t=2}^T ||u_t - u_{t-1}||_p <= path_length    (constraint)
                     or penalized with path_penalty * sum ||u_t - u_{t-1}||_p

    - Use norm='l1' (default) or 'l2'. L1 matches many path-length results in the literature.
    - If path_length=0, this reduces to STATIC BCRP (one u_t constant), which we can detect.

    This implements the canonical universal dynamic regret comparator used in OCO,
    where non-stationarity is captured by the path-length PT, a key regularity in
    dynamic regret bounds for OGD and SWORD families 【dynamic-regret.pdf】.
    """
    R = np.asarray(relatives, dtype=float)
    if R.ndim != 2:
        raise ValueError("relatives must be (T, n) gross relatives")
    T, n = R.shape
    if T == 0:
        return np.zeros((0, n), dtype=float)
    if norm not in ("l1", "l2"):
        raise ValueError("norm must be 'l1' or 'l2'")
    if path_length is None and path_penalty is None:
        raise ValueError(
            "Provide either path_length or path_penalty for universal dynamic comparator."
        )
    if path_length is not None and path_length < 0:
        raise ValueError("path_length must be nonnegative.")
    if path_penalty is not None and path_penalty < 0:
        raise ValueError("path_penalty must be nonnegative.")

    # Special case: path_length == 0 -> STATIC comparator (one constant u) = BCRP
    if path_length is not None and path_length <= 1e-18:
        # Equivalent to maximizing sum_t log(R[t]^T u) with u constant
        # Solve with a single vector u via BCRP on the whole sample
        # But here we return constant weights across time (project BCRP to all rows).
        # We reuse BCRP for exactness and robustness.
        # Note: fits on net returns, so we convert back to net here
        X_net = R - 1.0
        bcrp = BCRP().fit(X_net)
        w = np.asarray(bcrp.weights_, dtype=float)
        return np.repeat(w[None, :], T, axis=0)

    # Decision variable: one weight vector per t
    W = cp.Variable((T, n))
    constraints: list[Any] = [W >= 0, cp.sum(W, axis=1) == 1]

    diffs = W[1:, :] - W[:-1, :] if T >= 2 else None
    if diffs is not None:
        if norm == "l1":
            path_expr = cp.sum(cp.norm1(diffs, axis=1))
        else:
            path_expr = cp.sum(cp.norm2(diffs, axis=1))
    else:
        path_expr = 0.0

    # Objective: maximize sum log(R[t]^T W_t). This is concave; maximizing is DCP-valid.
    # R is strictly positive (CLIP_EPSILON), so we don't need extra positivity constraints for the affine inner product.
    obj_terms = [cp.log(R[t, :] @ W[t, :]) for t in range(T)]
    if path_length is not None:
        constraints.append(path_expr <= float(path_length))
        objective = cp.Maximize(cp.sum(obj_terms))
    else:
        objective = cp.Maximize(cp.sum(obj_terms) - float(path_penalty) * path_expr)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver, **(solver_params or {}))
    except Exception:
        prob.solve(**(solver_params or {}))

    if W.value is None:
        raise RuntimeError(
            "Universal dynamic comparator optimization failed to converge."
        )

    return np.asarray(W.value, dtype=float)


def regret(
    estimator: BaseOptimization,
    X: ArrayLike,
    *,
    comparator: BaseOptimization | None = None,
    regret_type: RegretType = RegretType.STATIC,
    average: bool | str = False,
    window: int | None = None,
    dynamic_config: dict | None = None,
) -> np.ndarray:
    """Compute regret curves over time using the negative log-wealth loss: ft(w) = -log(w^T r_t).

    Parameters
    ----------
    estimator : BaseOptimization
        Online estimator (e.g., OPS) providing `.fit(X)` and exposing `.all_weights_` of shape (T, n).
    X : array-like of shape (T, n)
        Net returns per period. Internally converted to gross relatives via `1 + r`.
    comparator : BaseOptimization, optional
        - For STATIC: if None, defaults to BCRP() (best fixed in hindsight).
        - For DYNAMIC_UNIVERSAL and DYNAMIC_WORST_CASE: ignored; comparator is defined by regret_type.
        - For DYNAMIC_LEGACY or DYNAMIC: if None, defaults to BCRP().fit_dynamic(...) as before.
    regret_type : RegretType
        STATIC, DYNAMIC_UNIVERSAL, DYNAMIC_WORST_CASE, or DYNAMIC_LEGACY (DYNAMIC is an alias to LEGACY).
    average : {False, "none", True, "running", "final"}, default=False
        Averaging mode for the returned curve:
        - False or "none": return cumulative (or windowed) regret curve
        - True or "running": return running-average curve (divide by t, or by window)
        - "final": return a constant array equal to final average (R_T/T, or last window average)
    window : int, optional
        If provided, compute sliding-window regret with the given window size.
    dynamic_config : dict, optional
        Additional options for universal dynamic regret comparator:
            - path_length: float >= 0           (sum_t ||u_t - u_{t-1}||_p <= path_length)
            - path_penalty: float >= 0          (Lagrangian penalty instead of constraint)
            - norm: "l1" (default) or "l2"      (p in the path length)
            - solver: str or None               (cvxpy solver; default "CLARABEL")
            - solver_params: dict               (forwarded to cvxpy)

    Notes
    -----
    - For path_length=0, DYNAMIC_UNIVERSAL reduces to STATIC comparator (BCRP).
    - For L1 norm and sufficiently large path_length (≥ 2*(T-1)), the universal dynamic comparator approaches the worst-case dynamic comparator (can change fully every round).

    Returns
    -------
    ndarray of shape (T,)
        Regret curve (cumulative or running-averaged, or windowed).

    References
    ----------
    - Static regret, universal dynamic regret, and worst-case dynamic regret definitions are standard
      in OCO (see, e.g., Zhao et al. 2024, Eqs. (1)-(3)) (https://jmlr.org/papers/volume25/21-0748/21-0748.pdf).
    - Path length PT is the canonical non-stationarity measure in dynamic regret bounds; SWORD
      algorithms achieve problem-dependent bounds in terms of PT and gradient variation VT (and small loss FT) under smoothness, within the universal dynamic regret framework.
    """
    # Fit the online estimator and collect per-period weights
    if isinstance(regret_type, RegretType):
        rt = regret_type
    elif isinstance(LegacyRegretType, type) and isinstance(
        regret_type, LegacyRegretType
    ):
        try:
            rt = _LEGACY_TO_REGRET[regret_type]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported regret_type: {regret_type!r}") from exc
    elif isinstance(regret_type, str):
        rt = RegretType(regret_type)
    else:
        rt = RegretType(regret_type)

    est = estimator.fit(X)
    if not hasattr(est, "all_weights_"):
        raise RuntimeError("estimator must expose all_weights_ after fit(X)")

    W_online = np.asarray(est.all_weights_, dtype=float)
    if W_online.ndim != 2:
        # Fallback: build from last weights repeated, but this should not happen for OPS
        W_online = np.repeat(
            np.asarray(est.weights_, dtype=float)[None, :],
            np.asarray(X).shape[0],
            axis=0,
        )

    relatives = net_to_relatives(X)
    online_losses = _losses_from_weights(relatives, W_online)

    # Comparator weights and losses
    if rt == RegretType.DYNAMIC:
        # Alias to legacy; warn to migrate
        import warnings

        warnings.warn(
            "RegretType.DYNAMIC is legacy (prefix BCRP). Use DYNAMIC_UNIVERSAL or "
            "DYNAMIC_WORST_CASE for literature-aligned dynamic regret.",
            UserWarning,
            stacklevel=2,
        )
        rt = RegretType.DYNAMIC_LEGACY

    if rt == RegretType.STATIC:
        comp = comparator if comparator is not None else BCRP()
        comp.fit(X)
        w_star = np.asarray(comp.weights_, dtype=float)
        comp_losses = _losses_from_weights(relatives, w_star)

    elif rt == RegretType.DYNAMIC_LEGACY:
        comp = comparator if comparator is not None else BCRP()
        if not hasattr(comp, "fit_dynamic"):
            raise ValueError(
                "Legacy dynamic regret requested but comparator has no fit_dynamic method"
            )
        comp.fit_dynamic(X)
        if not hasattr(comp, "all_weights_"):
            raise RuntimeError("comparator.fit_dynamic must set all_weights_")
        comp_losses = _losses_from_weights(
            relatives, np.asarray(comp.all_weights_, dtype=float)
        )

    elif rt == RegretType.DYNAMIC_WORST_CASE:
        W_wc = _worst_case_dynamic_weights(relatives)
        comp_losses = _losses_from_weights(relatives, W_wc)

    elif rt == RegretType.DYNAMIC_UNIVERSAL:
        cfg = dynamic_config or {}
        W_ud = _universal_dynamic_weights(
            relatives,
            path_length=cfg.get("path_length", None),
            path_penalty=cfg.get("path_penalty", None),
            norm=cfg.get("norm", "l1"),
            solver=cfg.get("solver", "CLARABEL"),
            solver_params=cfg.get("solver_params", None),
        )
        comp_losses = _losses_from_weights(relatives, W_ud)

    else:
        raise ValueError(
            "Unknown regret_type. Use STATIC, DYNAMIC_UNIVERSAL, DYNAMIC_WORST_CASE, "
            "or DYNAMIC_LEGACY (DYNAMIC aliases to legacy)."
        )

    # Running (or windowed) regret curve
    return _running_regret(online_losses, comp_losses, average=average, window=window)
