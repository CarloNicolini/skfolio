from enum import Enum, StrEnum, auto
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.online._benchmark import BCRP
from skfolio.optimization.online._loss import CLIP_EPSILON
from skfolio.optimization.online._utils import net_to_relatives


class RegretType(StrEnum):
    STATIC = auto()
    DYNAMIC = auto()


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


def _log_wealth_loss(relatives_row: np.ndarray, weights: np.ndarray) -> float:
    """Negative log-wealth loss for a single period.

    Computes: -log(max(w^T x, eps)) with numerical clipping.
    """
    dot = float(np.dot(weights, relatives_row))
    return -np.log(max(dot, CLIP_EPSILON))


def _losses_from_weights(
    relatives: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorized per-period losses from weights.

    Parameters
    ----------
    relatives : ndarray of shape (T, n)
        Price relatives per period.
    weights : ndarray of shape (T, n) or (n,)
        Sequence of weights per period or a single static weight vector.

    Returns
    -------
    ndarray of shape (T,)
        Per-period negative log-wealth losses.
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


def compute_regret_curve(
    estimator: BaseOptimization,
    X: ArrayLike,
    *,
    comparator: BaseOptimization | None = None,
    regret_type: RegretType = RegretType.STATIC,
    average: bool | str = False,
    window: int | None = None,
) -> np.ndarray:
    """Compute the regret curve over time using log-wealth loss.

    Parameters
    ----------
    estimator : BaseOptimization
        Online estimator (e.g., OPS) providing ``.fit`` and producing
        ``.all_weights_`` of shape (T, n).
    X : array-like of shape (T, n)
        Net returns per period (will be converted to price relatives internally).
    comparator : BaseOptimization, optional
        Comparator instance. If None, defaults to ``BCRP()``. For static regret we
        call ``fit`` and use ``weights_``. For dynamic regret we call ``fit_dynamic``
        and use ``all_weights_``.
    regret_type : RegretType, default=RegretType.STATIC
        Whether to compute static or dynamic regret.
    average : {False, "none", True, "running", "final"}, default=False
        Averaging mode for the returned curve:
        - False or "none": return cumulative (or windowed) regret curve
        - True or "running": return running-average curve (divide by t, or by window)
        - "final": return a constant array equal to final average (R_T/T, or last window average)
    window : int, optional
        If provided, compute sliding-window regret with the given window size.

    Returns
    -------
    ndarray of shape (T,)
        Running regret values per time step.
    """
    # Fit the online estimator and collect per-period weights
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

    # Use same data conversion as the online pipeline
    relatives = net_to_relatives(X)

    # Online losses
    online_losses = _losses_from_weights(relatives, W_online)

    # Comparator weights and losses
    comp = comparator if comparator is not None else BCRP()
    if regret_type == RegretType.STATIC:
        comp.fit(X)
        w_star = np.asarray(comp.weights_, dtype=float)
        comp_losses = _losses_from_weights(relatives, w_star)
    elif regret_type == RegretType.DYNAMIC:
        # Expect comparator to implement fit_dynamic(X) and expose all_weights_
        if not hasattr(comp, "fit_dynamic"):
            raise ValueError(
                "Dynamic regret requested but comparator has no fit_dynamic method"
            )
        comp.fit_dynamic(X)
        if not hasattr(comp, "all_weights_"):
            raise RuntimeError("comparator.fit_dynamic must set all_weights_")
        comp_losses = _losses_from_weights(
            relatives, np.asarray(comp.all_weights_, dtype=float)
        )
    else:
        raise ValueError(
            "Unknown regret_type. Use RegretType.STATIC or RegretType.DYNAMIC."
        )

    # Running (or windowed) regret curve
    return _running_regret(online_losses, comp_losses, average=average, window=window)
