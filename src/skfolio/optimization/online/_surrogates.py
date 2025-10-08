from dataclasses import dataclass
from typing import Protocol

import numpy as np

from skfolio.utils.tools import AutoEnum


class SurrogateLossType(AutoEnum):
    HINGE = "hinge"
    SQUARED_HINGE = "squared_hinge"
    SOFTPLUS = "softplus"


# ============================================================================
# Generic margin and loss functions
# ============================================================================


def compute_margin(w: np.ndarray, phi: np.ndarray, epsilon: float) -> float:
    """Compute margin m = epsilon - phi^T w."""
    return epsilon - float(phi @ w)


def hinge_loss(margin: float) -> float:
    """Hinge loss: max(0, margin)."""
    return max(0.0, margin)


def hinge_loss_derivative(margin: float) -> float:
    """Derivative of hinge loss w.r.t. margin: 1 if margin > 0 else 0."""
    return 1.0 if margin > 0.0 else 0.0


def squared_hinge_loss(margin: float) -> float:
    """Squared hinge loss: (max(0, margin))^2."""
    m_pos = max(0.0, margin)
    return m_pos * m_pos


def squared_hinge_loss_derivative(margin: float) -> float:
    """Derivative of squared hinge loss w.r.t. margin: 2 * max(0, margin)."""
    return 2.0 * max(0.0, margin)


def softplus_loss(margin: float, beta: float) -> float:
    """
    Softplus loss: (1/beta) * log(1 + exp(beta * margin)).

    Numerically stable implementation for large |margin|.
    """
    z = beta * margin
    if z > 50:
        return z / beta
    if z < -50:
        return np.exp(z) / beta
    return np.log1p(np.exp(z)) / beta


def softplus_loss_derivative(margin: float, beta: float) -> float:
    """
    Derivative of softplus loss w.r.t. margin: sigmoid(beta * margin).

    Numerically stable implementation.
    """
    z = beta * margin
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (1.0 + ez)


class SurrogateLoss(Protocol):
    def value(self, w: np.ndarray, phi: np.ndarray) -> float: ...
    def grad(self, w: np.ndarray, phi: np.ndarray) -> np.ndarray: ...


@dataclass
class HingeLoss(SurrogateLoss):
    r"""Hinge surrogate loss: :math:`L(w) = \max(0, \epsilon - \phi^T w)`."""

    epsilon: float

    def value(self, w, phi):
        margin = compute_margin(w, phi, self.epsilon)
        return hinge_loss(margin)

    def grad(self, w, phi):
        margin = compute_margin(w, phi, self.epsilon)
        # Chain rule: dL/dw = (dL/dm) * (dm/dw) = indicator(margin > 0) * (-phi)
        dloss_dmargin = hinge_loss_derivative(margin)
        return -dloss_dmargin * phi if dloss_dmargin > 0.0 else np.zeros_like(phi)


@dataclass
class SoftplusLoss(SurrogateLoss):
    r"""Softplus loss: :math:`L(w) = (1/\beta) \log(1 + \exp(\beta (\epsilon - \phi^T w)))`."""

    epsilon: float
    beta: float = 5.0

    def value(self, w, phi):
        margin = compute_margin(w, phi, self.epsilon)
        return softplus_loss(margin, self.beta)

    def grad(self, w, phi):
        margin = compute_margin(w, phi, self.epsilon)
        # Chain rule: dL/dw = (dL/dm) * (dm/dw) = sigmoid(beta * margin) * (-phi)
        dloss_dmargin = softplus_loss_derivative(margin, self.beta)
        return -dloss_dmargin * phi


@dataclass
class SquaredHingeLoss(SurrogateLoss):
    r"""Squared hinge loss: :math:`L(w) = (\max(0, \epsilon - \phi^T w))^2`."""

    epsilon: float

    def value(self, w, phi):
        margin = compute_margin(w, phi, self.epsilon)
        return squared_hinge_loss(margin)

    def grad(self, w, phi):
        margin = compute_margin(w, phi, self.epsilon)
        # Chain rule: dL/dw = (dL/dm) * (dm/dw) = 2*max(0, margin) * (-phi)
        dloss_dmargin = squared_hinge_loss_derivative(margin)
        return -dloss_dmargin * phi if dloss_dmargin > 0.0 else np.zeros_like(phi)
