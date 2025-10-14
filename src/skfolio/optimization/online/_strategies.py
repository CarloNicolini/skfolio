"""
Strategy classes for mean-reversion portfolio selection.

Each strategy (PAMR, OLMAR, CWMR) is implemented as a separate class
that handles both PA and MD update modes.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np  # mypy: ignore
from scipy.stats import norm  # mypy: ignore

from skfolio.optimization.online import _cwmr
from skfolio.optimization.online._autograd_objectives import BaseObjective
from skfolio.optimization.online._ftrl import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import EuclideanMirrorMap
from skfolio.optimization.online._mixins import PAMRVariant, UpdateMode
from skfolio.optimization.online._prediction import BaseReversionPredictor
from skfolio.optimization.online._projection import AutoProjector


class BaseStrategy(ABC):
    """Base class for mean-reversion strategies."""

    @abstractmethod
    def reset(self, d: int) -> None:
        """Reset strategy state for d assets."""
        pass

    @abstractmethod
    def should_update(self, t: int) -> bool:
        """Check if update should happen at time t."""
        pass

    @abstractmethod
    def step(
        self,
        trade_w: np.ndarray,
        arr: np.ndarray,
        update_mode: UpdateMode,
        projector: AutoProjector,
    ) -> np.ndarray:
        """Execute the strategy update step."""
        pass


class PAMRStrategy(BaseStrategy):
    """PAMR (Passive-Aggressive Mean Reversion) strategy."""

    def __init__(
        self,
        objective: BaseObjective,
        engine: FirstOrderOCO | None,
        epsilon: float,
        pamr_variant: PAMRVariant = PAMRVariant.SIMPLE,
        pamr_C: float = 10.0,
        transaction_costs_arr: np.ndarray | None = None,
    ):
        self.engine = engine
        self.objective = objective
        self.epsilon = epsilon
        self.variant = (
            PAMRVariant(pamr_variant) if isinstance(pamr_variant, int) else pamr_variant
        )
        self.pamr_C = pamr_C
        self.transaction_costs_arr = transaction_costs_arr

    def reset(self, d: int) -> None:
        pass  # PAMR has no state

    def should_update(self, t: int) -> bool:
        return True  # Always update

    def step(
        self,
        trade_w: np.ndarray,
        arr: np.ndarray,  # arr = phi_eff for PAMR
        update_mode: UpdateMode,
        projector: AutoProjector,
    ) -> np.ndarray:
        """Execute PAMR update (PA or MD mode)."""
        match update_mode:
            case UpdateMode.PA:
                return self._pa_step(trade_w, arr, projector)
            case UpdateMode.MD:
                return self._md_step(trade_w, arr, projector)

    def _pa_step(
        self, trade_w: np.ndarray, arr: np.ndarray, projector: AutoProjector
    ) -> np.ndarray:
        """PAMR passive-aggressive step using objective.

        PAMR wants w^T x <= epsilon, so loss = max(0, w^T x - epsilon).
        This is opposite to OLMAR which wants phi^T w >= epsilon.

        The objective is designed for OLMAR semantics, so we pass -x_t
        to flip the sign: loss(w, -x) = max(0, ε - w^T(-x)) = max(0, ε + w^T x)
        is wrong, so instead we negate the gradient result.
        """
        x_t = arr  # For PAMR, arr is x_t

        # PAMR semantics: pass -x_t to objective to flip sign
        # loss(w, -x_t) = max(0, ε - w^T(-x_t)) = max(0, ε + w^T x_t)
        # But we want: max(0, w^T x_t - ε)
        # So let's compute directly for PAMR
        margin = float(np.dot(trade_w, x_t))
        loss = max(0.0, margin - self.epsilon)

        if loss <= 0.0:  # passive case, no update
            return trade_w.copy()

        # PAMR gradient: +x_t when violated (want to reduce w^T x)
        # Objective returns -phi, so negate: -(-x_t) = x_t
        # Actually, let's compute directly for clarity
        c = x_t - np.mean(x_t)

        # Add transaction cost subgradient to direction
        if self.transaction_costs_arr is not None:
            prev = projector.config.previous_weights
            if prev is not None:
                prev_arr = np.asarray(prev, dtype=float)
                if prev_arr.shape == trade_w.shape:
                    delta = trade_w - prev_arr
                    tc_grad = self.transaction_costs_arr * np.sign(delta)
                    c = c + tc_grad
                    c = c - np.mean(c)  # Re-center

        c_norm_sq = float(np.dot(c, c))

        if c_norm_sq < 1e-16:
            return trade_w.copy()

        # Compute tau based on PAMR variant
        # All variants use the loss value directly (design decision: option a)
        match self.variant:
            case PAMRVariant.SIMPLE:
                tau = loss / c_norm_sq
            case PAMRVariant.SLACK_LINEAR:
                tau = min(self.pamr_C, loss / c_norm_sq)
            case PAMRVariant.SLACK_QUADRATIC:
                tau = loss / (c_norm_sq + 1.0 / (2.0 * self.pamr_C))
            case _:
                raise ValueError(f"Unhandled PAMR variant: {self.variant}")

        # Update: w_new = w - tau * c (gradient descent step)
        w_new = trade_w - tau * c
        projector.config.previous_weights = trade_w
        return projector.project(w_new)

    def _md_step(
        self, trade_w: np.ndarray, phi_eff: np.ndarray, projector: AutoProjector
    ) -> np.ndarray:
        """PAMR mirror-descent step.

        PAMR wants w^T x <= epsilon. When violated (w^T x > epsilon),
        gradient points in direction of x to reduce the margin.

        Since the objective is designed for OLMAR semantics (grad = -phi),
        we need to negate to get PAMR semantics (grad = +x).
        """
        if self.engine is None:
            raise ValueError("MD mode requires engine to be initialized")

        x_t = phi_eff  # For PAMR, phi_eff is x_t
        margin = float(np.dot(trade_w, x_t))

        # PAMR gradient: +x_t when margin > epsilon, else 0
        if margin > self.epsilon:
            g = x_t
        else:
            g = np.zeros_like(x_t)

        # Center gradient for Euclidean geometry
        if isinstance(self.engine.map, EuclideanMirrorMap):
            g -= np.mean(g)
        else:
            warnings.warn(
                "PAMR MD step with non-Euclidean mirror map is experimental and may not work as intended.",
                stacklevel=2,
            )

        # Add transaction cost subgradient
        if self.transaction_costs_arr is not None:
            prev = projector.config.previous_weights
            if prev is not None:
                prev_arr = np.asarray(prev, dtype=float)
                if prev_arr.shape == trade_w.shape:
                    delta = trade_w - prev_arr
                    g += self.transaction_costs_arr * np.sign(delta)

        # TODO this assignment is a bit ugly, we should refactor the engine interface?
        projector.config.previous_weights = trade_w

        return self.engine.step(g)


class CWMRStrategy(BaseStrategy):
    """CWMR (Confidence-Weighted Mean Reversion) strategy."""

    def __init__(
        self,
        cwmr_eta: float,
        cwmr_sigma0: float,
        cwmr_min_var: float | None,
        cwmr_max_var: float | None,
        cwmr_mean_lr: float,
        cwmr_var_lr: float,
        epsilon: float,
        initial_weights: np.ndarray,
    ):
        self.cwmr_eta = cwmr_eta
        self.cwmr_sigma0 = cwmr_sigma0
        self.cwmr_min_var = cwmr_min_var
        self.cwmr_max_var = cwmr_max_var
        self.cwmr_mean_lr = cwmr_mean_lr
        self.cwmr_var_lr = cwmr_var_lr
        self.epsilon = epsilon

        self._mu: np.ndarray | None = None
        self._Sdiag: np.ndarray | None = None
        self._quantile: float | None = None
        self._initial_weights = initial_weights

    def reset(self, d: int) -> None:
        if self._quantile is None:
            self._quantile = float(norm.ppf(self.cwmr_eta))
        if self._mu is None:
            self._mu = self._initial_weights.copy()
        if self._Sdiag is None:
            self._Sdiag = np.full(d, self.cwmr_sigma0, dtype=float)
            self._Sdiag = _cwmr.clip_variances(
                self._Sdiag, self.cwmr_min_var, self.cwmr_max_var
            )

    def should_update(self, t: int) -> bool:
        return True

    def step(
        self,
        trade_w: np.ndarray,
        arr: np.ndarray,
        update_mode: UpdateMode,
        projector: AutoProjector,
    ) -> np.ndarray:
        """Execute CWMR update (PA or MD mode)."""
        match update_mode:
            case UpdateMode.PA:
                return self._pa_step(trade_w, arr, projector)
            case UpdateMode.MD:
                return self._md_step(trade_w, arr, projector)

    def _pa_step(
        self, trade_w: np.ndarray, x_t: np.ndarray, projector: AutoProjector
    ) -> np.ndarray:
        """CWMR passive-aggressive step (closed-form KKT solution)."""
        mu_new, diag_new = _cwmr.pa_distribution_update(
            self._mu, self._Sdiag, x_t, self.epsilon, self._quantile
        )

        # Clip variances after update
        diag_new = _cwmr.clip_variances(diag_new, self.cwmr_min_var, self.cwmr_max_var)

        # Project mean to feasible set
        projector.config.previous_weights = trade_w
        w_next = projector.project(mu_new)

        # Update state
        self._mu = w_next.copy()
        self._Sdiag = diag_new.copy()
        return w_next

    def _md_step(
        self, trade_w: np.ndarray, x_t: np.ndarray, projector: AutoProjector
    ) -> np.ndarray:
        """
        CWMR OCO/mirror-descent step (experimental variant).

        WARNING: This is experimental. The standard CWMR uses PA mode.
        """
        phi = float(self._quantile)
        s = float(np.dot(self._Sdiag * x_t, x_t))
        sqrt_s = np.sqrt(max(s, 0.0))
        margin = float(np.dot(self._mu, x_t))
        violation = margin + phi * sqrt_s - self.epsilon

        if violation <= 0.0:
            # Constraint satisfied, no update
            self._mu = trade_w.copy()
            self._Sdiag = _cwmr.clip_variances(
                self._Sdiag, self.cwmr_min_var, self.cwmr_max_var
            )
            return trade_w.copy()

        # Mean update: gradient of (mu^T x) is x_t
        mu_candidate = self._mu - self.cwmr_mean_lr * x_t

        # Project to feasible set
        projector.config.previous_weights = trade_w
        w_next = projector.project(mu_candidate)
        self._mu = w_next.copy()

        # Variance update: gradient descent in log-space
        if self.cwmr_var_lr > 0.0 and sqrt_s > 0.0:
            grad_sigma = (phi / (2.0 * max(sqrt_s, 1e-18))) * (x_t**2)
            log_diag = np.log(np.maximum(self._Sdiag, 1e-18))
            log_diag_new = log_diag - self.cwmr_var_lr * grad_sigma
            diag_new = np.exp(log_diag_new)
        else:
            diag_new = self._Sdiag

        self._Sdiag = _cwmr.clip_variances(
            diag_new, self.cwmr_min_var, self.cwmr_max_var
        )
        return w_next


class OLMARStrategy(BaseStrategy):
    """OLMAR (Online Moving Average Reversion) strategy."""

    def __init__(
        self,
        predictor: BaseReversionPredictor,
        objective: BaseObjective,
        engine: FirstOrderOCO | None,
        epsilon: float,
        olmar_order: int,
        transaction_costs_arr: np.ndarray | None = None,
    ):
        self.predictor = predictor
        self.objective = objective
        self.engine = engine
        self.epsilon = epsilon
        self.olmar_order = olmar_order
        self.transaction_costs_arr = transaction_costs_arr
        self._t = 0

    def reset(self, d: int) -> None:
        self.predictor.reset(d)
        self._t = 0

    def should_update(self, t: int) -> bool:
        self._t = t
        if self.olmar_order == 1:
            # OLMAR-1 needs window periods of history before updating
            # After window periods (t=0,1,...,window-1), we have window+1 prices
            # Start updating at t=window
            return t >= self.predictor.window
        elif self.olmar_order == "rmr":
            # RMR uses same cold-start as OLMAR-1
            return t >= self.predictor.window
        # OLMAR-2 can start updating from t >= 1
        return t >= 1

    def step(
        self,
        trade_w: np.ndarray,
        arr: np.ndarray,
        update_mode: UpdateMode,
        projector: AutoProjector,
    ) -> np.ndarray:
        """Execute OLMAR update (PA or MD mode)."""
        match update_mode:
            case UpdateMode.PA:
                return self._pa_step(trade_w, arr, projector)
            case UpdateMode.MD:
                return self._md_step(trade_w, arr, projector)
            case _:
                raise ValueError(f"Unknown update mode: {update_mode}")

    def _pa_step(
        self, trade_w: np.ndarray, phi_eff: np.ndarray, projector: AutoProjector
    ) -> np.ndarray:
        """OLMAR passive-aggressive step using objective.

        OLMAR wants phi^T w >= epsilon, so loss = max(0, epsilon - phi^T w).
        The PA update moves toward phi to increase phi^T w.
        """
        # Compute loss using objective
        loss = self.objective.loss(trade_w, phi_eff)
        if loss <= 0.0:
            return trade_w.copy()

        # Compute gradient using objective
        # Objective grad = -phi (loss decrease direction)
        # But PA wants to move toward +phi (increase phi^T w)
        # So negate the gradient
        g = -self.objective.grad(trade_w, phi_eff)

        # Center for simplex tangent space
        c = g - np.mean(g)

        # Add transaction cost subgradient to direction
        if self.transaction_costs_arr is not None:
            prev = projector.config.previous_weights
            if prev is not None:
                prev_arr = np.asarray(prev, dtype=float)
                if prev_arr.shape == trade_w.shape:
                    delta = trade_w - prev_arr
                    tc_grad = self.transaction_costs_arr * np.sign(delta)
                    c = c + tc_grad
                    c = c - np.mean(c)  # Re-center

        c_norm_sq = float(np.dot(c, c))
        if c_norm_sq <= 1e-16:
            return trade_w.copy()

        lam = loss / c_norm_sq
        projector.config.previous_weights = trade_w
        return projector.project(trade_w + lam * c)

    def _md_step(
        self, trade_w: np.ndarray, phi_eff: np.ndarray, projector: AutoProjector
    ) -> np.ndarray:
        """OLMAR mirror-descent step using objective gradient."""
        if self.engine is None:
            raise ValueError("MD mode requires engine to be initialized")

        # Compute gradient using objective
        g = self.objective.grad(trade_w, phi_eff)

        # Center gradient for Euclidean geometry
        from skfolio.optimization.online._mirror_maps import EuclideanMirrorMap

        if isinstance(self.engine.map, EuclideanMirrorMap):
            g -= np.mean(g)

        # Add transaction cost subgradient
        if self.transaction_costs_arr is not None:
            prev = projector.config.previous_weights
            if prev is not None:
                prev_arr = np.asarray(prev, dtype=float)
                if prev_arr.shape == trade_w.shape:
                    delta = trade_w - prev_arr
                    g += self.transaction_costs_arr * np.sign(delta)

        projector.config.previous_weights = trade_w
        return self.engine.step(g)
