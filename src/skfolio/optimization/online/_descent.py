from abc import ABC, abstractmethod

import numpy as np
from scipy.special import softmax

from skfolio.optimization.online._projection import BaseProjector


class BaseDescent(ABC):
    @abstractmethod
    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def project(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OnlineGradientDescent(BaseDescent):
    """
    The Online Gradient Descent update rule.
    It updates the weights by learning rate eta.

    Parameters
    ----------
    projector : BaseProjector
        The projector to use.
    eta : float
        The learning rate.
    use_schedule : bool
        Whether to use a schedule for the learning rate.

    Notes
    -----
    Zinkevich (2003) Online convex programming and generalized infinitesimal gradient ascent
    """

    def __init__(
        self, projector: BaseProjector, eta: float = 1.0, use_schedule: bool = True
    ):
        self.projector = projector
        # Base learning-rate multiplier. If use_schedule, actual eta_t = eta / sqrt(t)
        self.eta = eta
        self.use_schedule = use_schedule
        self._t = 0

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._t += 1
        eta_t = self.eta / np.sqrt(self._t) if self.use_schedule else self.eta
        y = w - eta_t * g
        return y

    def project(self, y: np.ndarray) -> np.ndarray:
        return self.projector.project(y)


class OnlineFrankWolfe(BaseDescent):
    def __init__(self, projector: BaseProjector, gamma: float = 0.5):
        self.projector = projector
        self.gamma = gamma
        self._t = 0

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._t += 1
        n = w.shape[0]
        if n == 0:
            return w
        idx = int(np.argmin(g))
        s = np.zeros_like(w)
        s[idx] = 1.0
        gamma = self.gamma if self.gamma is not None else 2.0 / (self._t + 2.0)
        y = (1.0 - gamma) * w + gamma * s
        return y

    def project(self, y: np.ndarray) -> np.ndarray:
        return self.projector.project(y)


class EntropicMirrorDescent(BaseDescent):
    """
    Entropic mirror descent in probability/simplex geometry with projection.

    Uses a numerically stable softmax from SciPy instead of manual log-space,
    then projects via an AutoProjector.
    """

    def __init__(
        self, projector: BaseProjector, eta: float = 1.0, use_schedule: bool = True
    ):
        self.projector = projector
        # Base learning-rate multiplier. If use_schedule, actual eta_t = eta / sqrt(t)
        self.eta = float(eta)
        self.use_schedule = bool(use_schedule)
        self._t = 0

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        if w.size == 0:
            return w
        # Time-varying step-size per Hazan: eta_t = eta / sqrt(t)
        self._t += 1
        eta_t = self.eta / np.sqrt(self._t) if self.use_schedule else self.eta
        # Numerically stable softmax; ensure positivity
        y = np.log(np.maximum(w, 1e-16)) - eta_t * g
        return y

    def project(self, y: np.ndarray) -> np.ndarray:
        return self.projector.project(softmax(y, axis=0))


class EGTildeEntropicMirrorDescent(EntropicMirrorDescent):
    """
    Specialized Entropic Mirror Descent for EG-Tilde algorithm.

    Extends the standard EMD with EG-Tilde specific behavior:
    1. Uses time-varying eta from the associated loss function
    2. Applies portfolio mixing after the standard EMD update:
       tilde_x_{t+1} = (1 - alpha_t) * x_{t+1} + alpha_t/n

    The time-varying parameters (alpha_t, eta_t) are computed by the
    associated EGTildeLoss and accessed through the estimator.
    """

    def __init__(self, projector: BaseProjector, estimator, eta: float = 1.0):
        # Initialize parent without schedule since EG-Tilde uses custom eta
        super().__init__(projector, eta=eta, use_schedule=False)
        self.estimator = estimator

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Perform EG-Tilde step with time-varying eta and portfolio mixing."""
        if w.size == 0:
            return w

        # Get time-varying parameters from the loss function
        loss_method = getattr(self.estimator, "_loss_method", None)
        if loss_method and hasattr(loss_method, "_current_eta_t"):
            eta_t = loss_method._current_eta_t
            alpha_t = loss_method._current_alpha_t
        else:
            eta_t = self.eta
            alpha_t = 0.0

        if eta_t == 0.0:
            eta_t = self.eta

        # Standard EMD step with time-varying eta
        y = np.log(np.maximum(w, 1e-16)) - eta_t * g
        x_t_plus_1 = self.projector.project(softmax(y, axis=0))

        # EG-Tilde mixing step: tilde_x_{t+1} = (1 - alpha_t) * x_{t+1} + alpha_t/n
        if alpha_t == 0.0:
            return x_t_plus_1

        n = w.shape[0]
        uniform_weights = np.ones(n) / n
        tilde_x_t_plus_1 = (1.0 - alpha_t) * x_t_plus_1 + alpha_t * uniform_weights

        return tilde_x_t_plus_1


class OnlineNewtonStep(BaseDescent):
    """
    Online Newton Step (ONS) with a numerically safe Sherman–Morrison update.

    We maintain a positive definite matrix A_t that aggregates curvature
    information and take preconditioned (Newton-like) steps:

        A_t   = A_{t-1} + u_t u_t^T,   with A_0 = eps * I
        y     = w - eta_eff * (A_t^{-1} g_t)
        w^+   = project(y)

    where g_t is the gradient at the current point and u_t is a curvature
    direction. Following practical guidance (e.g., Numerical Recipes) and
    standard ONS literature, we normalize the curvature direction to avoid
    exploding condition numbers:

        u_t = g_t / max(1, ||g_t||_2)

    This keeps the Sherman-Morrison denominator well-scaled and yields more
    predictable progress on simple objectives (e.g., sum of squares).

    Notes on variants
    -----------------
    Two common implementations exist:
    1) Sherman-Morrison on the inverse (what we implement):
       A_t^{-1} ← A_{t-1}^{-1} - (A_{t-1}^{-1} u_t u_t^T A_{t-1}^{-1}) / (1 + u_t^T A_{t-1}^{-1} u_t)
       then y = w - eta * (A_t^{-1} g_t).
    2) Cholesky/factorization solve (more numerically robust):
       keep A_t and (optionally) a factor R (A_t = R^T R); solve A_t z = g_t,
       then y = w - eta * z. The projection can use the A_t-norm when available.

    This class follows (1) for efficiency but adopts a normalized curvature
    vector u_t to improve stability. The projection step is delegated to the
    provided projector. If the projector implements only Euclidean projection,
    the algorithm remains well-defined, though the theoretical guarantees for
    A-norm projections do not directly apply.

    Parameters
    ----------
    projector : BaseProjector
        Object providing .project(x: np.ndarray) -> np.ndarray
    eta : float
        Global step-size multiplier. For many exp-concave losses, eta=1 is
        standard; for general convex losses, smaller values are safer.
    eps : float
        Initialization jitter for A (A_0 = eps * I). Must be > 0.
    jitter : float
        Added to the Sherman–Morrison denominator if it is numerically tiny.
    recompute_every : Optional[int]
        If not None, periodically recompute A^{-1} from the accumulated A to
        correct numerical drift.
    auto_eta : bool
        If True, choose an effective eta following Hazan's guidance based on
        an online gradient-norm bound and the simplex diameter.
    D : float | None
        Diameter used by the auto step-size (defaults to sqrt(2) for simplex).
    """

    def __init__(
        self,
        projector: BaseProjector,
        eta: float = 1.0,
        eps: float = 1e-2,
        jitter: float = 1e-9,
        recompute_every: int | None = 50,
        auto_eta: bool = True,
        D: float | None = None,
        max_grad_norm: float | None = 20.0,
    ):
        self.projector = projector
        self.eta = float(eta)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.jitter = float(jitter)
        # internal state
        self._A_inv: np.ndarray | None = None  # inverse matrix
        self._A: np.ndarray | None = None  # optional full A for recompute
        self._t = 0
        self.recompute_every = recompute_every
        self.auto_eta = bool(auto_eta)
        # Euclidean diameter of simplex: sqrt(2) (used as default)
        self._D = float(D) if D is not None else float(np.sqrt(2.0))
        self._G_hat: float = 0.0
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

    def _ensure_matrices(self, d: int):
        """Initialize A and A_inv for dimension d."""
        if self._A_inv is None:
            self._A_inv = (1.0 / self.eps) * np.eye(d, dtype=float)
            if self.recompute_every is not None:
                self._A = self.eps * np.eye(d, dtype=float)

    def _sherman_morrison_update(self, g: np.ndarray):
        """
        Update A^{-1} with a rank-1 curvature using Sherman–Morrison.

        We normalize the curvature vector as u = g / max(1, ||g||) to keep
        the denominator well-conditioned. The full matrix A (if tracked) is
        updated consistently with the same u.
        """
        assert self._A_inv is not None
        # Build normalized curvature vector
        g_norm = float(np.linalg.norm(g))
        if g_norm > 1.0:
            u = g / g_norm
        else:
            u = g

        # A_inv_u = A_inv @ u
        A_inv_u = self._A_inv @ u  # shape (d,)
        denom = 1.0 + float(np.dot(u, A_inv_u))
        if denom <= self.jitter:
            # numerical safeguard: add jitter to denom
            denom = denom + self.jitter

        # rank-one correction: A_inv - (A_inv g)(A_inv g)^T / denom
        # Use broadcasting to avoid explicit outer product allocation
        self._A_inv -= np.multiply.outer(A_inv_u, A_inv_u) / denom

        # Update full A for occasional recompute, if needed
        if self._A is not None:
            self._A += np.multiply.outer(u, u)

    def _full_recompute_inverse(self):
        """
        Recompute A_inv from stored A (more numerically stable).
        Called periodically to correct numerical drift.
        """
        if self._A is None:
            return
        # Use np.linalg.inv with small regularization if needed
        try:
            self._A_inv = np.linalg.inv(self._A)
        except np.linalg.LinAlgError:
            # regularize slightly and invert
            reg = max(self.eps * 1e-6, 1e-12)
            self._A += reg * np.eye(self._A.shape[0])
            self._A_inv = np.linalg.inv(self._A)

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Perform one ONS step.

        Parameters
        ----------
        w : np.ndarray
            current point (e.g., current portfolio on simplex)
        g : np.ndarray
            gradient at w (same shape as w)

        Returns
        -------
        np.ndarray
            new (projected) iterate
        """
        if w.size == 0:
            return w

        if w.shape != g.shape:
            raise ValueError("w and g must have the same shape")

        d = w.shape[0]
        self._ensure_matrices(d)

        # Update step counter
        self._t += 1

        # Optionally clip gradient norm for stability
        if self.max_grad_norm is not None:
            gnorm_step = float(np.linalg.norm(g))
            if gnorm_step > self.max_grad_norm and gnorm_step > 0:
                g = (self.max_grad_norm / gnorm_step) * g

        # Sherman–Morrison inverse update with normalized curvature
        self._sherman_morrison_update(g)

        # Newton-like preconditioned step
        A_inv_g = self._A_inv @ g

        # Optionally choose eta following Hazan's ONS guidance via gamma
        if self.auto_eta:
            # Update gradient norm bound estimate
            gnorm = float(np.linalg.norm(g))
            if not np.isnan(gnorm) and gnorm > self._G_hat:
                self._G_hat = gnorm
            # gamma <= 0.5 * min{ 1/(G D), alpha } ; we ignore alpha and use 1/(G D)
            if self._G_hat > 0:
                gamma = 0.5 / (self._G_hat * self._D)
                # Use gamma directly as step-size and cap by self.eta for stability
                eta_eff = min(self.eta, gamma)
            else:
                eta_eff = self.eta
        else:
            eta_eff = self.eta

        # precaution: clip step magnitude optionally (not done here)
        y = w - eta_eff * A_inv_g

        # Periodic full recompute for numerical stability
        if self.recompute_every is not None and (self._t % self.recompute_every == 0):
            self._full_recompute_inverse()

        return y

    def project(self, y: np.ndarray) -> np.ndarray:
        return self.projector.project(y)

    def reset(self):
        """Reset internal accumulators."""
        self._A_inv = None
        self._A = None
        self._t = 0
