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


class SwordEntropicMirrorDescent(EntropicMirrorDescent):
    """
    Sword (Smoothness-aware online learning with dynamic regret) Entropic Mirror Descent.

    Implements adaptive step sizes for the Sword algorithm family from Zhao et al. (2020).
    The key innovation is using step sizes that adapt to problem difficulty:

    η_t ∝ √((1 + P_T + Q_T) / (1 + P_T))

    where Q_T can be:
    - V_T: gradient variation (Sword_var)
    - F_T: cumulative comparator loss (Sword_small)
    - min{V_T, F_T}: best of both (Sword_best)

    This achieves better dynamic regret bounds when the problem is "easy"
    (low gradient variation or low comparator loss).

    References
    ----------
    - Zhao, P., Zhang, Y.J., Zhang, L., & Zhou, Z.H. (2020). Dynamic Regret of Convex
      and Smooth Functions. NeurIPS 2020.
    """

    def __init__(
        self,
        projector: BaseProjector,
        estimator,
        eta: float = 1.0,
        sword_variant: str = "var",
    ):
        # Initialize parent without schedule since Sword uses adaptive eta
        super().__init__(projector, eta=eta, use_schedule=False)
        self.estimator = estimator
        self.sword_variant = sword_variant
        self._t = 0

        # Path-length tracking (P_T in the paper)
        self._previous_weights: np.ndarray | None = None
        self._path_length: float = 0.0

    def _compute_adaptive_step_size(self) -> float:
        """Compute adaptive step size based on tracked quantities."""
        loss_method = getattr(self.estimator, "_loss_method", None)

        if loss_method is None:
            return self.eta

        # Get tracked quantities from loss function
        if self.sword_variant == "var":
            # Sword_var: uses gradient variation V_T
            Q_T = getattr(loss_method, "_current_variation", 0.0)
        elif self.sword_variant == "small":
            # Sword_small: uses cumulative comparator loss F_T
            Q_T = getattr(loss_method, "_current_cumulative_loss", 0.0)
        elif self.sword_variant == "best":
            # Sword_best: uses min{V_T, F_T}
            V_T = getattr(loss_method, "_current_variation", 0.0)
            F_T = getattr(loss_method, "_current_cumulative_loss", 0.0)
            Q_T = min(V_T, F_T)
        else:
            Q_T = 0.0

        # Adaptive step size: η_t = η₀ * √((1 + P_T + Q_T) / (1 + P_T))
        # This reduces to η₀ when P_T and Q_T are large, but can be smaller when Q_T << P_T
        denominator = max(1.0 + self._path_length, 1e-8)
        numerator = 1.0 + self._path_length + Q_T

        adaptive_factor = np.sqrt(numerator / denominator)
        eta_t = self.eta * adaptive_factor

        return max(eta_t, 1e-8)  # Numerical stability

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Perform Sword EMD step with adaptive step size."""
        if w.size == 0:
            return w

        # Update path-length tracking (P_T in the paper)
        if self._previous_weights is not None and len(self._previous_weights) == len(w):
            weight_diff = w - self._previous_weights
            self._path_length += float(np.linalg.norm(weight_diff))

        self._previous_weights = w.copy()
        self._t += 1

        # Compute adaptive step size
        eta_t = self._compute_adaptive_step_size()

        # Standard EMD step with adaptive eta
        y = np.log(np.maximum(w, 1e-16)) - eta_t * g
        x_t_plus_1 = self.projector.project(softmax(y, axis=0))

        return x_t_plus_1


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
    Online Newton Step (ONS) with a numerically safe Sherman-Morrison update.

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
       A_t^{-1} <- A_{t-1}^{-1} - (A_{t-1}^{-1} u_t u_t^T A_{t-1}^{-1}) / (1 + u_t^T A_{t-1}^{-1} u_t)
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
        Added to the Sherman-Morrison denominator if it is numerically tiny.
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
        Update A^{-1} with a rank-1 curvature using Sherman-Morrison.

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

        # Sherman-Morrison inverse update with normalized curvature
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


class AdaGrad(BaseDescent):
    """
    AdaGrad (Adaptive Gradient) algorithm with coordinate-wise adaptive learning rates.

    AdaGrad adapts the learning rate for each coordinate based on the historical
    sum of squared gradients. This makes it particularly effective for sparse gradients
    and different-scaled coordinates.

    The update rule is:
        G_{t,i} = G_{t-1,i} + g_{t,i}²     (accumulate squared gradients)
        η_{t,i} = D_i / (√G_{t,i} + ε)     (adaptive learning rate)
        w_{t+1,i} = w_{t,i} - η_{t,i} * g_{t,i}

    where:
    - G_{t,i} is the accumulated squared gradient for coordinate i up to time t
    - D_i is the diameter bound for coordinate i (for hyperrectangles)
    - ε is a small constant for numerical stability (can be 0)
    - g_{t,i} is the gradient for coordinate i at time t

    Key properties:
    - Regret bound: D_∞ * ∑_{i=1}^d √(∑_{t=1}^T g_{t,i}²)
    - Adaptive to coordinate-wise gradient magnitudes
    - Works best with hyperrectangle domains (box constraints)
    - Scale-free when ε = 0

    Parameters
    ----------
    projector : BaseProjector
        The projector to use for constraint enforcement
    D : float or array-like, optional (default=√2)
        Diameter bound(s). If float, uses same bound for all coordinates.
        If array-like, should have length equal to problem dimension.
    eps : float, optional (default=1e-8)
        Small constant added to denominator for numerical stability.
        Set to 0 for truly scale-free updates.

    References
    ----------
    - Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for
      online learning and stochastic optimization. JMLR, 12, 2121-2159.
    - Blog post: https://parameterfree.com/2019/09/20/adaptive-algorithms-l-bounds-and-adagrad/
    - Orabona, F. (2019). A Modern Introduction to Online Learning, Chapter 4.
    """

    def __init__(
        self,
        projector: BaseProjector,
        D: float | np.ndarray | None = None,
        eps: float = 1e-8,
    ):
        self.projector = projector
        # Default diameter: √2 (appropriate for simplex)
        self.D = float(np.sqrt(2.0)) if D is None else D
        self.eps = float(eps)

        # Internal state
        self._G: np.ndarray | None = None  # Accumulated squared gradients
        self._t = 0

    def _ensure_accumulator(self, d: int):
        """Initialize gradient accumulator for dimension d."""
        if self._G is None:
            self._G = np.zeros(d, dtype=float)

        # Handle diameter parameter
        if np.isscalar(self.D) and self.D is not None:
            self._D_vec = np.full(d, np.asarray(self.D, dtype=float).item())
        elif self.D is not None:
            D_array = np.asarray(self.D, dtype=float)
            if D_array.shape == ():
                self._D_vec = np.full(d, D_array.item())
            elif len(D_array) == d:
                self._D_vec = D_array
            else:
                raise ValueError(
                    f"D has length {len(D_array)} but problem dimension is {d}"
                )
        else:
            # Default: sqrt(2) for simplex diameter
            self._D_vec = np.full(d, np.sqrt(2.0))

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Perform one AdaGrad step with coordinate-wise adaptive learning rates.

        Parameters
        ----------
        w : np.ndarray
            Current point
        g : np.ndarray
            Gradient at current point

        Returns
        -------
        np.ndarray
            Updated point (before projection)
        """
        if w.size == 0:
            return w

        if w.shape != g.shape:
            raise ValueError("w and g must have the same shape")

        d = w.shape[0]
        self._ensure_accumulator(d)

        # Update time counter
        self._t += 1

        # Accumulate squared gradients: G_t = G_{t-1} + g_t²
        self._G += g * g

        # Compute coordinate-wise adaptive learning rates
        # η_{t,i} = D_i / (√G_{t,i} + ε)
        assert self._G is not None  # Should be initialized by _ensure_accumulator
        sqrt_G_plus_eps = np.sqrt(self._G + self.eps)
        eta_t = self._D_vec / sqrt_G_plus_eps

        # AdaGrad update: w_{t+1} = w_t - η_t ⊙ g_t (coordinate-wise)
        y = w - eta_t * g

        return y

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project updated point back to feasible set."""
        return self.projector.project(y)

    def reset(self):
        """Reset internal accumulators."""
        self._G = None
        self._t = 0


class AdaBARRONS(BaseDescent):
    """
    Adaptive Barrier-Regularized Online Newton Step (Ada-BARRONS).

    Combines ONS with log-barrier regularization and adaptive step-size
    control to achieve polylog regret in adversarial OCO settings.

    References
    ----------
    - Hazan, Kale, Warmuth (2016+)
    - Li & Hoi, "Online Portfolio Selection: Principles and Algorithms"
    - Recent Ada-BARRONS papers (2022+)
    """

    def __init__(
        self,
        projector: BaseProjector,
        eta: float = 1.0,
        eps: float = 1e-2,
        jitter: float = 1e-9,
        lambda_init: float = 0.1,
        backtracking: bool = True,
        max_backtrack: int = 10,
    ):
        self.projector = projector
        # ONS core
        self._ons = OnlineNewtonStep(
            projector=projector,
            eta=eta,
            eps=eps,
            jitter=jitter,
            auto_eta=False,  # we control eta adaptively
        )
        # Barrier weights λ_t
        self._lambda = lambda_init
        self.backtracking = backtracking
        self.max_backtrack = max_backtrack

        # iteration counter
        self._t = 0

    def _barrier_grad(self, w: np.ndarray) -> np.ndarray:
        """Compute log-barrier gradient contribution."""
        return -self._lambda / np.maximum(w, 1e-12)

    def step(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Perform Ada-BARRONS update.

        Parameters
        ----------
        w : np.ndarray
            Current iterate
        g : np.ndarray
            Gradient of the loss at w
        """
        if w.size == 0:
            return w

        self._t += 1

        # Augment gradient with barrier term
        g_bar = g + self._barrier_grad(w)

        # Try candidate updates with backtracking
        eta_eff = self._ons.eta
        for _ in range(self.max_backtrack if self.backtracking else 1):
            # Temporarily override eta
            self._ons.eta = eta_eff
            y = self._ons.step(w, g_bar)
            w_new = self.project(y)

            if not self.backtracking:
                break

            # Stability check: ensure positivity & Armijo-like condition
            if np.all(w_new > 1e-12):
                # TODO: optionally implement a proper Armijo check with loss feedback
                break
            eta_eff *= 0.5  # shrink step-size and retry

        # restore base eta
        self._ons.eta = eta_eff
        return w_new

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project while keeping strict positivity."""
        w = self.projector.project(y)
        return np.clip(w, 1e-12, None) / np.sum(np.clip(w, 1e-12, None))
