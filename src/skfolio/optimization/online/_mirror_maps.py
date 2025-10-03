# _mirror_maps.py
"""
Mirror maps for online convex optimization.

Implements Legendre-type mirror potentials and their forward/inverse mappings
for use in mirror descent and FTRL algorithms.
Mirror maps play the rôle of the 'composite regulariser' ψ in COMID.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import line_search
from scipy.special import softmax


class _DampedNewtonInverse:
    r"""
    Damped Newton solver for coupled composite mirror-map inversion.

    Solves the stationarity condition for the inverse of a composite potential:

    .. math::
        z = \nabla \psi(w) = -D \oslash w + M w,

    where :math:`D \ge 0` are barrier coefficients, :math:`M = \text{diag}(a) + A`
    combines diagonal and full quadratic terms, and :math:`w > 0` (positivity).

    This is equivalent to minimizing the strictly convex objective:

    .. math::
        g(w) = -D^\top \log(w) + \frac{1}{2} w^\top M w - z^\top w.

    The residual :math:`f(w) = \nabla g(w) = -D \oslash w + M w - z` and
    Jacobian :math:`J(w) = \text{diag}(D \oslash w^2) + M` (SPD) define the
    Newton system. The solver uses positivity-preserving backtracking and
    scipy's Wolfe line search for robust convergence.

    Parameters
    ----------
    D : ndarray of shape (d,)
        Barrier coefficients (non-negative).
    a : ndarray of shape (d,)
        Diagonal quadratic coefficients.
    A : ndarray of shape (d, d)
        Full quadratic coupling matrix.
    eps : float, default=1e-16
        Numerical stability floor for positivity and conditioning.

    References
    ----------
    .. [1] McMahan, H. B. (2011). Follow-the-Regularized-Leader and Mirror
       Descent: Equivalence Theorems and Implicit Updates. arXiv:1009.3240.
       Section 3: implicit composite updates with quadratic regularization.

    Notes
    -----
    The implementation follows the "implicit composite FTRL" framework where
    the non-smooth barrier term and smooth quadratic term are jointly inverted
    via the first-order optimality condition.
    """

    def __init__(self, D: np.ndarray, a: np.ndarray, A: np.ndarray, eps: float = 1e-16):
        self.D = np.asarray(D, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.eps = float(eps)

        d = self.D.shape[0]
        # Build M = diag(a) + A with small regularization for conditioning
        self.M = self.A.copy()
        idx = np.arange(d)
        self.M[idx, idx] += self.a + 1e-12

    def objective(self, w: np.ndarray) -> float:
        """Objective g(w) = -D^T log(w) + (1/2) w^T M w - z^T w."""
        w_safe = np.maximum(w, self.eps)
        log_barrier = -self.D @ np.log(w_safe)
        quadratic = 0.5 * (w @ (self.M @ w))
        linear = -self._z @ w
        return log_barrier + quadratic + linear

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Gradient (residual) f(w) = -D/w + M w - z."""
        w_safe = np.maximum(w, self.eps)
        return -self.D / w_safe + self.M @ w - self._z

    def hessian(self, w: np.ndarray) -> np.ndarray:
        """Hessian (Jacobian) J(w) = diag(D / w^2) + M."""
        w_safe = np.maximum(w, self.eps)
        H = self.M.copy()
        idx = np.arange(len(w))
        H[idx, idx] += self.D / (w_safe**2)
        return H

    def solve(
        self, z: np.ndarray, max_iter: int = 100, tol: float = 1e-12
    ) -> np.ndarray:
        """
        Solve for w such that -D/w + M w = z, w > 0.

        Parameters
        ----------
        z : ndarray of shape (d,)
            Target dual variable.
        max_iter : int, default=100
            Maximum Newton iterations.
        tol : float, default=1e-12
            Convergence tolerance on residual norm.

        Returns
        -------
        w : ndarray of shape (d,)
            Solution (positive, normalized to simplex by caller).
        """
        self._z = z  # store for objective/gradient
        d = z.shape[0]

        # Initialize with positive guess
        w = np.ones(d) / d
        # Better initialization if barrier dominates
        try:
            guess = -self.D / np.minimum(z, -self.eps)
            if np.all(np.isfinite(guess)) and np.all(guess > 0):
                w = guess
        except Exception:
            pass

        for _ in range(max_iter):
            grad = self.gradient(w)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tol:
                break

            # Solve J(w) delta = -grad
            try:
                H = self.hessian(w)
                delta = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                # Regularize if singular
                H_reg = H + 1e-10 * np.eye(d)
                delta = np.linalg.solve(H_reg, -grad)

            # Positivity-preserving step size: ensure w + t*delta > 0
            t_max = 1.0
            neg_idx = delta < 0
            if np.any(neg_idx):
                # For coordinates where delta < 0, enforce w_i + t*delta_i >= eps
                # => t <= (w_i - eps) / (-delta_i)
                ratios = (w[neg_idx] - self.eps) / (-delta[neg_idx] + 1e-32)
                t_max = min(1.0, 0.99 * np.min(ratios))

            # Scipy Wolfe line search
            # line_search expects: f, myfprime, xk, pk, gfk
            ls_result = line_search(
                f=self.objective,
                myfprime=lambda w_vec: self.gradient(w_vec),
                xk=w,
                pk=delta,
                gfk=grad,
                old_fval=self.objective(w),
                c1=1e-4,
                c2=0.9,
                amax=t_max,
            )

            alpha_star = ls_result[0]

            if alpha_star is not None and alpha_star > 0:
                w_new = np.maximum(w + alpha_star * delta, self.eps)
            else:
                # Fallback: simple backtracking Armijo
                alpha = t_max
                c1 = 1e-4
                f_current = self.objective(w)
                directional_deriv = grad @ delta

                for _ in range(20):
                    w_trial = np.maximum(w + alpha * delta, self.eps)
                    f_trial = self.objective(w_trial)
                    if f_trial <= f_current + c1 * alpha * directional_deriv:
                        w_new = w_trial
                        break
                    alpha *= 0.5
                else:
                    # Accept small step if line search fails completely
                    w_new = np.maximum(w + 0.01 * t_max * delta, self.eps)

            w = w_new

        return np.maximum(w, self.eps)


class BaseMirrorMap(ABC):
    r"""
    Abstract base class for mirror maps (Legendre-type potentials).

    A mirror map is a strictly convex, differentiable potential :math:`\\psi`
    with gradient :math:`\\nabla \\psi : \\mathcal{W} \\to \\mathcal{Z}` a bijection
    between the interior of the primal domain :math:`\\mathcal{W}` and the dual space
    :math:`\\mathcal{Z}`. Its inverse mapping is :math:`\\nabla \\psi^*`, the gradient
    of the Fenchel conjugate :math:`\\psi^*`.

    For sums of potentials :math:`\\psi = \\sum_k \\psi_k`, we always have
    :math:`\\nabla \\psi(w) = \\sum_k \\nabla \\psi_k(w)`. However, the inverse
    generally does not decompose:
    :math:`\\nabla (\\psi_1 + \\psi_2)^*(z) \\neq \\nabla \\psi_1^*(z) + \\nabla \\psi_2^*(z)`.

    References
    ----------
    .. [1] Rockafellar, R. T. (1970). Convex Analysis. Princeton University Press.
    .. [2] Beck, A., & Teboulle, M. (2003). Mirror Descent and Nonlinear Projected
       Subgradient Methods for Convex Optimization. SIAM J. Optim., 13(1), 188–205.
    """

    @abstractmethod
    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        r"""
        Forward map: primal → dual.

        Parameters
        ----------
        w : ndarray of shape (d,)
            Primal variable in the interior of the domain.

        Returns
        -------
        ndarray of shape (d,)
            :math:`z = \\nabla \\psi(w)`.
        """
        pass

    @abstractmethod
    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        r"""
        Inverse map: dual → primal.

        Parameters
        ----------
        z : ndarray of shape (d,)
            Dual variable.

        Returns
        -------
        ndarray of shape (d,)
            :math:`w = \\nabla \\psi^*(z)`.

        Notes
        -----
        For sums of mirror maps, the inverse generally cannot be obtained by
        summing the individual inverses. See :class:`CompositeMirrorMap`.
        """
        pass

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """
        Optional geometry-specific normalization of :math:`w`.

        Parameters
        ----------
        w : ndarray
            Primal weights.

        Returns
        -------
        ndarray
            Projected weights. Default is identity.
        """
        return w

    def __add__(self, other: "BaseMirrorMap") -> "CompositeMirrorMap":
        r"""
        Compose two mirror maps by summing their potentials.

        Returns
        -------
        CompositeMirrorMap
            A composite map with flattened components.

        Notes
        -----
        While :math:`\\nabla` of the sum is the sum of gradients, the inverse is not
        additive and generally requires a dedicated solver.
        """
        return CompositeMirrorMap([self, other])


class DynamicMirrorMap(BaseMirrorMap):
    """
    A mirror map whose parameters/state can evolve online.

    Examples include AdaGrad-style diagonal adaptivity or Online Newton Step
    that accumulates second-order statistics.

    Methods
    -------
    ensure_dim(d)
        Initialize internal state with dimension d if needed.
    update_state(g)
        Update internal state from a new gradient or statistic g.
    """

    def ensure_dim(self, d: int) -> None:
        """Initialize internal dimension if needed."""
        pass

    def update_state(self, g: np.ndarray) -> None:
        """Update internal accumulators using gradient g."""
        pass

    def grad_psi_after(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Forward map with updated state (optional hook for online methods)."""
        return self.grad_psi(w)

    def grad_psi_star_after(self, z: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Inverse map with updated state (optional hook for online methods)."""
        return self.grad_psi_star(z)


# =============================================================================
# Atomic mirror maps
# =============================================================================


class EuclideanMirrorMap(BaseMirrorMap):
    """Euclidean mirror map for Online Gradient Descent."""

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        return w

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        return z

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Geometry-specific normalization (e.g., ensure positivity)."""
        return w


class EntropyMirrorMap(BaseMirrorMap):
    """Entropy mirror map for Entropic Mirror Descent."""

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        # Using log(w) directly as per existing EntropicMirrorDescent implementation
        # This corresponds to a potential function that is slightly different from
        # standard Shannon entropy but results in the multiplicative update rule.
        return np.log(np.maximum(w, 1e-16)) + 1

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        # The dual-to-primal mapping for entropy is the softmax function.
        return softmax(z - 1.0, axis=0)

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Ensure unit sum for simplex."""
        return w / np.sum(w)


class LogBarrierMap(DynamicMirrorMap):
    r"""
    Log-barrier potential.

    .. math::
        \psi(w) = -\sum_{i=1}^d D_i \log(w_i), \qquad w_i > 0.

    Gradient and inverse:

    .. math::
        \nabla \psi(w) = - D \oslash w, \qquad
        \nabla \psi^*(z) = - D \oslash z,

    where :math:`\oslash` is coordinate-wise division.

    Parameters
    ----------
    barrier_coef : float, default=1.0
        Scalar coefficient for the barrier (used if not dynamic).
    eps : float, default=1e-16
        Numerical stability floor.

    References
    ----------
    .. [1] Nesterov, Y., & Nemirovskii, A. (1994). Interior-Point Polynomial
       Algorithms in Convex Programming. SIAM.
    """

    def __init__(self, barrier_coef: float = 1.0, eps: float = 1e-16):
        self.barrier_coef = barrier_coef
        self.eps = eps
        self._D = np.array([], dtype=float)

    def _ensure(self, d: int):
        if self._D.size == 0:
            self._D = np.zeros(d, dtype=float)

    def ensure_dim(self, d: int) -> None:
        self._ensure(d)

    def update_state(self, g: np.ndarray) -> None:
        """Update barrier coefficients (example: accumulate magnitudes)."""
        self._ensure(g.shape[0])
        # Accumulate gradient magnitudes (one choice among many in OCO)
        self._D += np.abs(g)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        """Gradient ∇ψ(w) = -(barrier_coef + D) / w."""
        self._ensure(w.shape[0])
        D_total = self.barrier_coef + self._D
        return -D_total / np.maximum(w, self.eps)

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        """Inverse: w = -(barrier_coef + D) / z (requires z < 0)."""
        self._ensure(z.shape[0])
        D_total = self.barrier_coef + self._D
        z_safe = np.minimum(z, -self.eps)
        w = -D_total / z_safe
        return np.maximum(w, self.eps)

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Normalize to simplex."""
        s = np.sum(w)
        return w / s if s > self.eps else w


class TsallisMirrorMap(BaseMirrorMap):
    """Tsallis mirror. q in (0,1) typically."""

    def __init__(self, q: float = 0.7):
        self.q = q
        self._is_entropic_limit = abs(q - 1.0) < 1e-14
        self._entropy_map = EntropyMirrorMap() if self._is_entropic_limit else None

    def grad_psi(self, w):
        if self._is_entropic_limit:
            return self._entropy_map.grad_psi(w)

        q = self.q
        # Corresponds to potential 1/(1-q) * (sum(w_i^q) - 1), grad is q/(1-q) * w_i^(q-1)
        return self.q * (w ** (q - 1.0)) / (1.0 - q)

    def grad_psi_star(self, z):
        if self._is_entropic_limit:
            return self._entropy_map.grad_psi_star(z)

        q = self.q
        # Inverse map via Fenchel conjugate
        z_scaled = z * (1.0 - q) / q
        w = np.power(np.maximum(z_scaled, 1e-16), 1.0 / (q - 1.0))
        return w

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 1e-16)
        return w / np.sum(w)


class AdaptiveMahalanobisMap(DynamicMirrorMap):
    """
    Adaptive diagonal quadratic (AdaGrad-style).

    ψ(w) = (1/2) w^T diag(H) w, where H_ii = sqrt(sum g_{s,i}^2 + eps).
    """

    def __init__(self, d: int | None = None, eps: float = 1e-8):
        self.eps = eps
        self._Gsq = np.array([], dtype=float)

    def _ensure(self, d: int):
        if self._Gsq.size == 0:
            self._Gsq = np.zeros(d, dtype=float)

    def ensure_dim(self, d: int) -> None:
        self._ensure(d)

    def H_diag(self) -> np.ndarray:
        # Pre-geometry: keep absolute eps inside sqrt for numerical safety
        return np.sqrt(self._Gsq + self.eps)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        h = self.H_diag()
        return h * w

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        self._ensure(z.shape[0])
        h = self.H_diag()
        return z / h

    # Aggregated 'after' geometry (scale-homogeneous): NO eps here
    def _H_after(self, g: np.ndarray) -> np.ndarray:
        Gsq_after = self._Gsq + g * g
        return np.sqrt(Gsq_after + self.eps)

    def grad_psi_after(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        h_after = self._H_after(g)
        return h_after * w

    def grad_psi_star_after(self, z: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._ensure(z.shape[0])
        h_after = self._H_after(g)
        return z / h_after

    def update_state(self, g: np.ndarray) -> None:
        self._ensure(g.shape[0])
        self._Gsq += g * g


class AdaptiveLogBarrierMap(DynamicMirrorMap):
    """
    Adaptive log-barrier with gradient-magnitude adaptation.

    .. deprecated::
        This mirror map is no longer used by any FTRL strategy. For Ada-BARRONS,
        use :class:`AdaBarronsBarrierMap` instead, which correctly adapts based
        on weight proximity (1 - w) rather than gradient magnitude |g|.
        This class is kept for backward compatibility with existing tests.

    ψ(w) = -sum D_i log(w_i), where D accumulates gradient magnitudes |g|.
    """

    def __init__(self, d: int | None = None, eps: float = 1e-12):
        self.eps = eps
        self._D = np.array([], dtype=float)

    def _ensure(self, d):
        if self._D.size == 0:
            self._D = np.zeros(d, dtype=float)

    def ensure_dim(self, d: int) -> None:
        self._ensure(d)

    def update_state(self, g: np.ndarray) -> None:
        self._ensure(g.shape[0])
        self._D += np.abs(g)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        return -self._D / np.maximum(w, self.eps)

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        self._ensure(z.shape[0])
        w = np.zeros_like(z)
        mask = self._D > 0
        # z should be negative coordinatewise for positive w
        w[mask] = -self._D[mask] / np.minimum(z[mask], -1e-12)
        # For D_i=0, assign tiny positive mass; project_geom will normalize
        w[~mask] = 1e-16
        return np.clip(w, self.eps, None)

    def _D_after(self, g: np.ndarray) -> np.ndarray:
        self._ensure(g.shape[0])
        return self._D + np.abs(g)

    def grad_psi_after(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        D_after = self._D_after(g)
        return -D_after / np.maximum(w, self.eps)

    def grad_psi_star_after(self, z: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._ensure(z.shape[0])
        D_after = self._D_after(g)
        w = np.zeros_like(z)
        mask = D_after > 0
        w[mask] = -D_after[mask] / np.minimum(z[mask], -1e-12)
        w[~mask] = 1e-16
        return np.clip(w, self.eps, None)

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        return w / np.sum(w)


class AdaptiveVariationMap(DynamicMirrorMap):
    """
    Variation-adaptive map (SWORD).

    ψ(w) = (1/2) w^T diag(H) w, where H_ii = sqrt(sum (g_t - g_{t-1})^2 + eps).
    """

    def __init__(self, d: int | None = None, eps: float = 1e-12):
        self.eps = eps
        self._DeltaSq = np.array([], dtype=float)
        self._g_last = None

    def _ensure(self, d: int):
        if self._DeltaSq.size == 0:
            self._DeltaSq = np.zeros(d, dtype=float)

    def ensure_dim(self, d: int) -> None:
        self._ensure(d)

    def _H_diag_current(self) -> np.ndarray:
        return np.sqrt(self._DeltaSq + self.eps)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        h = self._H_diag_current()
        return h * w

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        self._ensure(z.shape[0])
        h = self._H_diag_current()
        return z / h

    def _H_diag_after(self, g: np.ndarray) -> np.ndarray:
        self._ensure(g.shape[0])
        if self._g_last is None:
            delta = np.zeros_like(g)
        else:
            delta = g - self._g_last
        DeltaSq_after = self._DeltaSq + delta * delta
        return np.sqrt(DeltaSq_after + self.eps)

    def grad_psi_after(self, w: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        h_after = self._H_diag_after(g)
        return h_after * w

    def grad_psi_star_after(self, z: np.ndarray, g: np.ndarray) -> np.ndarray:
        self._ensure(z.shape[0])
        h_after = self._H_diag_after(g)
        return z / h_after

    def update_state(self, g: np.ndarray) -> None:
        self._ensure(g.shape[0])
        if self._g_last is None:
            delta = np.zeros_like(g)
        else:
            delta = g - self._g_last
        self._DeltaSq += delta * delta
        self._g_last = g.copy()


# =============================================================================
# New compositional mirror maps for Ada-BARRONS
# =============================================================================


class EuclideanMap(DynamicMirrorMap):
    r"""
    Euclidean potential (for composition).

    .. math::
        \psi(w) = \frac{d \cdot d\_coef}{2} \|w\|_2^2

    Used in Ada-BARRONS compositions.
    """

    def __init__(self, d_coef: float = 1.0):
        self.d_coef = float(d_coef)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        d = w.shape[0]
        return self.d_coef * d * w

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        """Only valid when used alone! In composition, inverse is coupled."""
        d = z.shape[0]
        return z / (self.d_coef * d)


class DiagonalQuadraticMap(DynamicMirrorMap):
    r"""
    Diagonal second-order potential.

    .. math::
        \psi(w) = \frac{\beta}{2} \sum_i H_{ii} w_i^2

    where :math:`H_{ii}` accumulates gradient squared statistics.
    """

    def __init__(self, beta: float = 0.1, eps: float = 1e-16):
        self.beta = float(beta)
        self.eps = float(eps)
        self._H_diag = np.array([], dtype=float)

    def _ensure(self, d: int):
        if self._H_diag.size == 0:
            self._H_diag = np.zeros(d, dtype=float)

    def ensure_dim(self, d: int) -> None:
        self._ensure(d)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        return self.beta * self._H_diag * w

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        """Only valid when used alone! In composition, inverse is coupled."""
        self._ensure(z.shape[0])
        return z / (self.beta * self._H_diag + self.eps)

    def update_state(self, g: np.ndarray) -> None:
        self._ensure(g.shape[0])
        self._H_diag += g * g


class FullQuadraticMap(DynamicMirrorMap):
    r"""
    Full second-order potential (Online Newton Step).

    .. math::
        \psi(w) = \frac{\beta}{2} w^\top A w

    where :math:`A` accumulates outer products of gradients.

    References
    ----------
    .. [1] Hazan, E., Agarwal, A., & Kale, S. (2007).
       Logarithmic regret algorithms for online convex optimization.
       Machine Learning, 69, 169–192.
    """

    def __init__(self, beta: float = 0.1, eps: float = 1e-12):
        self.beta = float(beta)
        self.eps = float(eps)
        self._A = np.array([], dtype=float)

    def _ensure(self, d: int):
        if self._A.size == 0:
            self._A = np.zeros((d, d), dtype=float)

    def ensure_dim(self, d: int) -> None:
        self._ensure(d)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        self._ensure(w.shape[0])
        return self.beta * (self._A @ w)

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        """Only valid when used alone! In composition, completely wrong."""
        self._ensure(z.shape[0])
        A_reg = self.beta * self._A + self.eps * np.eye(len(z))
        return np.linalg.solve(A_reg, z)

    def update_state(self, g: np.ndarray) -> None:
        self._ensure(g.shape[0])
        self._A += np.outer(g, g)


class CompositeMirrorMap(DynamicMirrorMap):
    r"""
    Composite mirror map :math:`\psi = \sum_k \psi_k`.

    Forward composition is exact:

    .. math::
        \nabla \psi(w) = \sum_k \nabla \psi_k(w).

    The inverse is the unique solution of the stationarity condition:

    .. math::
        z = \sum_k \nabla \psi_k(w),

    which for barrier + diagonal + full quadratic is:

    .. math::
        z = -D \oslash w + \text{diag}(a) w + A w.

    Parameters
    ----------
    components : list of BaseMirrorMap
        Components to sum. Constructor flattens nested composites.
    eps : float, default=1e-16
        Numerical stability floor.

    Notes
    -----
    - Diagonal case (separable): Closed-form per-coordinate quadratic inversion.
    - Coupled case: Damped Newton on strictly convex objective with SPD Jacobian.
    """

    def __init__(self, components: list[BaseMirrorMap], eps: float = 1e-16):
        self.eps = float(eps)
        self.components_ = self._flatten(components)

    @staticmethod
    def _flatten(comps: list[BaseMirrorMap]) -> list[BaseMirrorMap]:
        """Recursively flatten nested composites."""
        out: list[BaseMirrorMap] = []
        for c in comps:
            if isinstance(c, CompositeMirrorMap):
                out.extend(c.components_)
            else:
                out.append(c)
        return out

    def __add__(self, other: BaseMirrorMap) -> "CompositeMirrorMap":
        """Flatten on addition."""
        new_comps = self.components_.copy()
        if isinstance(other, CompositeMirrorMap):
            new_comps.extend(other.components_)
        else:
            new_comps.append(other)
        return CompositeMirrorMap(new_comps, eps=self.eps)

    def ensure_dim(self, d: int) -> None:
        for c in self.components_:
            if isinstance(c, DynamicMirrorMap):
                c.ensure_dim(d)

    def update_state(self, g: np.ndarray) -> None:
        for c in self.components_:
            if isinstance(c, DynamicMirrorMap):
                c.update_state(g)

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        """Forward: sum of gradients."""
        z = np.zeros_like(w, dtype=float)
        for c in self.components_:
            z += c.grad_psi(w)
        return z

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        """Inverse: solve stationarity condition."""
        has_coupling = any(isinstance(c, FullQuadraticMap) for c in self.components_)
        if not has_coupling and self._is_diagonal_composite():
            return self._solve_diagonal_inverse(z)
        else:
            return self._solve_coupled_inverse(z)

    def _is_diagonal_composite(self) -> bool:
        """Check if all components are separable."""
        allowed = (
            AdaBarronsBarrierMap,
            LogBarrierMap,
            EuclideanMap,
            DiagonalQuadraticMap,
            AdaptiveMahalanobisMap,
            AdaptiveVariationMap,
        )
        return all(isinstance(c, allowed) for c in self.components_)

    def _collect_coefficients(
        self, d: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect barrier D, diagonal a, and coupling A from all components.

        Returns
        -------
        D : ndarray (d,)
            Barrier coefficients.
        a : ndarray (d,)
            Diagonal quadratic coefficients.
        A : ndarray (d, d)
            Full quadratic coupling matrix.
        """
        D = np.zeros(d, dtype=float)
        a = np.zeros(d, dtype=float)
        A = np.zeros((d, d), dtype=float)

        for c in self.components_:
            match c:
                case AdaBarronsBarrierMap():
                    # Ada-BARRONS barrier: D_i = barrier_coef + alpha * proximity_sum_i
                    D += c._get_D()
                case LogBarrierMap():
                    # Static barrier + dynamic D
                    D += c.barrier_coef + getattr(c, "_D", np.zeros(d))
                case EuclideanMap():
                    # Euclidean adds d*d_coef to each diagonal element
                    a += c.d_coef * d
                case DiagonalQuadraticMap():
                    # Diagonal quadratic adds beta * H_diag to diagonal
                    a += c.beta * getattr(c, "_H_diag", np.zeros(d))
                case AdaptiveMahalanobisMap():
                    # AdaptiveMahalanobis: ψ(w) = (1/2) w^T diag(H) w, H = sqrt(Gsq + eps)
                    a += c.H_diag()
                case AdaptiveVariationMap():
                    # AdaptiveVariation: ψ(w) = (1/2) w^T diag(H) w, H = sqrt(DeltaSq + eps)
                    a += c._H_diag_current()
                case FullQuadraticMap():
                    # Full quadratic coupling
                    A += c.beta * getattr(c, "_A", np.zeros((d, d)))
                case _:
                    raise ValueError(f"Unsupported component type: {type(c)}")

        return D, a, A

    def _solve_diagonal_inverse(self, z: np.ndarray) -> np.ndarray:
        """
        Closed-form diagonal inversion.

        Solves per-coordinate: a_i w_i^2 - z_i w_i - D_i = 0, w_i > 0.
        """
        d = z.shape[0]
        D, a, A = self._collect_coefficients(d)

        if np.linalg.norm(A, ord=2) > self.eps:
            raise RuntimeError("Diagonal solver called with nonzero coupling.")

        w = np.empty(d, dtype=float)

        # Coordinates with a_i > 0: quadratic formula
        pos = a > self.eps
        if np.any(pos):
            a_pos = a[pos]
            z_pos = z[pos]
            D_pos = D[pos]
            disc = z_pos * z_pos + 4.0 * a_pos * D_pos
            disc = np.maximum(disc, 0.0)
            w[pos] = (z_pos + np.sqrt(disc)) / (2.0 * a_pos)

        # Coordinates with a_i ≈ 0: pure barrier w_i = -D_i / z_i
        zero = ~pos
        if np.any(zero):
            z_zero = np.minimum(z[zero], -self.eps)
            D_zero = D[zero]
            w[zero] = -D_zero / z_zero

        return np.maximum(w, self.eps)

    def _solve_coupled_inverse(self, z: np.ndarray) -> np.ndarray:
        """
        Damped Newton solver for coupled system.

        Uses the dedicated _DampedNewtonInverse solver with scipy line search.
        """
        d = z.shape[0]
        D, a, A = self._collect_coefficients(d)

        solver = _DampedNewtonInverse(D, a, A, eps=self.eps)
        return solver.solve(z)

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Apply geometry projections from components."""
        for c in self.components_:
            w = c.project_geom(w)
        return w


# =============================================================================
# Ada-BARRONS Specialized Components
# =============================================================================


class AdaBarronsBarrierMap(DynamicMirrorMap):
    r"""
    Ada-BARRONS log-barrier with weight-proximity adaptation.

    Mirror potential:

    .. math::
        \psi(w) = -\sum_{i=1}^d D_{t,i} \log(w_i),

    where the barrier coefficients adapt as:

    .. math::
        D_{t,i} = \text{barrier\_coef} + \alpha \sum_{s=1}^{t-1} (1 - w_{s,i}).

    This captures the **proximity of weights to boundaries** rather than
    gradient magnitude, which is crucial for Ada-BARRONS' regret bound.

    Parameters
    ----------
    barrier_coef : float, default=1.0
        Initial (static) barrier coefficient.
    alpha : float, default=1.0
        Adaptation rate for proximity accumulation.
    eps : float, default=1e-16
        Numerical stability floor.

    References
    ----------
    .. [1] Gaillard, P., Gerchinovitz, S., Hebiri, M., & Lugosi, G. (2022).
       Adapting to Misspecification in Contextual Bandits with Offline Regression
       Oracles. arXiv:2202.07574.

    Notes
    -----
    Ada-BARRONS adapts the barrier based on how close each weight w_i has been
    to its boundaries (0 and 1). The update rule accumulates (1 - w_i) across
    rounds, increasing the barrier strength for weights that have remained
    closer to 1 (further from 0).

    Unlike `LogBarrierMap.update_state(g)` which accumulates gradient magnitude,
    this map's `update_state(w)` takes the **weight vector** and accumulates
    boundary proximity.
    """

    def __init__(
        self, barrier_coef: float = 1.0, alpha: float = 1.0, eps: float = 1e-16
    ):
        self.barrier_coef = float(barrier_coef)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._proximity_sum = np.array([], dtype=float)

    def _ensure(self, d: int):
        if self._proximity_sum.size == 0:
            self._proximity_sum = np.zeros(d, dtype=float)

    def ensure_dim(self, d: int) -> None:
        self._ensure(d)

    def update_state(self, w: np.ndarray) -> None:
        """
        Update barrier coefficients based on weight proximity to boundaries.

        Parameters
        ----------
        w : ndarray of shape (d,)
            Current weight vector (not gradient!).

        Notes
        -----
        Accumulates (1 - w_i) for each coordinate, increasing barrier strength
        for weights that stay closer to 1.
        """
        self._ensure(w.shape[0])
        # Accumulate proximity to upper boundary: (1 - w_i)
        self._proximity_sum += 1.0 - w

    def _get_D(self) -> np.ndarray:
        """Current barrier coefficients: D_i = barrier_coef + α * proximity_sum_i."""
        return self.barrier_coef + self.alpha * self._proximity_sum

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        r"""Gradient: :math:`\nabla \psi(w) = -D \oslash w`."""
        self._ensure(w.shape[0])
        D = self._get_D()
        return -D / np.maximum(w, self.eps)

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        r"""Inverse: :math:`w = -D \oslash z` (requires :math:`z_i < 0`)."""
        self._ensure(z.shape[0])
        D = self._get_D()
        z_safe = np.minimum(z, -self.eps)
        w = -D / z_safe
        return np.maximum(w, self.eps)

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Normalize to simplex."""
        s = np.sum(w)
        return w / s if s > self.eps else w


def make_ada_barrons_mirror_map(
    d: int,
    barrier_coef: float = 1.0,
    alpha: float = 1.0,
    euclidean_coef: float = 1.0,
    beta: float = 0.1,
    eps: float = 1e-16,
) -> CompositeMirrorMap:
    r"""
    Construct the Ada-BARRONS composite mirror map.

    The Ada-BARRONS mirror potential is:

    .. math::
        \phi_t(\mathbf{w}) = \sum_{i=1}^d -\frac{D_{t,i}}{} \ln(w_i)
                            + \frac{d \cdot \text{euclidean\_coef}}{2} \|w\|^2
                            + \frac{\beta}{2} w^\top A_t w,

    where:
    - :math:`D_{t,i} = \text{barrier\_coef} + \alpha \sum_{s=1}^{t-1} (1 - w_{s,i})`
      (adaptive barrier based on weight proximity),
    - :math:`d \cdot \text{euclidean\_coef} / 2 \|w\|^2` is Euclidean regularization,
    - :math:`\beta / 2 \, w^\top A_t w` with :math:`A_t = \sum_{s=1}^{t-1} g_s g_s^\top`
      is second-order curvature (Online Newton Step).

    Parameters
    ----------
    d : int
        Dimension (number of assets).
    barrier_coef : float, default=1.0
        Initial barrier coefficient.
    alpha : float, default=1.0
        Adaptation rate for barrier (weight-proximity accumulation).
    euclidean_coef : float, default=1.0
        Coefficient for Euclidean term (scaled by d in the potential).
    beta : float, default=0.1
        Coefficient for second-order curvature term.
    eps : float, default=1e-16
        Numerical stability floor.

    Returns
    -------
    CompositeMirrorMap
        The composed Ada-BARRONS mirror map:
        `AdaBarronsBarrierMap + EuclideanMap + FullQuadraticMap`.

    References
    ----------
    .. [1] Gaillard, P., Gerchinovitz, S., Hebiri, M., & Lugosi, G. (2022).
       Adapting to Misspecification in Contextual Bandits with Offline Regression
       Oracles. arXiv:2202.07574.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.optimization.online._mirror_maps import make_ada_barrons_mirror_map
    >>> d = 10
    >>> mirror = make_ada_barrons_mirror_map(d, alpha=1.0, beta=0.1)
    >>> w = np.ones(d) / d
    >>> mirror.ensure_dim(d)
    >>> z = mirror.grad_psi(w)
    >>> w_recovered = mirror.grad_psi_star(z)
    >>> np.allclose(w, w_recovered)
    True

    Notes
    -----
    The composite is **coupled** (non-diagonal) due to the `FullQuadraticMap`,
    so the inverse uses damped Newton iteration.

    The barrier coefficients adapt **based on weight proximity**, not gradient
    magnitude. You must call the barrier component's `update_state(w)` with
    the weight vector (not gradient) to accumulate proximity statistics:

    - AdaptiveLogBarrierMap: Accumulates gradient magnitude |g| (incorrect for Ada-BARRONS)
    - AdaBarronsBarrierMap: Accumulates weight proximity (1 - w) (specific for Ada-BARRONS)

    The full quadratic component's `update_state(g)` should be called with
    the gradient to accumulate outer products.
    """
    barrier = AdaBarronsBarrierMap(barrier_coef=barrier_coef, alpha=alpha, eps=eps)
    euclidean = EuclideanMap(d_coef=euclidean_coef)
    full_quad = FullQuadraticMap(beta=beta, eps=eps)

    # Ensure all components are initialized
    barrier.ensure_dim(d)
    full_quad.ensure_dim(d)

    # Compose
    composite = barrier + euclidean + full_quad
    return composite
