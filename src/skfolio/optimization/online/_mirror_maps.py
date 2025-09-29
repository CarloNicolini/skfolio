from abc import ABC, abstractmethod

import numpy as np
from scipy.special import softmax


class BaseMirrorMap(ABC):
    """Abstract mirror map defining primal-dual operations."""

    @abstractmethod
    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        """Gradient ∇ψ(w): primal → dual mapping."""
        pass

    @abstractmethod
    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        """Inverse map (∇ψ)^{-1}: dual → primal."""
        pass

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Geometry-specific normalization (e.g., ensure positivity)."""
        return w


class DynamicMirrorMap(BaseMirrorMap):
    """Mirror map that may update internal state each round (for AdaGrad, etc.)."""

    def ensure_dim(self, d: int) -> None:
        """Ensure internal state is initialized for dimension d."""
        return

    def update_state(self, g: np.ndarray) -> None:
        """Update internal accumulators using gradient g (called by engine)."""
        return

    def grad_psi_star_after(self, z: np.ndarray, g: np.ndarray) -> np.ndarray:
        # Default: same as grad_psi_star (use pre-update geometry)
        return self.grad_psi_star(z)

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Geometry-specific normalization (e.g., ensure positivity)."""
        return w


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
        """Geometry-specific normalization (e.g., ensure positivity)."""
        # Softmax in grad_psi_star already projects to simplex.
        # This step ensures strict positivity and normalization if needed.
        w = np.clip(w, 1e-16, None)
        return w / np.sum(w)


class LogBarrierMap(BaseMirrorMap):
    """
    Log-barrier mirror map for the positive orthant.

    The potential function is ψ(w) = -∑ log(w_i), which ensures that the
    iterates remain strictly positive. The induced Bregman divergence is the
    Itakura-Saito divergence: Dψ(p||q) = ∑ (p_i/q_i - 1 - log(p_i/q_i)).
    """

    def __init__(self, barrier_coef: float = 1.0):
        self.barrier_coef = barrier_coef

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        """Gradient ∇ψ(w) = -barrier_coef / w."""
        return -self.barrier_coef / np.maximum(w, 1e-16)

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        # Solve -barrier_coef/w = z  => w = -barrier_coef / z (componentwise).
        # We need z to be negative for w to be positive.
        w = -self.barrier_coef / np.minimum(z, -1e-12)  # avoid dividing by 0
        w = np.clip(w, 1e-16, None)
        return w

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Geometry-specific normalization (e.g., ensure positivity)."""
        return w / np.sum(w)


class TsallisMirrorMap(BaseMirrorMap):
    """Tsallis mirror. q in (0,1) typically."""

    def __init__(self, q: float = 0.7):
        if q <= 0:
            raise ValueError("q must be positive.")

        self.q = float(q)
        self._is_entropic_limit = np.isclose(self.q, 1.0)

        if self._is_entropic_limit:
            self._entropy_map = EntropyMirrorMap()
        elif self.q > 1.0:
            raise ValueError(
                f"q={q} > 1 is not supported by this implementation due to numerical "
                "instability with log-wealth objective. Use q in (0, 1], where q=1 "
                "corresponds to Entropic Mirror Descent (EG)."
            )

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
        # Solve z = q/(1-q) * w^(q-1) => w = (z * (1-q)/q)^(1/(q-1))
        # For 0 < q < 1, 1-q > 0. With our loss, z > 0, so tmp > 0.
        tmp = z * (1.0 - q) / self.q
        w = np.maximum(tmp, 1e-16) ** (1.0 / (q - 1.0))
        return w

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        """Geometry-specific normalization (e.g., ensure positivity)."""
        if self._is_entropic_limit:
            return self._entropy_map.project_geom(w)

        w = np.clip(w, 1e-16, None)
        s = np.sum(w)
        if s > 1e-15:
            return w / s

        # Fallback to uniform if sum is too small
        return np.ones_like(w) / w.shape[0]


class AdaptiveMahalanobisMap(DynamicMirrorMap):
    """
    Diagonal adaptive quadratic mirror: psi_t(w) = 0.5 * w^T H_t w
    where H_t = diag(h_{t,i}) with h_{t,i} = sqrt(sum_{s<=t} g_{s,i}^2) + eps
    This reproduces AdaGrad-like updates in FTRL form.
    """

    def __init__(self, d: int | None = None, eps: float = 1e-8):
        self.eps = float(eps)
        self._Gsq = np.zeros(d, dtype=float) if d is not None else np.array([])
        self._d = d

    def _ensure(self, d: int):
        if self._Gsq.size == 0:
            self._Gsq = np.zeros(d, dtype=float)
            self._d = d

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
        base = self._Gsq + g * g
        h = np.sqrt(base)
        if self.eps > 0:
            # per-coordinate safety when base == 0 exactly
            zero = h <= 0.0
            if np.any(zero):
                h = h.copy()
                h[zero] = np.sqrt(self.eps)
        return h

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
    """Adaptive Log-Barrier (for AdaBARRONS).

    Regularizer: psi_t(w) = - sum_i D_{t,i} log(w_i),
    with D_{t,i} accumulating |g_{s,i}|. After-geometry uses D_t + |g_t|.
    """

    def __init__(self, d: int | None = None, eps: float = 1e-12):
        self.eps = float(eps)
        self._d = d
        self._D: np.ndarray = (
            np.zeros(d, dtype=float) if d is not None else np.array([])
        )

    def _ensure(self, d):
        if self._D.size == 0:
            self._D = np.zeros(d, dtype=float)
            self._d = d

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
        w = np.clip(w, 1e-16, None)
        return w

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
        w = np.clip(w, 1e-16, None)
        return w

    def project_geom(self, w: np.ndarray) -> np.ndarray:
        return w / np.sum(w)


class AdaptiveVariationMap(DynamicMirrorMap):
    """
    Variation-adaptive diagonal quadratic mirror:
    psi_t(w) = 0.5 * w^T H_t w, with H_t = diag(h_{t,i}),
    and h_{t,i} = sqrt(sum_{s<=t} (g_{s,i} - g_{s-1,i})^2 + eps).

    This geometry tracks gradient variation rather than magnitude, as used in
    SWORD-Var. The 'after' variants incorporate the current round's delta
    before applying grad_psi_star to maintain the OMD/FTRL equivalence.
    """

    def __init__(self, d: int | None = None, eps: float = 1e-12):
        self.eps = float(eps)
        self._d = d
        self._DeltaSq = np.zeros(d, dtype=float) if d is not None else np.array([])
        self._g_last: np.ndarray | None = None

    def _ensure(self, d: int):
        if self._DeltaSq.size == 0:
            self._DeltaSq = np.zeros(d, dtype=float)
            self._d = d

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
        base = self._DeltaSq + delta * delta
        h = np.sqrt(base + self.eps)
        return h

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
