import numpy as np

from skfolio.optimization.online._utils import CLIP_EPSILON


def l1median_vazhz_vec(X, maxiter=200, tol=1e-9, zerotol=1e-15, medIn=None):
    """
    Compute the L1 (geometric) median of X using the Vardi-Zhang (Weiszfeld-type) algorithm.
    Vectorized implementation (no explicit loop over samples).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data points.
    maxiter : int, optional
        Maximum number of iterations (default: 200).
    tol : float, optional
        Convergence tolerance (default: 1e-9).
    zerotol : float, optional
        Distance below which points are considered coincident (default: 1e-15).
    medIn : ndarray of shape (n_features,), optional
        Initial median estimate (default: coordinate-wise median of X).

    Returns
    -------
    y : ndarray of shape (n_features,)
        Estimated L1 (geometric) median.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    y = np.median(X, axis=0) if medIn is None else np.asarray(medIn, dtype=float)

    for _ in range(maxiter):
        diffs = X - y  # shape: (n, d)
        dists = np.linalg.norm(diffs, axis=1)  # shape: (n,)

        # Mask for nonzero distances
        mask = dists >= zerotol
        if not np.any(mask):
            break

        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / dists[mask]

        # Weighted averages
        Tnum = np.sum(X * inv_dists[:, None], axis=0)
        Tden = np.sum(inv_dists)
        T = Tnum / Tden if Tden != 0 else np.zeros(d)

        # Compute R and yita
        R = np.sum(diffs[mask] * inv_dists[mask, None], axis=0)
        yita = np.any(~mask).astype(float)

        Rnorm = np.linalg.norm(R)
        r = min(1.0, yita / Rnorm) if Rnorm != 0 else 0.0

        Ty = (1 - r) * T + r * y

        # Convergence test
        iterdis = np.linalg.norm(Ty - y, 1) - tol * np.linalg.norm(y, 1)
        y = Ty
        if iterdis <= 0:
            break

    return y


class BaseReversionPredictor:
    def reset(self, d: int) -> None:
        self._d = d

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OLMAR1Predictor(BaseReversionPredictor):
    """OLMAR-SMA moving-average reversion predictor."""

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1.")
        self.window = int(window)
        self._history: list[np.ndarray] = []

    def reset(self, d: int) -> None:
        super().reset(d)
        self._history = []

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        x_t = np.asarray(x_t, dtype=float)
        self._history.append(x_t)
        T = len(self._history)
        W = self.window

        if T < W + 1:
            return self._history[-1].copy()

        recent = np.stack(self._history[-W:], axis=0)
        # Reverse order and compute cumulative products in one step
        recent = recent[::-1, :]
        # Vectorized computation: clip, cumprod, invert, and mean
        cumprods = np.cumprod(np.maximum(recent, CLIP_EPSILON), axis=0)
        return np.mean(1.0 / cumprods, axis=0)


class OLMAR2Predictor(BaseReversionPredictor):
    """OLMAR-EWMA exponential-smoothing reversion predictor."""

    def __init__(self, alpha: float = 0.5):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self._phi: np.ndarray | None = None

    def reset(self, d: int) -> None:
        super().reset(d)
        self._phi = np.ones(d, dtype=float)

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        if self._phi is None:
            self._phi = np.ones_like(x_t)
        self._phi = self.alpha * np.ones_like(x_t) + (1.0 - self.alpha) * (
            self._phi / np.maximum(x_t, CLIP_EPSILON)
        )
        return self._phi.copy()


class RMRPredictor(BaseReversionPredictor):
    """RMR L1-median reversion predictor.

    Uses the geometric median (L1-median) of recent price vectors
    as a robust predictor, less sensitive to outliers than OLMAR's
    arithmetic mean.

    Parameters
    ----------
    window : int, default=5
        Window size for historical price vectors.
    max_iter : int, default=200
        Maximum iterations for L1-median computation.
    tolerance : float, default=1e-9
        Convergence tolerance for L1-median algorithm.

    References
    ----------
    Huang, D., Zhou, J., Li, B., Hoi, S. C. H., & Zhou, S. (2013).
    Robust Median Reversion Strategy for On-Line Portfolio Selection.
    IJCAI.
    """

    def __init__(self, window: int = 5, max_iter: int = 200, tolerance: float = 1e-9):
        if window < 1:
            raise ValueError("window must be >= 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0.")
        self.window = int(window)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self._history: list[np.ndarray] = []
        self._price_history: list[np.ndarray] = []

    def reset(self, d: int) -> None:
        super().reset(d)
        self._history = []
        self._price_history = []

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        x_t = np.asarray(x_t, dtype=float)
        self._history.append(x_t)

        # Reconstruct prices (cumulative product)
        if not self._price_history:
            p_t = np.ones_like(x_t)
        else:
            p_t = self._price_history[-1] * x_t
        self._price_history.append(p_t)

        T = len(self._price_history)
        W = self.window

        # Cold-start: match OLMAR-1 behavior
        if T < W + 1:
            return self._history[-1].copy()

        # Compute L1-median of recent W price vectors
        recent_prices = np.stack(self._price_history[-W:], axis=0)
        median_price = l1median_vazhz_vec(
            recent_prices, maxiter=self.max_iter, tol=self.tolerance
        )

        # Convert to price relative: median_price / current_price
        p_current = self._price_history[-1]
        phi = median_price / np.maximum(p_current, CLIP_EPSILON)
        return phi


class LastGradPredictor:
    """Predictor that returns the last observed gradient.

    This implements the "smooth prediction" strategy from online learning theory,
    where we predict that the next gradient will be the same as the current one.

    When used with optimistic OMD, this leads to regret bounds that scale with
    the temporal variation of gradients: sum_t ||g_t - g_{t-1}||^2, rather than
    sum_t ||g_t||^2.

    References
    ----------
    Chiang et al. (2012). "Online optimization with gradual variations." JMLR.
    Rakhlin & Sridharan (2013). "Online Learning with Predictable Sequences."
    """

    def __call__(
        self, t: int, last_played_x: np.ndarray | None, last_grad: np.ndarray | None
    ) -> np.ndarray:
        if last_grad is not None:
            return last_grad.copy()
        if last_played_x is not None:
            return np.zeros_like(last_played_x)
        # fallback: empty array — caller must handle shape
        return np.array([])


class SmoothPredictor:
    """Predictor that returns clipped last gradient for bounded variation.

    This implements optimistic prediction with bounded variation assumptions,
    clipping the last observed gradient to [-L, L] where L is a smoothness constant.

    When used with optimistic OMD, this is appropriate when gradients are known
    to have bounded temporal variation, leading to improved regret bounds.

    Parameters
    ----------
    smoothness_L : float, default=1.0
        Lipschitz constant bounding gradient variation (clips to [-L, L]).

    References
    ----------
    Chiang et al. (2012). "Online optimization with gradual variations." JMLR.
    Rakhlin & Sridharan (2013). "Online Learning with Predictable Sequences."
    """

    def __init__(self, smoothness_L: float = 1.0):
        self.smoothness_L = smoothness_L
        self.last_grad = None

    def __call__(
        self, t: int, last_played_x: np.ndarray | None, last_grad: np.ndarray | None
    ) -> np.ndarray:
        if last_grad is not None:
            self.last_grad = last_grad.copy()
        if self.last_grad is None:
            return (
                np.zeros_like(last_played_x)
                if last_played_x is not None
                else np.array([])
            )
        # Predict bounded variation: clip to [-L, L]
        return np.clip(self.last_grad, -self.smoothness_L, self.smoothness_L)
