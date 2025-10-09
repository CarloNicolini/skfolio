import numpy as np

from skfolio.optimization.online._utils import CLIP_EPSILON


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
