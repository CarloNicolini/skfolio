from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import entropy


class Regularizer(ABC):
    @abstractmethod
    def __call__(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        pass


class L1Regularizer(Regularizer):
    def __init__(self, l1_coef: float):
        self.l1_coef = l1_coef

    def __call__(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return self.l1_coef * np.sum(np.abs(w - w_prev))

    def gradient(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return self.l1_coef * np.sign(w - w_prev)


class L2Regularizer(Regularizer):
    def __init__(self, l2_coef: float, regularize_turnover: bool = False):
        self.l2_coef = l2_coef
        self.regularize_turnover = regularize_turnover

    def __call__(self, w: np.ndarray, w_prev: np.ndarray | None = None) -> np.ndarray:
        if self.regularize_turnover:
            if w_prev is None:
                raise ValueError("w_prev must be provided for turnover regularization")
            return self.l2_coef * np.sum(np.square(w - w_prev))
        else:
            return self.l2_coef * np.sum(np.square(w))

    def gradient(self, w: np.ndarray, w_prev: np.ndarray | None = None) -> np.ndarray:
        if self.regularize_turnover:
            if w_prev is None:
                raise ValueError("w_prev must be provided for turnover regularization")
            return 2 * self.l2_coef * (w - w_prev)
        else:
            return 2 * self.l2_coef * w


class NegEntropyRegularizer(Regularizer):
    def __call__(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return entropy(w)

    def gradient(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return np.log(w) + 1


class KLDivRegularizer(Regularizer):
    def __call__(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return entropy(w, w_prev)

    def gradient(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 1e-10)
        w_prev = np.maximum(w_prev, 1e-10)
        return np.log(w) - np.log(w_prev) + 1
