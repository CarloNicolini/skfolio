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
    def __init__(self, l2_coef: float):
        self.l2_coef = l2_coef

    def __call__(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return self.l2_coef * np.sum(np.square(w - w_prev))

    def gradient(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return 2 * self.l2_coef * (w - w_prev)


class ElasticNetRegularizer(Regularizer):
    def __init__(self, l1_coef: float, l2_coef: float):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __call__(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return self.l1_coef * np.sum(np.abs(w - w_prev)) + self.l2_coef * np.sum(
            np.square(w - w_prev)
        )

    def gradient(self, w: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return self.l1_coef * np.sign(w - w_prev) + 2 * self.l2_coef * (w - w_prev)


class EntropyRegularizer(Regularizer):
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
