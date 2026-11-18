
"""Ztratove funkce (vzor Strategy): hodnota chyby a jeji gradient."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):

    @abstractmethod
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.forward(prediction, target)


class MSE(Loss):

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        assert prediction.shape == target.shape
        return float(np.mean((target - prediction) ** 2))

    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert prediction.shape == target.shape
        return 2.0 * (prediction - target) / prediction.size


class BCE(Loss):

    eps: float = 1e-12

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        assert prediction.shape == target.shape
        assert np.all((target >= 0) & (target <= 1))

        p = np.clip(prediction, self.eps, 1.0 - self.eps)

        return float(-np.mean(
            target * np.log(p) + (1.0 - target) * np.log(1.0 - p)
        ))

    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert prediction.shape == target.shape

        p = np.clip(prediction, self.eps, 1.0 - self.eps)

        return (p - target) / (p * (1.0 - p)) / prediction.size


_LOSS_REGISTRY: dict[str, type[Loss]] = {
    "mse": MSE,
    "bce": BCE,
}


def make_loss(name: str) -> Loss:
    if name not in _LOSS_REGISTRY:
        povolene = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Neznama ztratova funkce: {name!r}. Povolene hodnoty jsou: {povolene}.")
    return _LOSS_REGISTRY[name]()
