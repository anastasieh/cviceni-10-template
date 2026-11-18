
"""Afinni transformace (vazeny soucet vstupu plus volitelny bias) bez aktivace."""

from __future__ import annotations

import numpy as np


class Linear:

    def __init__(self, weights: np.ndarray, bias: np.ndarray | float | None = None) -> None:
        self.weights = weights
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray) and x.ndim == 2
        assert x.shape[1] == self.weights.shape[0]

        return x @ self.weights + (
            self.bias if self.bias is not None else 0.0
        )

    def update(
        self,
        weight_gradient: np.ndarray,
        bias_gradient: np.ndarray | float | None,
        learning_rate: float,
        ) -> None:

        assert weight_gradient.shape == self.weights.shape
        assert learning_rate > 0

        self.weights = self.weights - learning_rate * weight_gradient

        if self.bias is not None:
            self.bias = self.bias - learning_rate * bias_gradient

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
