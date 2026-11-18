
"""Vicevrstva sit jako retezeni vrstev - s doprednym i zpetnym pruchodem."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.activations import Sigmoid, make_activation
from src.linear import Linear
from src.neuron import Neuron


class Sequential:

    def __init__(self, layers: list) -> None:
        self.layers: list[Any] = layers
        self.io_: list[np.ndarray] | None = None
        self.z_: list[np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(self.layers) > 0
        assert isinstance(x, np.ndarray) and x.ndim == 2

        self.io_ = [x]
        self.z_ = []

        for layer in self.layers:
            z = layer.linear(self.io_[-1])
            self.z_.append(z)
            self.io_.append(layer.activation(z))

        return self.io_[-1]

    def backward(self, loss_gradient: np.ndarray) -> list[tuple[np.ndarray, np.ndarray | None]]:
        assert self.io_ is not None
        assert self.z_ is not None
        assert loss_gradient.shape == self.io_[-1].shape

        grad = loss_gradient
        gradients = []

        for k in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[k]

            delta = grad * layer.activation.derivative(self.z_[k])

            dW = self.io_[k].T @ delta

            if layer.linear.bias is not None:
                db = delta.sum(axis=0)
            else:
                db = None

            grad = delta @ layer.linear.weights.T

            gradients.append((dW, db))

        return gradients[::-1]

    def update(
        self,
        gradients: list[tuple[np.ndarray, np.ndarray | None]],
        learning_rate: float,
        ) -> None:

        assert len(gradients) == len(self.layers)

        for layer, (dW, db) in zip(self.layers, gradients):
            layer.linear.update(dW, db, learning_rate)

    def save(self, path: str) -> None:
        assert len(self.layers) > 0, "Sequential musi obsahovat alespon jednu vrstvu"

        archiv: dict[str, object] = {"pocet_vrstev": len(self.layers)}
        for k, layer in enumerate(self.layers, start=1):
            archiv[f"vrstva{k}_weights"] = layer.linear.weights
            archiv[f"vrstva{k}_activation_name"] = type(layer.activation).__name__
            if layer.linear.bias is not None:
                archiv[f"vrstva{k}_bias"] = layer.linear.bias
            if isinstance(layer.activation, Sigmoid):
                archiv[f"vrstva{k}_temperature"] = layer.activation.temperature

        np.savez(path, **archiv)

    @classmethod
    def load(cls, path: str) -> Sequential:
        assert os.path.exists(path), f"Soubor s modelem neexistuje: {path}"

        layers = []
        with np.load(path) as archiv:
            for k in range(1, int(archiv["pocet_vrstev"]) + 1):
                weights = archiv[f"vrstva{k}_weights"]
                bias = archiv[f"vrstva{k}_bias"] if f"vrstva{k}_bias" in archiv else None
                temperature = (
                    float(archiv[f"vrstva{k}_temperature"])
                    if f"vrstva{k}_temperature" in archiv else 1.0
                )
                activation = make_activation(
                    str(archiv[f"vrstva{k}_activation_name"]), temperature=temperature
                )
                layers.append(Neuron(Linear(weights, bias), activation))

        return cls(layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
