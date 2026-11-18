
"""Inicializace vah vrstev (vzor Strategy + Factory)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WeightsInitializer(ABC):

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state
        self.rng: np.random.Generator = np.random.default_rng(random_state)

    @abstractmethod
    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        pass


class RandomUniformInit(WeightsInitializer):

    def __init__(self, low: float = -1.0, high: float = 1.0,
                 random_state: int | None = None) -> None:
        super().__init__(random_state=random_state)
        self.low = low
        self.high = high

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        assert n_inputs >= 1 and n_units >= 1, "n_inputs i n_units musi byt >= 1"
        return self.rng.uniform(self.low, self.high, size=(n_inputs, n_units))


class XavierInit(WeightsInitializer):

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        assert n_inputs >= 1 and n_units >= 1

        limit = np.sqrt(6.0 / (n_inputs + n_units))

        return self.rng.uniform(
            -limit, limit, size=(n_inputs, n_units)
        )


class HeInit(WeightsInitializer):

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        assert n_inputs >= 1 and n_units >= 1

        limit = np.sqrt(6.0 / n_inputs)

        return self.rng.uniform(
            -limit, limit, size=(n_inputs, n_units)
        )


class PyTorchDefaultInit(WeightsInitializer):

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        assert n_inputs >= 1 and n_units >= 1
        limit = 1.0 / np.sqrt(n_inputs)
        return self.rng.uniform(-limit, limit, size=(n_inputs, n_units))


_INITIALIZER_REGISTRY: dict[str, type[WeightsInitializer]] = {
    "random_uniform": RandomUniformInit,
    "xavier": XavierInit,
    "he": HeInit,
    "pytorch_default": PyTorchDefaultInit,
}


def make_weights_initializer(name: str, random_state: int | None = None) -> WeightsInitializer:
    if name not in _INITIALIZER_REGISTRY:
        povolene = ", ".join(sorted(_INITIALIZER_REGISTRY))
        raise ValueError(f"Neznamy inicializator: {name!r}. Povolene hodnoty jsou: {povolene}.")
    return _INITIALIZER_REGISTRY[name](random_state=random_state)
