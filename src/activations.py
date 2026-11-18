
"""Rodina aktivacnich funkci (vzor Strategy) pro neuron a vrstvy site.

BRANA+ (continuity gate s rozsirenim): Do metod `forward` a `output_range`
vlozte sve reseni `src/activations.py` z Cviceni 09 (signatury jsou totozne).
NEPREPISUJTE cely soubor najednou - v Cviceni 10 pribyla v kazde tride nova
metoda `derivative`, ktera je novym ukolem tohoto cviceni.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):

    @property
    @abstractmethod
    def output_range(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Step(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        output = np.where(x >= 0, 1.0, 0.0)
        assert np.all(np.isin(output, [0.0, 1.0]))
        return output

    def derivative(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return np.zeros_like(x, dtype=float)

    @property
    def output_range(self) -> tuple[float, float]:
        result = (0.0, 1.0)
        assert isinstance(result, tuple)
        assert all(isinstance(v, float) for v in result)
        return result


class Sigmoid(Activation):

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        assert self.temperature > 0
        return 1.0 / (1.0 + np.exp(-x / self.temperature))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        assert self.temperature > 0
        s = self.forward(x)
        return (1.0 / self.temperature) * s * (1.0 - s)

    @property
    def output_range(self) -> tuple[float, float]:
        result = (0.0, 1.0)
        assert isinstance(result, tuple)
        assert all(isinstance(v, float) for v in result)
        return result


class Tanh(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        t = self.forward(x)
        return 1.0 - t**2

    @property
    def output_range(self) -> tuple[float, float]:
        result = (-1.0, 1.0)
        assert isinstance(result, tuple)
        assert all(isinstance(v, float) for v in result)
        return result


class ReLU(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return (x > 0).astype(float)

    @property
    def output_range(self) -> tuple[float, float]:
        result = (0.0, np.inf)
        assert isinstance(result, tuple)
        assert result[1] == np.inf
        return result


class ReLU6(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return np.clip(x, 0.0, 6.0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return ((x > 0) & (x < 6)).astype(float)

    @property
    def output_range(self) -> tuple[float, float]:
        result = (0.0, 6.0)
        assert isinstance(result, tuple)
        assert all(isinstance(v, float) for v in result)
        return result


_ACTIVATION_REGISTRY: dict[str, type[Activation]] = {
    "Step": Step,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "ReLU": ReLU,
    "ReLU6": ReLU6,
}


def make_activation(name: str, *, temperature: float = 1.0) -> Activation:
    if name not in _ACTIVATION_REGISTRY:
        povolene = ", ".join(sorted(_ACTIVATION_REGISTRY))
        raise ValueError(f"Neznama aktivace: {name!r}. Povolene hodnoty jsou: {povolene}.")

    cls = _ACTIVATION_REGISTRY[name]
    if cls is Sigmoid:
        return Sigmoid(temperature=temperature)
    return cls()
