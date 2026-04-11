from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    def apply(self, x):
        x = np.array(x)
        return self._activate(x)

    @abstractmethod
    def _activate(self, x):
        pass
