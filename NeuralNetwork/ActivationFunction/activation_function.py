from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def apply(self, x):
        pass
