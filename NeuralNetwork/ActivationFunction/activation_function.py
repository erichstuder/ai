from abc import ABC, abstractmethod


class IActivationFunction(ABC):
    @abstractmethod
    def apply(self, x):
        pass
