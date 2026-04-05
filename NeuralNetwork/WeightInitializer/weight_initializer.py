from abc import ABC, abstractmethod


class IWeightInitializer(ABC):
    @abstractmethod
    def init(self, x):
        pass
