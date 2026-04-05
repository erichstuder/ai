from abc import ABC, abstractmethod


class IWeightInitializer(ABC):
    @abstractmethod
    def init(self, input_size, output_size):
        pass
