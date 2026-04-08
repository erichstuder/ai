from abc import ABC, abstractmethod


class WeightInitializer(ABC):
    @abstractmethod
    def init(self, input_size, output_size):
        pass
