from abc import ABC, abstractmethod
import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def init(self, input_size, output_size):
        pass
