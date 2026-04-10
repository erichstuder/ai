from abc import ABC, abstractmethod
import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def init(self, input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
        pass
