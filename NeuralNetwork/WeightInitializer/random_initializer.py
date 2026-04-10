from .weight_initializer import WeightInitializer
import numpy as np


class RandomInitializer(WeightInitializer):
    def init(self, input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(0) # Use a fixed seed for reproducibility
        weights = np.random.uniform(low=-0.5, high=0.5, size=(output_size, input_size))
        offsets = np.zeros(output_size)
        return (weights, offsets)