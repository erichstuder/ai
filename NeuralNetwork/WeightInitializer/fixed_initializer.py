from .weight_initializer import WeightInitializer
import numpy as np


class FixedInitializer(WeightInitializer):
    def __init__(self, weights: list[np.ndarray], offsets: list[np.ndarray]):
        self.weights = weights
        self.offsets = offsets
        self.current_layer = 0

    def init(self, input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
        values = self.weights[self.current_layer]
        if np.shape(values) != (output_size, input_size):
            raise ValueError(f"Expected weights shape {(output_size, input_size)}, but got {np.shape(values)} with {values}")
        
        offsets = self.offsets[self.current_layer]
        if np.shape(offsets) != (output_size,):
            raise ValueError(f"Expected offsets shape {(output_size,)}, but got {np.shape(offsets)} with {offsets}")

        self.current_layer += 1
        return (values, offsets)
