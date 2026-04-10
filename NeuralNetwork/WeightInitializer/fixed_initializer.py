from .weight_initializer import WeightInitializer
import numpy as np


class FixedInitializer(WeightInitializer):
    def __init__(self, fixed_values):
        self.fixed_values = fixed_values
        self.current_layer = 0

    def init(self, input_size, output_size):
        values = self.fixed_values[self.current_layer]
        if np.shape(values) != (output_size, input_size):
            raise ValueError(f"Expected shape {(output_size, input_size)}, but got {np.shape(values)}")
        self.current_layer += 1
        return values
