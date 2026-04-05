from .weight_initializer import IWeightInitializer
import numpy as np


class RandomInitializer(IWeightInitializer):
    def init(self, input_size, output_size):
        np.random.seed(0) # Use a fixed seed for reproducibility
        return np.random.uniform(low=-0.5, high=0.5, size=(output_size, input_size))