from .activation_function import ActivationFunction
import numpy as np


class ReLU(ActivationFunction):
    def apply(self, x):
        x = np.array(x)
        return np.maximum(0, x)
