from .activation_function import ActivationFunction
import numpy as np


class ReLU(ActivationFunction):
    def _activate(self, x):
        return np.maximum(0, x)
