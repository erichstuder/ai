from .activation_function import ActivationFunction
import numpy as np


class StdLogisticFunction(ActivationFunction):
    def apply(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
