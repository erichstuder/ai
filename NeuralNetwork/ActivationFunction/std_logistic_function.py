from .activation_function import IActivationFunction
import numpy as np


class StdLogisticFunction(IActivationFunction):
    def apply(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
