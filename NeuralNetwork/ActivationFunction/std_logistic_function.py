from .activation_function import ActivationFunction
import numpy as np


class StdLogisticFunction(ActivationFunction):
    def _activate(self, x):
        return 1 / (1 + np.exp(-x))
