from .activation_function import IActivationFunction
import math


class Sigmoid(IActivationFunction):
    def apply(self, x):
        return 1 / (1 + math.exp(-x))
