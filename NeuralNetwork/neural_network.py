from .WeightInitializer.weight_initializer import WeightInitializer
from .ActivationFunction.activation_function import ActivationFunction
import numpy as np


# Note:
# - The special input layer distributing the input data is left out of the neurons_per_layer list. It is handled separately in the code.
class NeuralNetwork:
    def __init__(self,
                 neurons_per_layer,
                 weight_initializer: WeightInitializer,
                 activation_function: ActivationFunction):
        self.neurons_per_layer = neurons_per_layer

        self.layer_weights = []
        for i in range(len(neurons_per_layer) - 1):
            input_size = neurons_per_layer[i]
            output_size = neurons_per_layer[i+1]
            self.layer_weights.append(weight_initializer.init(input_size, output_size))
        
        self.activation_function = activation_function
    
    def query(self, inputs):
        outputs = np.array(inputs)
        for weights in self.layer_weights:
            outputs = self.activation_function.apply(np.dot(weights, outputs))
        return outputs
    
    def pretty_print_weights(self):
        np.set_printoptions(precision=3, suppress=True)
        for idx, w in enumerate(self.layer_weights):
            print(f"Layer {idx} weights (shape {np.array(w).shape}):")
            print(np.array(w))
            print("-"*40)
